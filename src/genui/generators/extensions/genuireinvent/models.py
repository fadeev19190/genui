# genui/generators/extensions/genuireinvent/models.py

from __future__ import annotations
import pkgutil
import inspect
import importlib

import os
import shutil
import subprocess
import tempfile
from typing import Tuple
import re

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import models

from genui.compounds.models import MolSet
from genui.models.models import Model, ModelFile, TrainingStrategy, ValidationStrategy
from genui.projects.models import DataSet

from reinvent.runmodes.RL.memories.diversity_filter import DiversityFilter
import reinvent.runmodes.RL.memories as mem

# ───────────────────────────────────────────────────────────────────────────────
# Hard-coded prior: adjust this absolute path to your machine if needed.
# ───────────────────────────────────────────────────────────────────────────────
PRIOR_ABS = "/Users/artemfadeev/diplom/genui/files/checkpoints/prior/reinvent.prior"

_BEST_EPOCH_RE = re.compile(
    r"Best\s+validation\s+loss\s*\(\s*(?P<loss>[-+]?(\d+(\.\d+)?|\.\d+))\s*\)\s*was\s*at\s*epoch\s*(?P<epoch>\d+)",
    re.IGNORECASE,
)

def _parse_best_from_log(text: str) -> tuple[int | None, float | None]:
    if not text:
        return None, None
    m = _BEST_EPOCH_RE.search(text)
    if not m:
        return None, None
    return int(m.group("epoch")), float(m.group("loss"))


# ───────────────────────────────────────────────────────────────────────────────
# Small helper: overwrite a hashed ModelFile in-place
# ───────────────────────────────────────────────────────────────────────────────
def _overwrite_filefield(mf: ModelFile, data: bytes | str, *, filename: str | None = None) -> None:
    """
    Overwrite an existing FileField content while keeping its hashed location.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    current_rel = mf.file.name  # e.g. "models/ReinventNet18_project34_<hash>_aux.toml"
    if not filename:
        filename = os.path.basename(current_rel)

    # Remove old content first to avoid orphaned blobs on some storages
    try:
        mf.file.storage.delete(current_rel)
    except Exception:
        pass

    mf.file.save(filename, ContentFile(data), save=True)

def _bemis_murcko(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if not m: return ""
    core = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(core, isomericSmiles=False) if core else ""

def _split_indices(n, frac, seed):
    r = random.Random(seed)
    idx = list(range(n))
    r.shuffle(idx)
    cut = max(1, int(n * frac))
    valid = set(idx[:cut])
    train = [i for i in idx if i not in valid]
    valid = list(valid)
    return train, valid


class _ReinventCLIModel:
    """
    Tiny façade so the builder/algorithm API works while training happens via CLI.
    """
    def __init__(self, net: "ReinventNet"):
        self._net = net
        self._checkpoint: str | None = None

    def fit(self, X=None, y=None):
        self._net.prepareData()
        self._checkpoint = self._net.run_transfer_learning(device="cpu")
        return self

    def loadStatesFromFile(self, path: str):
        return self

    def getModel(self):
        return {"checkpoint": self._checkpoint}


class ReinventNet(Model):
    # AUX notes (DrugEx-style)
    CORPUS_FULL_NOTE     = "reinvent_corpus_full"     # full cleaned .smi for CLI
    CORPUS_PREVIEW_NOTE  = "reinvent_corpus_preview"  # short preview for UI/tests
    TOML_FILE_NOTE       = "reinvent_tl_toml"         # generated TL config
    TRAIN_LOG_NOTE       = "reinvent_train_log"       # TL stdout/stderr log
    CHECKPOINT_FILE_NOTE = "reinvent_tl_checkpoint"   # where REINVENT writes
    CORPUS_TRAIN_NOTE = "reinvent_corpus_train"
    CORPUS_VALID_NOTE = "reinvent_corpus_valid"

    molset = models.ForeignKey(MolSet, on_delete=models.CASCADE, null=True)
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True)

    # ── AUX getters (create the record lazily with empty payload) ──────────────
    def _get_or_create_aux(self, note: str, filename: str) -> ModelFile:
        mf = self.files.filter(kind=ModelFile.AUXILIARY, note=note).first()
        if mf is None:
            mf = ModelFile.create(self, filename, ContentFile(b""), note=note)
        return mf

    @property
    def corpusFileTrain(self):  # backwards-compat alias
        return self.corpusTrainFile

    @property
    def corpusTrainFile(self) -> ModelFile:
        return self._get_or_create_aux(self.CORPUS_TRAIN_NOTE, f"corpus_train_{self.pk}.smi")

    @property
    def corpusValidFile(self) -> ModelFile:
        return self._get_or_create_aux(self.CORPUS_VALID_NOTE, f"corpus_valid_{self.pk}.smi")

    @property
    def corpusFullFile(self) -> ModelFile:
        # Full cleaned corpus consumed by REINVENT CLI
        return self._get_or_create_aux(self.CORPUS_FULL_NOTE, f"corpus_full_{self.pk}.smi")

    @property
    def corpusPreviewFile(self) -> ModelFile:
        # Optional short preview for UI/tests
        return self._get_or_create_aux(self.CORPUS_PREVIEW_NOTE, f"corpus_preview_{self.pk}.smi")

    @property
    def tlTomlFile(self) -> ModelFile:
        return self._get_or_create_aux(self.TOML_FILE_NOTE, f"tl_reinvent_{self.pk}.toml")

    @property
    def trainLogFile(self) -> ModelFile:
        return self._get_or_create_aux(self.TRAIN_LOG_NOTE, f"reinvent_training_{self.pk}.log")

    @property
    def checkpointFile(self) -> ModelFile:
        # We keep the checkpoint managed as an AUX file too
        return self._get_or_create_aux(self.CHECKPOINT_FILE_NOTE, f"reinvent_{self.pk}.model")

    # Backwards-compat convenience (tests may call this):
    def get_clean_corpus_path(self) -> str:
        return self.corpusFullFile.path

    # ── Prior path (hard-coded) ────────────────────────────────────────────────
    def get_prior_path(self) -> str:
        if not os.path.isfile(PRIOR_ABS):
            raise FileNotFoundError(f"REINVENT prior not found at: {PRIOR_ABS}")
        return PRIOR_ABS

    # ── Clean corpus preparation (hashed AUX only) ─────────────────────────────
    def prepareData(self) -> Tuple[ModelFile, ModelFile]:
        """
        Clean SMILES via reinvent.datapipeline and write:
          - Full cleaned corpus directly to corpusFullFile.path (hashed in media/)
          - Short preview (first 1000 lines) into corpusPreviewFile (hashed)
        """
        if not self.molset:
            raise RuntimeError(f"No MolSet attached to {self}.")

        # Decide input for datapipeline
        input_path = None
        if getattr(self.molset, "files", None) and self.molset.files.exists():
            f = self.molset.files.first()
            if f and getattr(f, "file", None):
                input_path = f.file.path

        # If needed, emit a temporary TSV with a SMILES header
        temp_in = None
        if not input_path:
            with tempfile.NamedTemporaryFile(prefix=f"reinvent_raw_{self.pk}_", suffix=".smi.tsv", delete=False) as tf:
                temp_in = tf.name
            with open(temp_in, "w", encoding="utf-8") as w:
                w.write("SMILES\n")
                for s in self.molset.allSmiles:
                    w.write(s + "\n")
            input_path = temp_in

        out_full_path = self.corpusFullFile.path  # hashed media path

        try:
            from reinvent.datapipeline import preprocess
        except Exception as e:
            if temp_in:
                try:
                    os.remove(temp_in)
                except OSError:
                    pass
            raise RuntimeError("reinvent.datapipeline.preprocess is required.") from e

        cfg_text = f"""\
        input_csv_file = "{input_path}"
        smiles_column = "SMILES"
        separator = "\\t"
        output_smiles_file = "{out_full_path}"

        [filter]
        elements = []
        transforms = ["standard"]
        inchi_key_deduplicate = true
        """
        with tempfile.NamedTemporaryFile(prefix=f"reinvent_preprocess_{self.pk}_",
                                         suffix=".toml", delete=False) as tf:
            cfg_path = tf.name
        try:
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write(cfg_text)
            args = type("Args", (), {"config_filename": cfg_path, "log_filename": None})
            preprocess.main(args)
        finally:
            try:
                os.remove(cfg_path)
            except OSError:
                pass
            if temp_in:
                try:
                    os.remove(temp_in)
                except OSError:
                    pass

        # 2) Read CLEANED full corpus and split
        vs = getattr(self, "validationStrategy", None)
        method = (getattr(vs, "split_method", None) or "random").lower()
        frac = max(0.0, min(0.9, float(getattr(vs, "valid_fraction", 0.1))))
        seed = int(getattr(vs, "random_seed", 1337))
        cutoff = getattr(vs, "temporal_cutoff", None)
        max_valid = int(getattr(vs, "validSetSize", 0)) or None

        with open(out_full_path, "r", encoding="utf-8") as fh:
            smiles = [ln.strip() for ln in fh if ln.strip()]
        # Guard empty corpus before splitting. If the preprocessor yields 0–1 lines, your split can produce empty files.
        if not smiles:
            raise RuntimeError(f"Cleaned corpus is empty at {out_full_path}.")
        if len(smiles) == 1:
            _overwrite_filefield(self.corpusTrainFile, smiles[0] + "\n",
                                 filename=os.path.basename(self.corpusTrainFile.file.name))
            _overwrite_filefield(self.corpusValidFile, "",
                                 filename=os.path.basename(self.corpusValidFile.file.name))
            # preview build as you do…
            return self.corpusTrainFile, self.corpusValidFile

        if method == "scaffold":
            buckets = {}
            for s in smiles:
                scf = _bemis_murcko(s) or f"NOSCAF_{hash(s) % 10_000_000}"
                buckets.setdefault(scf, []).append(s)
            rng = random.Random(seed)
            scaf_ids = list(buckets.keys());
            rng.shuffle(scaf_ids)
            valid_target = max(1, int(len(smiles) * frac))
            train, valid, acc = [], [], 0
            for scf in scaf_ids:
                grp = buckets[scf]
                if acc < valid_target:
                    valid.extend(grp);
                    acc += len(grp)
                else:
                    train.extend(grp)
        elif method == "temporal" and cutoff:
            raise NotImplementedError("Temporal split needs SMILES->date mapping in MolSet.")
        else:
            tr_idx, va_idx = _split_indices(len(smiles), frac, seed)
            train = [smiles[i] for i in tr_idx]
            valid = [smiles[i] for i in va_idx]

        if max_valid is not None and len(valid) > max_valid:
            valid = valid[:max_valid]
        if not train:
            move_n = max(1, len(valid) // 2)
            train, valid = valid[:move_n], valid[move_n:]

        _overwrite_filefield(self.corpusTrainFile, "\n".join(train) + "\n",
                             filename=os.path.basename(self.corpusTrainFile.file.name))
        _overwrite_filefield(self.corpusValidFile, "\n".join(valid) + "\n",
                             filename=os.path.basename(self.corpusValidFile.file.name))

        # 3) Build preview from CLEANED corpus
        head = []
        with open(out_full_path, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i >= 1000: break
                s = ln.strip()
                if s: head.append(s)
        preview_text = ("\n".join(head) + "\n") if head else ""
        _overwrite_filefield(self.corpusPreviewFile, preview_text,
                             filename=os.path.basename(self.corpusPreviewFile.file.name))

        # 4) Return actual train/valid
        return self.corpusTrainFile, self.corpusValidFile

    # ── TOML (hashed AUX only) ────────────────────────────────────────────────
    def build_tl_toml(self, *, device: str = "cpu") -> str:
        ts = self.trainingStrategy
        if not isinstance(ts, ReinventNetTraining):
            raise RuntimeError("ReinventNetTraining required.")

        prior = self.get_prior_path()
        out_path = self.checkpointFile.path   # REINVENT will write here
        train = self.corpusTrainFile.path
        valid = self.corpusValidFile.path
        sbs = max(100, ts.sample_batch_size)
        tb_dir = os.path.join(settings.MEDIA_ROOT, "models", f"tb_TL_{self.pk}")
        os.makedirs(tb_dir, exist_ok=True)

        body = f"""\
run_type = "transfer_learning"
device = "{device}"
tb_logdir = "{tb_dir}"

[parameters]
num_epochs = {ts.epochs}
save_every_n_epochs = {ts.save_every_n_epochs}
batch_size = {ts.batch_size}
sample_batch_size = {sbs}

input_model_file = "{prior}"
smiles_file = "{train}"
validation_smiles_file = "{valid}"
output_model_file = "{out_path}"
"""

        _overwrite_filefield(
            self.tlTomlFile,
            body,
            filename=os.path.basename(self.tlTomlFile.file.name),
        )
        return self.tlTomlFile.path

    @staticmethod
    def _pick_best_checkpoint(tb_dir: str) -> tuple[int, float] | None:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(tb_dir);
            ea.Reload()
            vals = ea.Scalars("valid/nll") or ea.Scalars("validation/nll")
            if not vals: return None
            best = min(vals, key=lambda x: x.value)
            return (best.step, best.value)
        except Exception:
            return None

    def get_active_checkpoint_path(self) -> str:
        """
        Returns the canonical checkpoint to load for the next stage.
        Prefer the selected best-epoch (copied into checkpointFile.path).
        """
        p = self.checkpointFile.path
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Active checkpoint missing at {p}.")
        return p

    # ── TL run (hashed AUX only) ───────────────────────────────────────────────
    def run_transfer_learning(self, *, device: str = "cpu") -> str:
        toml_path = self.build_tl_toml(device=device)
        out_path = self.checkpointFile.path

        reinvent_bin = (getattr(settings, "REINVENT_BIN", None)
                        or os.environ.get("REINVENT_BIN")
                        or shutil.which("reinvent"))
        if not reinvent_bin:
            raise RuntimeError("REINVENT binary not found. Set settings.REINVENT_BIN or $REINVENT_BIN.")

        cmd = [reinvent_bin, toml_path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=settings.BASE_DIR)
        lines = [ln for ln in (proc.stdout or [])]
        rc = proc.wait()

        log_text = f"[CMD] {' '.join(cmd)}\n{''.join(lines)}"
        _overwrite_filefield(self.trainLogFile, log_text,
                             filename=os.path.basename(self.trainLogFile.file.name))

        if rc != 0:
            raise RuntimeError(f"REINVENT TL failed (exit={rc}). See TOML: {toml_path}")

        # optional: swap to best epoch checkpoint based on TensorBoard
        tb_dir = os.path.join(settings.MEDIA_ROOT, "models", f"tb_TL_{self.pk}")
        best_epoch, best_loss = _parse_best_from_log(log_text)

        if best_epoch is not None and best_loss is not None:
            ts = self.trainingStrategy
            ts.best_epoch = best_epoch
            ts.best_valid_loss = best_loss
            ts.save(update_fields=["best_epoch", "best_valid_loss"])

        return out_path

    # Keep the façade so builders can call into “a model”
    def getModel(self):
        return _ReinventCLIModel(self)


class ReinventNetValidation(ValidationStrategy):
    validSetSize = models.IntegerField(default=10000)  # keep if you want “cap”
    split_method = models.CharField(
        max_length=16, default="random",  # "random" | "scaffold" | "temporal"
    )
    valid_fraction = models.FloatField(default=0.1)  # ignored if validSetSize used
    random_seed = models.IntegerField(default=1337)
    temporal_cutoff = models.CharField(max_length=32, null=True, blank=True)  # e.g. "2024-06-01"


class ReinventNetTraining(TrainingStrategy):
    epochs = models.IntegerField(default=10)
    batch_size = models.IntegerField(default=64)
    save_every_n_epochs = models.IntegerField(default=1)
    sample_batch_size = models.IntegerField(default=100)

    best_epoch = models.IntegerField(null=True, blank=True)
    best_valid_loss = models.FloatField(null=True, blank=True)

    def processMetaData(self, metadata: dict):
            self.epochs = metadata.get("epochs", self.epochs)
            self.batch_size = metadata.get("batch_size", self.batch_size)
            self.sample_batch_size = metadata.get("sample_batch_size", self.sample_batch_size)
            self.save()


# =====================================================================
#  STAGED LEARNING MODULE  (AFTER TL CODE)
# =====================================================================

# Hardcoded unwanted SMARTS from REINVENT supplement
UNWANTED_SMARTS_DEFAULT = [
    "[*;r8]", "[*;r9]", "[*;r10]", "[*;r11]", "[*;r12]", "[*;r13]", "[*;r14]",
    "[*;r15]", "[*;r16]", "[*;r17]",
    "[#8][#8]", "[#6;+]", "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]", "[#7;!n][#7;!n]",
    "C#C", "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]",
]


class ReinventEnvironmentHelper:

    @staticmethod
    def get_diversity_filters():
        import reinvent.runmodes.RL.memories as mem
        from reinvent.runmodes.RL.memories.diversity_filter import DiversityFilter
        results = []

        for _, modname, _ in pkgutil.walk_packages(mem.__path__, prefix="reinvent.runmodes.RL.memories."):
            lname = modname.lower()
            if any(x in lname for x in ["murcko", "topological", "similarity", "penalize"]):
                module = importlib.import_module(modname)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, DiversityFilter) and obj is not DiversityFilter:
                        results.append(name)

        return sorted(set(results))

    @staticmethod
    def get_learning_strategies():
        return ["dap"]


def df_choices():
    return [(x, x) for x in ReinventEnvironmentHelper.get_diversity_filters()]


# =====================================================================
# ENVIRONMENT
# =====================================================================

class ReinventDiversityFilter(models.Model):
    type = models.CharField(max_length=128, choices=df_choices)
    bucket_size = models.IntegerField(default=25)
    minscore = models.FloatField(default=0.4)
    minsimilarity = models.FloatField(default=0.4)
    penalty_multiplier = models.FloatField(default=0.5)

    def to_reinvent(self):
        d = {"type": self.type, "bucket_size": self.bucket_size, "minscore": self.minscore}
        if self.type == "ScaffoldSimilarity":
            d["minsimilarity"] = self.minsimilarity
        if self.type == "PenalizeSameSmiles":
            d["penalty_multiplier"] = self.penalty_multiplier
        return d


class ReinventEnvironment(models.Model):
    name = models.CharField(max_length=255)

    prior_model = models.ForeignKey(
        ModelFile, on_delete=models.PROTECT, related_name="reinvent_prior_files"
    )
    agent_model = models.ForeignKey(
        ModelFile, on_delete=models.PROTECT, related_name="reinvent_agent_files"
    )

    diversity_filter = models.ForeignKey(ReinventDiversityFilter, null=True, blank=True, on_delete=models.SET_NULL)
    reward_scheme = models.ForeignKey("ReinventEnvironmentScores", null=True, blank=True, on_delete=models.SET_NULL)

    inception_smiles = models.ForeignKey(ModelFile, null=True, blank=True, on_delete=models.SET_NULL)
    inception_memory_size = models.IntegerField(default=0)
    inception_sample_size = models.IntegerField(default=0)


# =====================================================================
# SCORING
# =====================================================================

class ScoreModifier(models.Model):
    """
    DrugEx-like modifiers that export REINVENT-compatible transforms.
    """

    modifier_type = models.CharField(max_length=32, choices=[
        ("ClippedScore", "ClippedScore"),
        ("SmoothHump", "SmoothHump"),
    ])

    # DrugEx parameters
    upper = models.FloatField(null=True, blank=True)
    lower = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    low = models.FloatField(null=True, blank=True)
    smooth = models.BooleanField(default=False)
    sigma = models.FloatField(null=True, blank=True)

    def to_reinvent_transform(self):

        # ClippedScore (smooth=False) → double sigmoid
        if self.modifier_type == "ClippedScore" and not self.smooth:
            return {
                "type": "double_sigmoid",
                "low": self.lower,
                "high": self.upper,
                "coef_div": float(self.upper - self.lower),
                "coef_si": 20,
                "coef_se": 20,
            }

        # ClippedScore (smooth=True) → mirrored sigmoid
        if self.modifier_type == "ClippedScore" and self.smooth:
            span = float(self.upper - self.lower)
            k = 1.0 / (span + 1e-9)
            return {
                "type": "reverse_sigmoid",
                "low": self.lower,
                "high": self.upper,
                "k": k,
            }

        # SmoothHump → double sigmoid hump
        if self.modifier_type == "SmoothHump":
            return {
                "type": "double_sigmoid",
                "low": self.lower,
                "high": self.upper,
                "coef_div": float(self.upper - self.lower),
                "coef_si": int(self.sigma * 20),
                "coef_se": int(self.sigma * 20),
            }

        return None


class ReinventEnvironmentScores(models.Model):
    aggregation_type = models.CharField(
        max_length=64,
        choices=[
            ("geometric_mean", "geometric_mean"),
            ("weighted_arithmetic_mean", "weighted_arithmetic_mean"),
        ]
    )


class ScoringMethod(models.Model):
    name = models.CharField(max_length=255)
    weight = models.FloatField(default=1.0)
    scheme = models.ForeignKey(
        ReinventEnvironmentScores,
        on_delete=models.CASCADE,
        related_name="%(class)s_set"
    )
    modifier = models.ForeignKey(ScoreModifier, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        abstract = True

    def build_transform(self):
        return self.modifier.to_reinvent_transform() if self.modifier else None


class PropertyScorer(ScoringMethod):
    property_name = models.CharField(max_length=64)

class GenUIModelScorer(ScoringMethod):
    model = models.ForeignKey(Model, on_delete=models.PROTECT)

class UnwantedSmartsScorer(ScoringMethod):
    enabled = models.BooleanField(default=True)

    def load_patterns(self):
        """
        Return the SMARTS patterns to use for custom alerts.

        For now we just use a static default list (UNWANTED_SMARTS_DEFAULT).
        This can be extended later to load user-defined SMARTS from another
        model or file if needed.
        """
        return UNWANTED_SMARTS_DEFAULT


# =====================================================================
# AGENT
# =====================================================================

def learning_strategy_choices():
    return [(x, x) for x in ReinventEnvironmentHelper.get_learning_strategies()]


class ReinventAgentTraining(models.Model):
    batch_size = models.IntegerField(default=64)
    unique_sequences = models.BooleanField(default=True)
    randomize_smiles = models.BooleanField(default=True)
    tb_isim = models.BooleanField(default=False)

    use_checkpoint = models.BooleanField(default=False)
    purge_memories = models.BooleanField(default=False)

    summary_csv_prefix = models.CharField(max_length=128, default="reinvent")

    learning_type = models.CharField(max_length=32, choices=learning_strategy_choices, default="dap")
    sigma = models.FloatField(default=128.0)
    rate = models.FloatField(default=0.0001)


class ReinventAgentValidation(models.Model):
    validate_every = models.IntegerField(default=50)
    validation_dataset = models.ForeignKey(ModelFile, null=True, blank=True, on_delete=models.SET_NULL)


class ReinventAgent(models.Model):
    environment = models.ForeignKey(ReinventEnvironment, on_delete=models.PROTECT)
    training = models.ForeignKey(ReinventAgentTraining, on_delete=models.PROTECT)
    validation = models.ForeignKey(ReinventAgentValidation, null=True, blank=True, on_delete=models.SET_NULL)
    output_model = models.ForeignKey(ModelFile, null=True, blank=True, on_delete=models.SET_NULL)


# =====================================================================
# STAGED LEARNING GENERATOR
# =====================================================================

class Reinvent(models.Model):
    name = models.CharField(max_length=255)
    environment = models.ForeignKey(ReinventEnvironment, on_delete=models.PROTECT)
    agent = models.ForeignKey(ReinventAgent, on_delete=models.PROTECT)

    tb_logdir = models.CharField(max_length=255, default="tb_logs")
    json_out_config = models.CharField(max_length=255, default="_staged_learning.json")

    # ------------------------------------------------------------------
    # Simple file helpers for staged learning (no ModelFile here)
    # ------------------------------------------------------------------

    def _sl_dir(self) -> str:
        """Directory where staged-learning TOML and logs are stored."""
        from django.conf import settings  # already imported at top of file
        base = getattr(settings, "MEDIA_ROOT", ".")
        return os.path.join(base, "reinvent_sl")

    def get_toml_path(self) -> str:
        """Absolute path to this run's staged-learning TOML."""
        return os.path.join(self._sl_dir(), f"staged_learning_{self.pk}.toml")

    def get_rl_log_path(self) -> str:
        """Absolute path to this run's RL log file."""
        return os.path.join(self._sl_dir(), f"reinvent_rl_{self.pk}.log")

    # ------------------------------------------------------------------
    # Build staged-learning TOML
    # ------------------------------------------------------------------
    def build_staged_toml(self, device="cuda:0") -> str:

        env = self.environment
        train_cfg = self.agent.training

        lines = []
        lines.append('run_type = "staged_learning"')
        lines.append(f'device = "{device}"')
        lines.append(f'tb_logdir = "{self.tb_logdir}"')
        lines.append(f'json_out_config = "{self.json_out_config}"')
        lines.append("")

        # PARAMETERS
        lines.append("[parameters]")
        lines.append(f"use_checkpoint = {str(train_cfg.use_checkpoint).lower()}")
        lines.append(f'summary_csv_prefix = "{train_cfg.summary_csv_prefix}"')

        prior_file = env.prior_model
        if not prior_file:
            raise RuntimeError("No prior model file set.")
        lines.append(f'prior_file = "{prior_file.file.path}"')

        agent_file = env.agent_model
        lines.append(f'agent_file = "{agent_file.file.path}"')

        lines.append(f"batch_size = {train_cfg.batch_size}")
        lines.append(f"unique_sequences = {str(train_cfg.unique_sequences).lower()}")
        lines.append(f"randomize_smiles = {str(train_cfg.randomize_smiles).lower()}")
        lines.append(f"tb_isim = {str(train_cfg.tb_isim).lower()}")
        lines.append("")

        # LEARNING STRATEGY
        lines.append("[learning_strategy]")
        lines.append(f'type = "{train_cfg.learning_type}"')
        lines.append(f"sigma = {train_cfg.sigma}")
        lines.append(f"rate = {train_cfg.rate}")
        lines.append("")

        # DIVERSITY FILTER
        if env.diversity_filter:
            d = env.diversity_filter.to_reinvent()
            lines.append("[diversity_filter]")
            for k, v in d.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")

        # INCEPTION
        if env.inception_smiles:
            lines.append("[inception]")
            lines.append(f'smiles_file = "{env.inception_smiles.file.path}"')
            if env.inception_memory_size:
                lines.append(f"memory_size = {env.inception_memory_size}")
            if env.inception_sample_size:
                lines.append(f"sample_size = {env.inception_sample_size}")
            lines.append("")

        # STAGES ---------------------------------------------------------
        stages = list(self.stages.order_by("order"))
        if not stages:
            raise RuntimeError("Staged learning requires at least one stage.")

        for st in stages:
            lines.append("[[stage]]")

            if st.chkpt_file:
                chkpt_path = st.chkpt_file.file.path

            else:
                # simple default path under the same RL run folder
                chkpt_path = os.path.join(
                    self._sl_dir(),  # you already have this from earlier fixes
                    f"agent_stage{st.order}_run{self.pk}.chkpt",
                )
            lines.append(f'chkpt_file = "{chkpt_path}"')

            lines.append(f'termination = "{st.termination_type}"')
            lines.append(f"max_score = {st.max_score}")
            lines.append(f"min_steps = {st.min_steps}")
            lines.append(f"max_steps = {st.max_steps}")
            lines.append("")

            prefix = "stage.scoring"

            # EXTERNAL SCORING FILE -------------------------------------
            if st.scoring_source == "file":
                scheme = st.scoring_scheme
                if not scheme:
                    raise RuntimeError(
                        "Stage requires a scoring_scheme when using scoring_source='file'."
                    )

                if not st.scoring_file:
                    raise RuntimeError(
                        "Stage scoring_source='file' but scoring_file is not set."
                    )

                lines.append(f"[{prefix}]")
                lines.append(f'type = "{scheme.aggregation_type}"')
                lines.append(f'filename = "{st.scoring_file.file.path}"')
                lines.append('filetype = "toml"')
                lines.append("")
                continue

            # INLINE SCORING ---------------------------------------------
            scheme = st.scoring_scheme or env.reward_scheme
            if not scheme:
                raise RuntimeError(
                    "Inline scoring requires either stage scoring_scheme or environment.reward_scheme."
                )

            lines.append(f"[{prefix}]")
            lines.append(f'type = "{scheme.aggregation_type}"')
            lines.append("")

            # PROPERTY SCORERS
            for ps in PropertyScorer.objects.filter(scheme=scheme).order_by("id"):
                prop = ps.property_name
                lines.append(f"[[{prefix}.component]]")
                lines.append(f"[{prefix}.component.{prop}]")
                lines.append(f"[[{prefix}.component.{prop}.endpoint]]")
                lines.append(f'name = "{ps.name}"')
                lines.append(f"weight = {ps.weight}")

                tr = ps.build_transform()
                if tr:
                    for k, v in tr.items():
                        if isinstance(v, str):
                            lines.append(
                                f'{prefix}.component.{prop}.endpoint.transform.{k} = "{v}"'
                            )
                        else:
                            lines.append(
                                f"{prefix}.component.{prop}.endpoint.transform.{k} = {v}"
                            )
                lines.append("")

            # UNWANTED SMARTS
            for us in UnwantedSmartsScorer.objects.filter(scheme=scheme).order_by("id"):
                if not us.enabled:
                    continue

                patterns = us.load_patterns()

                lines.append(f"[[{prefix}.component]]")
                lines.append(f"[{prefix}.component.custom_alerts]")
                lines.append(f"[[{prefix}.component.custom_alerts.endpoint]]")
                lines.append(f'name = "{us.name}"')
                lines.append(f"weight = {us.weight}")
                lines.append("params.smarts = [")

                for p in patterns:
                    lines.append(f'  "{p}",')
                lines.append("]")
                lines.append("")

            lines.append("")  # blank line after each stage

        # END STAGES -----------------------------------------------------

        # Write TOML to a simple file under MEDIA_ROOT/reinvent_sl/
        body = "\n".join(lines) + "\n"
        sl_dir = self._sl_dir()
        os.makedirs(sl_dir, exist_ok=True)

        toml_path = self.get_toml_path()
        with open(toml_path, "w", encoding="utf-8") as fh:
            fh.write(body)

        return toml_path

    # ------------------------------------------------------------------
    # RUN STAGED LEARNING
    # ------------------------------------------------------------------
    def run_staged_learning(self, device="cuda:0") -> str:
        toml_path = self.build_staged_toml(device=device)

        reinvent_bin = (
            getattr(settings, "REINVENT_BIN", None)
            or os.environ.get("REINVENT_BIN")
            or shutil.which("reinvent")
        )

        if not reinvent_bin:
            raise RuntimeError(
                "REINVENT binary not found. Set REINVENT_BIN in settings or environment."
            )

        cmd = [reinvent_bin, toml_path]

        rl_dir = self._sl_dir()
        os.makedirs(rl_dir, exist_ok=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=rl_dir,
        )

        lines = [ln for ln in (proc.stdout or [])]
        rc = proc.wait()

        # Write log to a simple file
        sl_dir = self._sl_dir()
        os.makedirs(sl_dir, exist_ok=True)

        log_path = self.get_rl_log_path()
        log_text = f"[CMD] {' '.join(cmd)}\n{''.join(lines)}"
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(log_text)

        if rc != 0:
            raise RuntimeError(
                f"REINVENT staged learning failed (exit {rc}). See log: {log_path}"
            )

        return toml_path

# =====================================================================
# 6. STAGE MODEL
# =====================================================================

class ReinventStage(models.Model):
    generator = models.ForeignKey(Reinvent, related_name="stages", on_delete=models.CASCADE)
    order = models.IntegerField()

    chkpt_file = models.ForeignKey(
        ModelFile, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="reinvent_stage_checkpoints"
    )

    termination_type = models.CharField(max_length=64, default="simple")
    max_score = models.FloatField(default=1.0)
    min_steps = models.IntegerField(default=1)
    max_steps = models.IntegerField(default=100)

    scoring_source = models.CharField(
        max_length=16,
        choices=[("inline", "Inline"), ("file", "File")],
        default="inline"
    )

    scoring_scheme = models.ForeignKey(
        ReinventEnvironmentScores,
        null=True,
        blank=True,
        on_delete=models.SET_NULL
    )

    scoring_file = models.ForeignKey(
        ModelFile,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reinvent_stage_scoring_files"
    )

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"Stage {self.order} (Generator {self.generator_id})"

# =====================================================================
# 7. PERFORMANCE LOGGING
# =====================================================================

class ModelPerformanceReinvent(models.Model):
    agent = models.ForeignKey(ReinventAgent, on_delete=models.CASCADE)
    step = models.IntegerField()
    stage_index = models.IntegerField()

    avg_score = models.FloatField(null=True, blank=True)
    fraction_valid = models.FloatField(null=True, blank=True)
    avg_nll = models.FloatField(null=True, blank=True)
    unique_scaffolds = models.IntegerField(null=True, blank=True)

    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created"]