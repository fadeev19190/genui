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
from django.utils import timezone

from genui.compounds.models import MolSet, ActivitySet
from genui.models.models import Model, ModelFile, TrainingStrategy, ValidationStrategy, ModelPerfomanceNN, ModelPerformance
from genui.projects.models import DataSet
from genui.generators.models import Generator


DEFAULT_PRIOR_REL = os.path.join("checkpoints", "prior", "reinvent.prior")

def _resolve_reinvent_prior_path() -> str:
    candidates = []
    candidates.append(getattr(settings, "REINVENT_PRIOR_PATH", None))
    try:
        candidates.append(getattr(settings, "GENUI_SETTINGS", {}).get("REINVENT_PRIOR_PATH"))
    except Exception:
        candidates.append(None)
    candidates.append(os.environ.get("REINVENT_PRIOR"))

    files_dir = None
    try:
        files_dir = getattr(settings, "GENUI_SETTINGS", {}).get("FILES_DIR")
    except Exception:
        files_dir = None
    if files_dir:
        candidates.append(os.path.join(files_dir, DEFAULT_PRIOR_REL))

    tried = [c for c in candidates if c]
    for c in tried:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError(
        "REINVENT prior not found. Tried: "
        + ", ".join(tried or ["<no candidates>"])
        + ". Configure REINVENT_PRIOR (env) or REINVENT_PRIOR_PATH (settings), "
        + "or place the prior at "
        + (os.path.join(files_dir, DEFAULT_PRIOR_REL) if files_dir else "<FILES_DIR>/" + DEFAULT_PRIOR_REL)
    )

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

    # Prefer true in-place overwrite for local storage
    try:
        p = mf.file.path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
        return
    except Exception:
        pass

    # Fallback for storages without .path (S3 etc.)
    current_rel = mf.file.name
    if not current_rel:
        # first save
        mf.file.save(filename or f"aux_{mf.pk}", ContentFile(data), save=True)
        return

    try:
        mf.file.storage.delete(current_rel)
    except Exception:
        pass

    # Force same name
    mf.file.save(current_rel, ContentFile(data), save=False)
    mf.file.name = current_rel
    mf.save(update_fields=["file"])

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

    # ── Prior path  ────────────────────────────────────────────────
    def get_prior_path(self) -> str:
        return _resolve_reinvent_prior_path()

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
        try:
            import reinvent.runmodes.RL.memories as mem
            from reinvent.runmodes.RL.memories.diversity_filter import DiversityFilter
        except ModuleNotFoundError:
            return []
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

# TODO: predelat na Dataset
class ReinventEnvironment(DataSet):
    name = models.CharField(max_length=255)

    # Backwards-compatible (already in your model)
    prior_model = models.ForeignKey(
        ModelFile, on_delete=models.PROTECT, related_name="reinvent_prior_files",
        null=True, blank=True
    )
    agent_model = models.ForeignKey(
        ModelFile, on_delete=models.PROTECT, related_name="reinvent_agent_files",
        null=True, blank=True
    )

    # NEW: choose from existing models (user-added or previously trained)
    prior_net = models.ForeignKey(
        "ReinventNet", on_delete=models.PROTECT, related_name="reinvent_as_prior",
        null=True, blank=True
    )
    agent_net = models.ForeignKey(
        "ReinventNet", on_delete=models.PROTECT, related_name="reinvent_as_agent",
        null=True, blank=True
    )

    diversity_filter = models.ForeignKey(ReinventDiversityFilter, null=True, blank=True, on_delete=models.SET_NULL)
    reward_scheme = models.ForeignKey("ReinventEnvironmentScores", null=True, blank=True, on_delete=models.SET_NULL)

    inception_smiles = models.ForeignKey(ModelFile, null=True, blank=True, on_delete=models.SET_NULL)
    inception_memory_size = models.IntegerField(default=0)
    inception_sample_size = models.IntegerField(default=0)

    def get_prior_path(self) -> str:
        if self.prior_net:
            return self.prior_net.get_active_checkpoint_path()
        if self.prior_model:
            return self.prior_model.file.path
        raise RuntimeError("No prior selected: set prior_net or prior_model.")

    def get_agent_path(self) -> str:
        if self.agent_net:
            return self.agent_net.get_active_checkpoint_path()
        if self.agent_model:
            return self.agent_model.file.path
        raise RuntimeError("No agent selected: set agent_net or agent_model.")


# =====================================================================
# SCORING
# =====================================================================

class ScoreModifier(DataSet):
    """
    Base class for score modifiers (DrugEx style).
    Stored polymorphically; concrete implementations are subclasses.
    """

    def to_reinvent_transform(self) -> dict:
        raise NotImplementedError("Override in subclass.")



# TODO: dat do Environmentu
class ReinventEnvironmentScores(ActivitySet):
    aggregation_type = models.CharField(
        max_length=64,
        choices=[
            ("geometric_mean", "geometric_mean"),
            ("weighted_arithmetic_mean", "weighted_arithmetic_mean"),
        ]
    )

class ClippedScore(ScoreModifier):
    """
    Mirrors DrugEx.modifiers.ClippedScore / SmoothClippedScore configuration,
    but exports REINVENT transform dict.
    """
    upper = models.FloatField(null=False)
    lower = models.FloatField(null=False, default=0.0)
    high = models.FloatField(null=False, default=1.0)
    low = models.FloatField(null=False, default=0.0)
    smooth = models.BooleanField(null=False, default=False)

    def to_reinvent_transform(self) -> dict:
        # Keep your existing transform mapping logic, now per-class.
        if not self.smooth:
            # "clipped" effect via double sigmoid window
            return {
                "type": "double_sigmoid",
                "low": float(self.lower),
                "high": float(self.upper),
                "coef_div": float(self.upper - self.lower),
                "coef_si": 20,
                "coef_se": 20,
            }

        # smooth=True -> gentler boundary (your earlier mapping)
        span = float(self.upper - self.lower)
        k = 1.0 / (span + 1e-9)
        return {
            "type": "reverse_sigmoid",
            "low": float(self.lower),
            "high": float(self.upper),
            "k": float(k),
        }


class SmoothHump(ScoreModifier):
    """
    Mirrors DrugEx.modifiers.SmoothHump, exports REINVENT hump transform.
    """
    upper = models.FloatField(null=False, default=1.0)
    lower = models.FloatField(null=False, default=0.0)
    sigma = models.FloatField(null=False, default=0.5)

    def to_reinvent_transform(self) -> dict:
        # Your earlier mapping
        return {
            "type": "double_sigmoid",
            "low": float(self.lower),
            "high": float(self.upper),
            "coef_div": float(self.upper - self.lower),
            "coef_si": int(float(self.sigma) * 20),
            "coef_se": int(float(self.sigma) * 20),
        }


class ScoringMethod(models.Model):
    name = models.CharField(max_length=255)
    weight = models.FloatField(default=1.0)
    scheme = models.ForeignKey("ReinventEnvironmentScores", on_delete=models.CASCADE, related_name="%(class)s_set")
    modifier = models.ForeignKey("ScoreModifier", null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        abstract = True

    def build_transform(self) -> dict | None:
        mod = getattr(self, "modifier", None)
        if not mod:
            return None
        # If ScoreModifier is base, try to downcast via reverse relations
        for attr in ("clippedscore", "smoothhump"):
            child = getattr(mod, attr, None)
            if child is not None:
                mod = child
                break
        fn = getattr(mod, "to_reinvent_transform", None)
        return fn() if callable(fn) else None


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


class ReinventAgentTraining(TrainingStrategy):
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


class ReinventAgentValidation(ValidationStrategy):
    validate_every = models.IntegerField(default=50)
    validation_dataset = models.ForeignKey(ModelFile, null=True, blank=True, on_delete=models.SET_NULL)


class ReinventAgent(Model):
    environment = models.ForeignKey(ReinventEnvironment, on_delete=models.PROTECT)
    training = models.ForeignKey(ReinventAgentTraining, on_delete=models.PROTECT)
    validation = models.ForeignKey(ReinventAgentValidation, null=True, blank=True, on_delete=models.SET_NULL)
    #TODO pouzit Models aby dedilo logiku
    output_model = models.ForeignKey(ModelFile, null=True, blank=True, on_delete=models.SET_NULL)

    # moved from Generator (keep defaults to avoid breaking existing configs)
    tb_logdir = models.CharField(max_length=255, default="tb_logs")
    json_out_config = models.CharField(max_length=255, default="_staged_learning.json")

    def getGenerator(self):
        # Mirrors DrugExAgent.getGenerator()
        return self.generator.order_by("-id").first()

    # ------------------------------------------------------------------
    # Simple file helpers for staged learning
    # ------------------------------------------------------------------
    def _sl_dir(self) -> str:
        base = getattr(settings, "MEDIA_ROOT", ".")
        return os.path.join(base, "reinvent_sl")

    def get_toml_path(self) -> str:
        gen = self.getGenerator()
        if not gen:
            raise RuntimeError("This ReinventAgent is not attached to a Reinvent generator.")
        return os.path.join(self._sl_dir(), f"staged_learning_{gen.pk}.toml")

    def get_rl_log_path(self) -> str:
        gen = self.getGenerator()
        if not gen:
            raise RuntimeError("This ReinventAgent is not attached to a Reinvent generator.")
        return os.path.join(self._sl_dir(), f"reinvent_rl_{gen.pk}.log")

    # ------------------------------------------------------------------
    # Build staged-learning TOML
    # ------------------------------------------------------------------
    def build_staged_toml(self, device="cuda:0") -> str:
        gen = self.getGenerator()
        if not gen:
            raise RuntimeError("This ReinventAgent is not attached to a Reinvent generator.")

        env = self.environment
        train_cfg = self.training

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

        # NOTE: model selection now supports Model OR ModelFile
        lines.append(f'prior_file = "{env.get_prior_path()}"')
        lines.append(f'agent_file = "{env.get_agent_path()}"')

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
                lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
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

        # STAGES
        stages = list(gen.stages.order_by("order"))
        if not stages:
            raise RuntimeError("Staged learning requires at least one stage.")

        sl_dir = self._sl_dir()
        os.makedirs(sl_dir, exist_ok=True)

        for st in stages:
            lines.append("[[stage]]")

            default_chkpt = os.path.join(sl_dir, f"agent_stage{st.order}_run{gen.pk}.chkpt")
            chkpt_path = st.resolve_checkpoint_path(default_chkpt)
            lines.append(f'chkpt_file = "{chkpt_path}"')

            lines.append(f'termination = "{st.termination_type}"')
            lines.append(f"max_score = {st.max_score}")
            lines.append(f"min_steps = {st.min_steps}")
            lines.append(f"max_steps = {st.max_steps}")
            lines.append("")

            prefix = "stage.scoring"

            # EXTERNAL SCORING FILE
            if st.scoring_source == "file":
                scheme = st.scoring_scheme
                if not scheme:
                    raise RuntimeError("Stage requires scoring_scheme when scoring_source='file'.")
                if not st.scoring_file:
                    raise RuntimeError("Stage scoring_source='file' but scoring_file is not set.")

                lines.append(f"[{prefix}]")
                lines.append(f'type = "{scheme.aggregation_type}"')
                lines.append(f'filename = "{st.scoring_file.file.path}"')
                lines.append('filetype = "toml"')
                lines.append("")
                continue

            # INLINE SCORING
            scheme = st.scoring_scheme or env.reward_scheme
            if not scheme:
                raise RuntimeError("Inline scoring requires stage scoring_scheme or environment.reward_scheme.")

            lines.append(f"[{prefix}]")
            lines.append(f'type = "{scheme.aggregation_type}"')
            lines.append("")

            for ps in PropertyScorer.objects.filter(scheme=scheme).order_by("id"):
                prop = ps.property_name
                lines.append(f"[[{prefix}.component]]")
                lines.append(f"[{prefix}.component.{prop}]")
                lines.append(f"[[{prefix}.component.{prop}.endpoint]]")
                lines.append(f'name = "{ps.name}"')
                lines.append(f"weight = {ps.weight}")

                tr = ps.build_transform()
                if tr:
                    lines.append(f"[{prefix}.component.{prop}.endpoint.transform]")
                    for k, v in tr.items():
                        lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
                lines.append("")

            for us in UnwantedSmartsScorer.objects.filter(scheme=scheme).order_by("id"):
                if not us.enabled:
                    continue
                patterns = us.load_patterns()

                lines.append(f"[[{prefix}.component]]")
                lines.append(f"[{prefix}.component.custom_alerts]")

                # IMPORTANT: endpoint must be a table/dict (array-of-tables is fine)
                lines.append(f"[[{prefix}.component.custom_alerts.endpoint]]")
                lines.append(f'name = "{us.name}"')
                lines.append(f"weight = {us.weight}")

                # IMPORTANT: REINVENT expects params under endpoint, use dotted keys
                lines.append("params.smarts = [")
                for p in patterns:
                    lines.append(f'  "{p}",')
                lines.append("]")
                lines.append("")

            lines.append("")

        body = "\n".join(lines) + "\n"
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
            raise RuntimeError("REINVENT binary not found. Set REINVENT_BIN in settings or env.")

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

        log_path = self.get_rl_log_path()
        log_text = f"[CMD] {' '.join(cmd)}\n{''.join(lines)}"
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(log_text)

        if rc != 0:
            raise RuntimeError(f"REINVENT staged learning failed (exit {rc}). See log: {log_path}")

        return toml_path


class Reinvent(Generator):
    # keep fields if you already migrated data; frontend can keep using them
    environment = models.ForeignKey(ReinventEnvironment, on_delete=models.PROTECT)
    agent = models.ForeignKey(ReinventAgent, on_delete=models.PROTECT, related_name="generator")

    def build_staged_toml(self, device="cuda:0") -> str:
        return self.agent.build_staged_toml(device=device)

    def run_staged_learning(self, device="cuda:0") -> str:
        return self.agent.run_staged_learning(device=device)

# =====================================================================
# 6. STAGE MODEL
# =====================================================================

class ReinventStage(models.Model):
    generator = models.ForeignKey("Reinvent", related_name="stages", on_delete=models.CASCADE)
    order = models.IntegerField()

    # existing
    chkpt_file = models.ForeignKey(
        ModelFile, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="reinvent_stage_checkpoints"
    )

    # NEW: pick an existing trained net as the checkpoint source
    chkpt_net = models.ForeignKey(
        "ReinventNet", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="reinvent_stage_checkpoint_nets"
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
        "ReinventEnvironmentScores",
        null=True, blank=True,
        on_delete=models.SET_NULL
    )

    scoring_file = models.ForeignKey(
        ModelFile,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="reinvent_stage_scoring_files"
    )

    class Meta:
        ordering = ["order"]

    def resolve_checkpoint_path(self, default_path: str) -> str:
        if self.chkpt_net:
            return self.chkpt_net.get_active_checkpoint_path()
        if self.chkpt_file:
            return self.chkpt_file.file.path
        return default_path

# =====================================================================
# 7. PERFORMANCE LOGGING
# =====================================================================



class ModelPerformanceReinvent(ModelPerformance):
    epoch = models.IntegerField()
    step = models.IntegerField()
    created = models.DateTimeField(default=timezone.now, db_index=True)

    isOnValidationSet = models.BooleanField(default=False)
    note = models.CharField(max_length=128, blank=True)

    stage_index = models.IntegerField(null=True, blank=True)
    avg_score = models.FloatField(null=True, blank=True)
    fraction_valid = models.FloatField(null=True, blank=True)
    avg_nll = models.FloatField(null=True, blank=True)
    unique_scaffolds = models.IntegerField(null=True, blank=True)