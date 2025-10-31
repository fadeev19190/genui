# genui/generators/extensions/genuireinvent/models.py

from __future__ import annotations

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


# class ReinventEnvironment(DataSet):
#     class RewardScheme(models.TextChoices):
#         paretoCrowding = 'PC', _('Pareto Front with Crowding Distance (PC)')
#         paretoSimilarity = 'PS', _('Pareto Front with Similarity (PS)')
#         weightedSum = 'WS', _('Weighted Sum (WS)')
#
#     rewardScheme = models.CharField(max_length=2, choices=RewardScheme.choices, default=RewardScheme.paretoCrowding)
#
#     def getInstance(self, use_modifiers=True):
#         scorers = []
#         thresholds = []
#         for scorer in self.scorers.all():
#             scorers.append(scorer.getInstance(use_modifiers=use_modifiers))
#             thresholds.append(scorer.getThreshold())
#
#         schemes = {
#             self.RewardScheme.paretoCrowding: ParetoCrowdingDistance(),
#             self.RewardScheme.paretoSimilarity: ParetoSimilarity(),
#             self.RewardScheme.weightedSum: WeightedSum()
#         }
#         reward_scheme = schemes[self.rewardScheme]
#         return environment.DrugExEnvironment(scorers, thresholds, reward_scheme)
#
#
#
# class ReinventAgent(Model):
#     model = models.ForeignKey(ReinventNet, on_delete=models.CASCADE)
#     parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True)