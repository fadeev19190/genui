# genui/generators/extensions/genuireinvent/models.py
"""
Integration layer between GENUI and REINVENT.

This module defines Django models and utilities to:
- prepare a cleaned SMILES corpus from a MolSet and store a preview in ModelFile,
- build TOML configurations for REINVENT transfer learning,
- run the REINVENT CLI to perform transfer learning,
- manage well-known directories (corpora, checkpoints, tmp) under GENUI's files area.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Iterator

import shutil
import sys

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import models

from genui.compounds.models import MolSet
from genui.models.models import Model, ModelFile, TrainingStrategy, ValidationStrategy

import subprocess


def _ensure_dir(path: str) -> str:
    """
    Create a directory path if it does not exist.

    Args:
        path: Absolute directory path to create.

    Returns:
        The same path for chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def _files_root() -> str:
    """
    Return the root directory for generated files as defined by GENUI settings.

    Returns:
        Absolute path of GENUI's files directory.
    """
    return settings.GENUI_SETTINGS["FILES_DIR"]


def corpora_dir() -> str:
    """
    Directory where cleaned SMILES corpora are stored.

    Returns:
        Absolute path to the 'corpora' directory, ensured to exist.
    """
    return _ensure_dir(os.path.join(_files_root(), "corpora"))


def checkpoints_dir() -> str:
    """
    Directory where REINVENT checkpoints and logs are stored.

    Returns:
        Absolute path to the 'checkpoints' directory, ensured to exist.
    """
    return _ensure_dir(os.path.join(_files_root(), "checkpoints"))


def prior_path() -> str:
    """
    Canonical, fixed path to the REINVENT prior model file.
    Resolves two levels above BASE_DIR (…/genui) to reach /genui/files/…,
    regardless of the temp FILES_DIR in tests.
    """
    repo_root = os.path.abspath(os.path.join(settings.BASE_DIR, os.pardir, os.pardir))
    fixed_path = os.path.join(repo_root, "files", "checkpoints", "prior", "reinvent.prior")

    if not os.path.isfile(fixed_path):
        raise FileNotFoundError(f"Hard-coded REINVENT prior not found at: {fixed_path}")
    return fixed_path


def tmp_dir() -> str:
    """
    Directory for temporary files produced during preprocessing and config building.

    Returns:
        Absolute path to the 'tmp' directory, ensured to exist.
    """
    return _ensure_dir(os.path.join(_files_root(), "tmp"))


class ReinventNet(Model):
    """
    Django model that orchestrates data preparation and transfer learning for REINVENT.

    Responsibilities:
    - Prepare a cleaned SMILES corpus from a MolSet and store a short preview in ModelFile.
    - Build a TOML config for REINVENT transfer learning based on the attached training strategy.
    - Run the external 'reinvent' CLI to execute transfer learning.

    Fields:
        molset: Optional link to a MolSet providing input molecules.
        parent: Optional link to a parent ReinventNet (e.g., for lineage tracking).
    """

    molset = models.ForeignKey(MolSet, on_delete=models.CASCADE, null=True)
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True)

    # --- ModelFile compatibility helpers ---
    def createCorpusFile(self, note: str, name: str) -> ModelFile:
        """
        Get or create a ModelFile for auxiliary corpus data.

        Looks up a ModelFile with the given note; if not present, creates an empty file.

        Args:
            note: Logical note used to identify the file record.
            name: File name to create when no record exists.

        Returns:
            ModelFile instance.
        """
        mf = self.files.filter(kind=ModelFile.AUXILIARY, note=note).first()
        if mf:
            return mf
        return ModelFile.create(self, name, ContentFile(""), note=note)

    @property
    def corpusFileTrain(self) -> ModelFile:
        """
        Auxiliary ModelFile that stores a short preview of the cleaned training corpus (.smi).
        """
        return self.createCorpusFile("Reinvent_corpus_train", "corpus_train.smi")

    @property
    def corpusFileTest(self) -> ModelFile:
        """
        Reserved auxiliary ModelFile for a test corpus (.smi).
        """
        return self.createCorpusFile("Reinvent_corpus_test", "corpus_test.smi")

    # --- Corpus I/O ---
    @staticmethod
    def _read_smiles_from_path(path: str) -> Iterable[str]:
        """
        Stream SMILES lines from a plain text file.

        Args:
            path: Absolute path to a text file with one SMILES per line.

        Yields:
            Non-empty SMILES strings stripped of surrounding whitespace.
        """
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s:
                    yield s

    # --- MolSet fallback (iterate SMILES directly from DB) ---
    def _iter_molset_smiles(self) -> Iterator[str]:
        """
        Yield SMILES strings from molecules linked to this MolSet when no input file is attached.
        Tries several common related manager names and also discovers one-to-many relations dynamically.
        Accepts common SMILES field names.
        """
        if not self.molset:
            return
        managers = []
        # explicit guesses first
        for name in ("molecules", "items", "entries"):
            mgr = getattr(self.molset, name, None)
            if mgr:
                managers.append(mgr)
        # discover all reverse one-to-many accessors (e.g., MoleculeInSet, GeneratedMolecule, etc.)
        try:
            for f in self.molset._meta.get_fields():
                if getattr(f, "one_to_many", False):
                    mgr = getattr(self.molset, f.get_accessor_name(), None)
                    if mgr and mgr not in managers:
                        managers.append(mgr)
        except Exception:
            pass

        smi_fields = ("smiles", "SMILES", "canonical_smiles", "smi")
        for mgr in managers:
            try:
                for obj in mgr.all():
                    for fld in smi_fields:
                        val = getattr(obj, fld, None)
                        if val:
                            s = str(val).strip()
                            if s:
                                yield s
                            break
            except Exception:
                continue

    # --- Public corpus props ---
    @property
    def corpusTrain(self) -> List[str]:
        """
        Load the previewed training corpus from the auxiliary ModelFile.

        Returns:
            List of SMILES strings.
        """
        return list(self._read_smiles_from_path(self.corpusFileTrain.path))

    # --- Paths ---
    def get_clean_corpus_path(self) -> str:
        """
        Compute the on-disk path for the cleaned training corpus for this model.

        Returns:
            Absolute path under the corpora directory.
        """
        fname = f"reinvent_{self.pk}_train.smi"
        return os.path.join(corpora_dir(), fname)

    def get_checkpoints_dir(self) -> str:
        """
        Directory used to store REINVENT output models and logs.

        Returns:
            Absolute path to the checkpoints directory.
        """
        return checkpoints_dir()

    # --- Prior ---
    def get_prior_path(self) -> str:
        """
        Validate and return the path to the prior REINVENT model.

        Returns:
            Absolute path to the prior model file.

        Raises:
            FileNotFoundError: If the prior file does not exist.
        """
        path = prior_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prior not found at: {path}")
        return path

    # --- TOML for TL ---
    def build_tl_toml(self, *, device: str = "cpu") -> str:
        """
        Build a temporary TOML configuration file for REINVENT transfer learning.

        The configuration references:
        - the prior model file,
        - the cleaned SMILES corpus (both train and validation),
        - output checkpoint path,
        - learning parameters from the attached ReinventNetTraining strategy.

        Args:
            device: PyTorch device string, e.g., 'cpu' or 'cuda'.

        Returns:
            Absolute path to the generated temporary TOML file.

        Raises:
            RuntimeError: If the training strategy is missing or invalid.
            FileNotFoundError: If the prior model file does not exist.
        """
        # Previously there was no device; make it explicit to avoid runtime failures.
        if not self.trainingStrategy or not isinstance(self.trainingStrategy, ReinventNetTraining):
            raise RuntimeError("ReinventNetTraining is required to build TL config.")
        prior = self.get_prior_path()
        corpus = self.get_clean_corpus_path()
        trn = self.trainingStrategy
        out_file = os.path.join(self.get_checkpoints_dir(), f"reinvent_{self.pk}_tl.model")
        min_sbs = 100
        sbs = max(min_sbs, trn.sample_batch_size)


        body = f"""\
run_type = "transfer_learning"
device = "{device}"

[parameters]
num_epochs = {trn.epochs}
save_every_n_epochs = {trn.save_every_n_epochs}
batch_size = {trn.batch_size}
sample_batch_size = {sbs}

input_model_file = "{prior}"
smiles_file = "{corpus}"
validation_smiles_file = "{corpus}"
output_model_file = "{out_file}"
        """

        # Write the TOML to a unique temp file under GENUI tmp.
        fd, cfg_path = tempfile.mkstemp(prefix=f"tl_reinvent_{self.pk}_", suffix=".toml", dir=tmp_dir())
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            tf.write(body)
        return cfg_path

    # --- Preprocessing ---
    def prepareData(self) -> str:
        """
        Prepare a cleaned SMILES corpus for REINVENT.

        Behavior:
        - If the attached MolSet has a file, use it as input.
        - Otherwise, fall back to iterating MolSet molecules and writing a raw .smi file.
        - Build a temporary TOML for `reinvent.datapipeline.preprocess` and execute it.
        - Store a preview (first 1000 SMILES) in the auxiliary ModelFile.

        Returns:
            Absolute path to the cleaned corpus (.smi) stored under the corpora directory.

        Raises:
            ValueError: If no MolSet is attached.
            RuntimeError: If neither an input file nor any molecules with SMILES are available.
            RuntimeError: If the REINVENT datapipeline is unavailable.
        """
        if not self.molset:
            raise ValueError("MolSet is not attached.")

        # Prefer an input file attached to MolSet if available.
        msf = getattr(self.molset, "files", None)
        input_path = None
        if msf and msf.exists():
            first = msf.first()
            if first and getattr(first, "file", None):
                input_path = first.file.path

        # If no file is attached to the MolSet, create a raw .smi from DB molecules.
        if not input_path:
            raw_tsv = os.path.join(tmp_dir(), f"reinvent_raw_{self.pk}.smi.tsv")
            _ensure_dir(os.path.dirname(raw_tsv))
            wrote = 0
            with open(raw_tsv, "w", encoding="utf-8") as rawf:
                rawf.write("SMILES\n")
                for smi in self._iter_molset_smiles():
                    rawf.write(smi + "\n")
                    wrote += 1
            if wrote == 0:
                raise RuntimeError("MolSet has no input file and contains no molecules with SMILES.")
            input_path = raw_tsv
        output_path = self.get_clean_corpus_path()

        # Use REINVENT's datapipeline to clean/filter/transform SMILES.
        try:
            from reinvent.datapipeline import preprocess
        except Exception as e:
            raise RuntimeError(
                "reinvent.datapipeline.preprocess is required to prepare data."
            ) from e

        # Build a minimal preprocess configuration TOML.
        pre_body = f"""\
input_csv_file = "{input_path}"
smiles_column = "SMILES"
separator = "\\t"
output_smiles_file = "{output_path}"

[filter]
elements = []
transforms = ["standard"]
inchi_key_deduplicate = true
"""
        fd, cfg_tmp = tempfile.mkstemp(
            prefix=f"reinvent_preprocess_{self.pk}_", suffix=".toml", dir=tmp_dir()
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(pre_body)

            # The preprocess CLI expects an object with `config_filename` and optional `log_filename`.
            args = type("Args", (), {"config_filename": cfg_tmp, "log_filename": None})
            preprocess.main(args)
        finally:
            # Best-effort cleanup of the temp TOML.
            try:
                os.remove(cfg_tmp)
            except OSError:
                pass

        # Sync a short preview (up to 1000 lines) into the auxiliary ModelFile.
        try:
            preview = []
            for i, s in enumerate(self._read_smiles_from_path(output_path)):
                if i >= 1000:
                    break
                preview.append(s)
            data = "\n".join(preview) + ("\n" if preview else "")
            self.corpusFileTrain.file.save(
                os.path.basename(self.corpusFileTrain.path), ContentFile(data), save=True
            )
        except Exception:
            # Preview sync is non-critical; ignore any errors here.
            pass

        return output_path

    # --- Transfer learning ---
    def run_transfer_learning(self, *, device: str = "cpu") -> str:
        toml_path = self.build_tl_toml(device=device)
        out_file = os.path.join(self.get_checkpoints_dir(), f"reinvent_{self.pk}_tl.model")

        # pick the right binary (prefer settings/env; fallback to PATH)
        reinvent_bin = (
                getattr(settings, "REINVENT_BIN", None)
                or os.environ.get("REINVENT_BIN")
                or shutil.which("reinvent")
        )
        if not reinvent_bin:
            raise RuntimeError("REINVENT binary not found. Set settings.REINVENT_BIN or $REINVENT_BIN.")

        # **positional** config file (no flags)
        cmd = [reinvent_bin, toml_path]

        # run & log
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=settings.BASE_DIR,  # optional; set a stable cwd
        )
        lines = []
        for ln in proc.stdout or []:
            lines.append(ln)
        rc = proc.wait()

        # persist to Reinvent_training_log (unchanged from your code)
        try:
            self.createCorpusFile("Reinvent_training_log", "reinvent_training.log").file.save(
                "reinvent_training.log",
                ContentFile(f"[CMD] {' '.join(cmd)}\n[CWD] {os.getcwd()}\n{''.join(lines)}"),
                save=True,
            )
        except Exception:
            pass

        if rc != 0:
            raise RuntimeError(f"REINVENT TL failed (exit={rc}). See TOML: {toml_path}")

        return out_file

    def getModel(self) -> str:
        """
        Prepare data, build TOML, run TL, and return the trained model path.
        """
        # Ensure cleaned corpus exists for this model
        self.prepareData()

        # Run TL and return the produced checkpoint path
        return self.run_transfer_learning(device="cpu")


class ReinventNetValidation(ValidationStrategy):
    """
    Validation strategy for REINVENT runs.

    Fields:
        validSetSize: Number of molecules to use for validation.
    """
    validSetSize = models.IntegerField(default=10000)


class ReinventNetTraining(TrainingStrategy):
    """
    Training strategy for REINVENT transfer learning.

    Fields:
        epochs: Number of training epochs.
        batch_size: Mini-batch size for training.
        save_every_n_epochs: Frequency of checkpoint saving.
        sample_batch_size: Batch size for sampling during training.
    """
    epochs = models.IntegerField(default=10)
    batch_size = models.IntegerField(default=64)
    save_every_n_epochs = models.IntegerField(default=1)
    sample_batch_size = models.IntegerField(default=100)

    class Meta:
        verbose_name = "Reinvent Training Strategy"
        verbose_name_plural = "Reinvent Training Strategies"

    def __str__(self):
        """
        Human-readable representation for admin/logging.
        """
        return (
            f"ReinventTraining(id={self.id}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, "
            f"sample_batch_size={self.sample_batch_size}, "
            f"save_every={self.save_every_n_epochs})"
        )

    def get_prior_path(self) -> str:
        """
        Validate and return the path to the prior REINVENT model.

        Returns:
            Absolute path to the prior model file.

        Raises:
            FileNotFoundError: If the prior file does not exist.
        """
        path = prior_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prior not found at: {path}")
        return path

    def get_checkpoints_dir(self) -> str:
        """
        Directory used to store REINVENT output models and logs.

        Returns:
            Absolute path to the checkpoints directory.
        """
        return checkpoints_dir()

    def processMetaData(self, metadata: dict):
        """
        Update strategy fields from a metadata mapping and persist the changes.

        Recognized keys:
        - 'epochs'
        - 'batch_size'
        - 'sample_batch_size'
        - 'save_every_n_epochs'

        Args:
            metadata: Mapping of parameter names to values.
        """
        self.epochs = metadata.get("epochs", self.epochs)
        self.batch_size = metadata.get("batch_size", self.batch_size)
        self.sample_batch_size = metadata.get("sample_batch_size", self.sample_batch_size)
        self.save_every_n_epochs = metadata.get("save_every_n_epochs", self.save_every_n_epochs)
        self.save()
