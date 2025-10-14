import json
import os
import re
import sys
from unittest import mock
import shutil

import pytest
from django.conf import settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from genui.qsar.tests import QSARModelInit
from genui.models.models import Algorithm, AlgorithmMode, ModelFileFormat
from . import models

TEST_EPOCHS = 2
REINVENT_BIN = "/opt/anaconda3/envs/reinvent4/bin/reinvent"  # your working binary


def _print_toml_if_present(error_payload):
    """If the error includes 'See TOML: <path>', print the TOML."""
    try:
        msg = str(error_payload.get("error", ""))
    except Exception:
        print("Cannot parse error payload:", error_payload)
        return
    m = re.search(r"See TOML:\s*([^\s)]+\.toml)", msg)
    if not m:
        print("No TOML path found in error:", msg)
        return
    toml_path = m.group(1)
    try:
        with open(toml_path, "r", encoding="utf-8") as fh:
            toml_text = fh.read()
        print("\n=== TL TOML @", toml_path, "===\n", toml_text, "\n=============================\n")
    except Exception as e:
        print("Could not read TOML at", toml_path, "->", repr(e))


class SetUpReinventGeneratorsMixIn(QSARModelInit):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls._admin = User.objects.create_superuser(
            username="fadeevartem",
            email="fadeev19190@gmail.com",
            password="1234",
        )
        cls.mode_generator = AlgorithmMode.objects.get_or_create(name="generator")[0]
        cls.alg_reinvent = Algorithm.objects.get_or_create(name="ReinventNet")[0]
        cls.alg_reinvent.validModes.add(cls.mode_generator)

        # Ensure a file format exists so saveFile() can pick [0]
        fmt, _ = ModelFileFormat.objects.get_or_create(
            fileExtension=".pkg",
            defaults={"description": "State of a neural network built with pytorch."},
        )
        if fmt not in cls.alg_reinvent.fileFormats.all():
            cls.alg_reinvent.fileFormats.add(fmt)
        # Make sure discovery points at the correct core package
        if cls.alg_reinvent.corePackage != "genui.generators.extensions.genuireinvent.genuimodels":
            cls.alg_reinvent.corePackage = "genui.generators.extensions.genuireinvent.genuimodels"
            cls.alg_reinvent.save(update_fields=["corePackage"])
        cls.alg_reinvent.save()

    def setUp(self):
        super().setUp()
        self.client.force_login(self._admin)

        if getattr(self.project, "owner_id", None) != self._admin.id:
            self.project.owner = self._admin
            self.project.save()

        # --- Point FILES_DIR to the repo 'files' folder (same place as prior) ---
        repo_files = os.path.abspath(os.path.join(settings.BASE_DIR, os.pardir, os.pardir, "files"))
        settings.GENUI_SETTINGS = {**settings.GENUI_SETTINGS, "FILES_DIR": repo_files}

        # Ensure required subdirs exist
        for sub in ("corpora", "checkpoints", "tmp"):
            os.makedirs(os.path.join(repo_files, sub), exist_ok=True)

        # Patch the function used by the integration to compute the files root
        # so corpora/checkpoints/tmp all land under repo_files (not /var/folders/...).
        self._files_root_patcher = mock.patch(
            "genui.generators.extensions.genuireinvent.models._files_root",
            return_value=repo_files,
        )
        self._files_root_patcher.start()
        self.addCleanup(self._files_root_patcher.stop)

        # NEVER touch prior; only clean corpora/tmp and non-prior checkpoints
        corpora_dir = os.path.join(repo_files, "corpora")
        tmp_dir = os.path.join(repo_files, "tmp")
        ckpt_dir = os.path.join(repo_files, "checkpoints")

        for d in (corpora_dir, tmp_dir):
            for name in os.listdir(d):
                p = os.path.join(d, name)
                try:
                    if os.path.isfile(p) or os.path.islink(p):
                        os.remove(p)
                    else:
                        shutil.rmtree(p)
                except Exception:
                    pass

        for name in os.listdir(ckpt_dir):
            if name == "prior":
                continue
            p = os.path.join(ckpt_dir, name)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.remove(p)
                else:
                    shutil.rmtree(p)
            except Exception:
                pass

        prior_path = os.path.join(repo_files, "checkpoints", "prior", "reinvent.prior")
        if not os.path.isfile(prior_path):
            print(f"[WARN] Prior not found at expected repo path: {prior_path}")

    def createReinventNet(self, create_url, initial=None):
        post_data = {
            "name": "Test Reinvent Network (pretraining)" if not initial else "Test Reinvent Network (finetuning)",
            "description": "test description",
            "project": self.project.id,
            "build": True,  # trigger the builder (Celery eager in tests)
            "trainingStrategy": {
                "algorithm": Algorithm.objects.get(name="ReinventNet").id,
                "mode": AlgorithmMode.objects.get(name="generator").id,
                "epochs": TEST_EPOCHS,
                "batch_size": 16,
                "sample_batch_size": 100,  # REINVENT 4.6.22 requires ≥ 100
                "save_every_n_epochs": 1,
            },
            "validationStrategy": {"validSetSize": 5},
            "molset": self.molset.id,
        }
        if initial:
            post_data["parent"] = initial.id

        def _dummy_serializer(path: str) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(b"\x00")

        # Only patch legacy builder touchpoints; run real preprocess + real TL
        with mock.patch(
            "genui.generators.extensions.genuireinvent.genuimodels.algorithms.ReinventNetwork.fit",
            new=lambda self, X, y=None: None,
        ), mock.patch(
            "genui.generators.extensions.genuireinvent.genuimodels.algorithms.ReinventAlgorithm.getSerializer",
            new=lambda self: _dummy_serializer,
        ), mock.patch.dict(
            os.environ, {"REINVENT_BIN": REINVENT_BIN}, clear=False,
        ):
            resp = self.client.post(create_url, data=post_data, format="json")

        if resp.status_code != 201:
            _print_toml_if_present(resp.data)
        self.assertEqual(resp.status_code, 201, msg=resp.data)
        print(json.dumps(resp.data, indent=4))
        return models.ReinventNet.objects.get(pk=resp.data["id"])


# ----------------------------- Test cases -----------------------------

@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class ReinventFromMolsetTestCase(SetUpReinventGeneratorsMixIn, APITestCase):
    def test_create_and_prepare(self):
        instance = self.createReinventNet(reverse("reinvent-net-list"))

        # Real preprocessing via the action
        resp = self.client.post(reverse("reinvent-net-prepare-corpus", args=[instance.id]), data={}, format="json")
        if resp.status_code != status.HTTP_201_CREATED:
            _print_toml_if_present(resp.data)
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)
        print(json.dumps(resp.data, indent=4))

        # Cleaned corpus assertions
        cleaned = instance.get_clean_corpus_path()
        self.assertTrue(os.path.isfile(cleaned))
        with open(cleaned, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(all(l != "SMILES" for l in lines))
        self.assertEqual(instance.corpusTrain[: len(lines)], lines[: len(instance.corpusTrain)])

    def test_build_tl_toml_and_run_cli(self):
        instance = self.createReinventNet(reverse("reinvent-net-list"))

        # Ensure corpus exists (real preprocess)
        try:
            instance.prepareData()
        except Exception as e:
            _print_toml_if_present({"error": repr(e)})
            raise

        # Build TOML points to real prior and real corpus under repo files dir
        prior = models.prior_path()
        toml_path = instance.build_tl_toml(device="cpu")
        self.assertTrue(os.path.isfile(toml_path))
        with open(toml_path, "r", encoding="utf-8") as fh:
            cfg = fh.read()
        out_file = os.path.join(instance.get_checkpoints_dir(), f"reinvent_{instance.pk}_tl.model")
        self.assertIn('run_type = "transfer_learning"', cfg)
        self.assertIn(f'input_model_file = "{prior}"', cfg)
        self.assertIn(f'smiles_file = "{instance.get_clean_corpus_path()}"', cfg)
        self.assertIn(f'validation_smiles_file = "{instance.get_clean_corpus_path()}"', cfg)
        self.assertIn(f'output_model_file = "{out_file}"', cfg)

        # Run real TL
        try:
            produced = instance.run_transfer_learning(device="cpu")
        except Exception as e:
            _print_toml_if_present({"error": repr(e)})
            log_mf = instance.files.filter(note="Reinvent_training_log").first()
            if log_mf and getattr(log_mf, "file", None) and os.path.isfile(log_mf.file.path):
                with open(log_mf.file.path, "r", encoding="utf-8") as fh:
                    log_txt = fh.read()
                print("\n=== Reinvent Training Log ===\n" + log_txt + "\n=============================\n")
            raise
        self.assertEqual(produced, out_file)

        # Log file (if persisted) should exist and be non-empty
        log_mf = instance.files.filter(note="Reinvent_training_log").first()
        if log_mf and getattr(log_mf, "file", None):
            with open(log_mf.file.path, "r", encoding="utf-8") as fh:
                log_txt = fh.read()
            self.assertTrue(len(log_txt.strip()) > 0)


class ReinventHierarchyTestCase(SetUpReinventGeneratorsMixIn, APITestCase):
    def test_parent_child(self):
        root = self.createReinventNet(reverse("reinvent-net-list"))
        child = self.createReinventNet(reverse("reinvent-net-list"), initial=root)
        self.assertIsNotNone(child.parent)
        self.assertEqual(child.parent_id, root.id)