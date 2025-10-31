import json
import os
from django.conf import settings
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

import datetime
import shutil

from genui.qsar.tests import QSARModelInit
from genui.models.models import Algorithm, AlgorithmMode, ModelFileFormat
from . import models


TEST_EPOCHS = 2
REINVENT_BIN = "/opt/anaconda3/envs/reinvent4/bin/reinvent"   # <-- set this to your working reinvent CLI


class SetUpReinventMixIn(QSARModelInit):
    """Minimal setup for creating a ReinventNet via REST."""

    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls._admin = User.objects.create_superuser(
            username="fadeevartem",
            email="fadeev19190@gmail.com",
            password="1234",
        )

        # ensure Algorithm/Mode exist & are linked
        cls.mode_generator = AlgorithmMode.objects.get_or_create(name="generator")[0]
        cls.alg_reinvent = Algorithm.objects.get_or_create(name="ReinventNet")[0]
        cls.alg_reinvent.validModes.add(cls.mode_generator)

        # point algorithm to your extension package
        cls.alg_reinvent.corePackage = "genui.generators.extensions.genuireinvent.genuimodels"
        cls.alg_reinvent.save(update_fields=["corePackage"])

        # ensure the .pkg file format exists/attached
        fmt, _ = ModelFileFormat.objects.get_or_create(
            fileExtension=".pkg",
            defaults={"description": "State of a neural network built with pytorch."},
        )
        if fmt not in cls.alg_reinvent.fileFormats.all():
            cls.alg_reinvent.fileFormats.add(fmt)

    def setUp(self):
        super().setUp()
        self.client.force_login(self._admin)

        # ensure project ownership for queryset visibility
        if getattr(self.project, "owner_id", None) != self._admin.id:
            self.project.owner = self._admin
            self.project.save()

        # wire the reinvent binary
        os.environ.setdefault("REINVENT_BIN", REINVENT_BIN)

        repo_files = os.path.abspath(
            os.path.join(settings.BASE_DIR, os.pardir, os.pardir, "files")
        )
        media_debug = os.path.join(repo_files, "media_debug")
        os.makedirs(media_debug, exist_ok=True)
        os.makedirs(os.path.join(repo_files, "checkpoints", "prior"), exist_ok=True)

        # 1) media root for AUX artifacts
        settings.MEDIA_ROOT = media_debug

        # 2) keep GENUI "files" root stable
        settings.GENUI_SETTINGS = {
            **settings.GENUI_SETTINGS,
            "FILES_DIR": repo_files,
        }

        # sanity: required prior must exist at the absolute path used by model code
        prior_abs = models.PRIOR_ABS
        if not os.path.isfile(prior_abs):
            self.fail(
                f"Required prior not found at {prior_abs}. "
                f"Either place it there or change PRIOR_ABS in models.py for tests."
            )

    def _create_reinvent(self, url, initial=None):
        """POST a ReinventNet; returns the DB instance."""
        payload = {
            "name": "Test Reinvent Network (pretraining)" if not initial else "Test Reinvent Network (finetuning)",
            "description": "test description",
            "project": self.project.id,
            "build": True,
            "trainingStrategy": {
                "algorithm": Algorithm.objects.get(name="ReinventNet").id,
                "mode": AlgorithmMode.objects.get(name="generator").id,
                "epochs": TEST_EPOCHS,
                "batch_size": 16,
                "sample_batch_size": 100,         # REINVENT 4.x requires >= 100
                "save_every_n_epochs": 1,
            },
            # small valid set to keep split deterministic on tiny corpora
            "validationStrategy": {"validSetSize": 5, "split_method": "random", "valid_fraction": 0.2},
            "molset": self.molset.id,
        }
        if initial:
            payload["parent"] = initial.id

        resp = self.client.post(url, data=payload, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)
        print(json.dumps(resp.data, indent=4))
        return models.ReinventNet.objects.get(pk=resp.data["id"])

    def _snapshot_artifacts(self, label):
        src = settings.MEDIA_ROOT
        debug_root = os.path.join(
            os.path.abspath(os.path.join(settings.BASE_DIR, os.pardir, os.pardir)),
            "files", "debug_artifacts"
        )
        os.makedirs(debug_root, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dst = os.path.join(debug_root, f"{stamp}_{self.__class__.__name__}_{label}")
        os.makedirs(dst, exist_ok=True)

        for dirpath, _, filenames in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            outdir = os.path.join(dst, rel if rel != "." else "")
            os.makedirs(outdir, exist_ok=True)
            for f in filenames:
                shutil.copy2(os.path.join(dirpath, f), os.path.join(outdir, f))

        print(f"[SNAPSHOT] Copied artifacts to: {dst}")

    def tearDown(self):
        try:
            test_method = getattr(self, self._testMethodName, None)
            label = self._testMethodName if test_method else "unknown_test"
            self._snapshot_artifacts(label)
        finally:
            super().tearDown()


@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class ReinventFromMolsetTestCase(SetUpReinventMixIn, APITestCase):
    """End-to-end: create, prepare corpus, build TOML, run TL via REINVENT CLI."""

    def test_create_and_prepare(self):
        instance = self._create_reinvent(reverse("reinvent-net-list"))

        # Prepare corpus (uses REINVENT datapipeline.preprocess)
        url = reverse("reinvent-net-prepare-corpus", args=[instance.id])
        resp = self.client.post(url, data={}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)
        print(json.dumps(resp.data, indent=4))

        # Preview exists & contains non-empty lines sans header
        aux = instance.corpusPreviewFile
        self.assertIsNotNone(aux)
        self.assertTrue(aux.file and os.path.isfile(aux.file.path))
        with open(aux.file.path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(all(l != "SMILES" for l in lines))

        # Train/valid files exist & non-empty
        train_mf = instance.corpusTrainFile
        valid_mf = instance.corpusValidFile
        self.assertTrue(os.path.isfile(train_mf.path))
        self.assertTrue(os.path.isfile(valid_mf.path))
        with open(train_mf.path, "r", encoding="utf-8") as fh:
            train_lines = [l.strip() for l in fh if l.strip()]
        with open(valid_mf.path, "r", encoding="utf-8") as fh:
            valid_lines = [l.strip() for l in fh if l.strip()]
        self.assertGreaterEqual(len(train_lines), 1)
        self.assertGreaterEqual(len(valid_lines), 1)
        self.assertTrue(all(l != "SMILES" for l in train_lines))
        self.assertTrue(all(l != "SMILES" for l in valid_lines))

    def test_build_tl_toml_and_run_cli(self):
        instance = self._create_reinvent(reverse("reinvent-net-list"))

        # Ensure corpus exists (and split)
        instance.prepareData()

        # Build TOML and verify content
        toml_path = instance.build_tl_toml(device="cpu")
        self.assertTrue(os.path.isfile(toml_path))
        with open(toml_path, "r", encoding="utf-8") as fh:
            cfg = fh.read()
        out_file = instance.checkpointFile.path

        self.assertIn('run_type = "transfer_learning"', cfg)
        self.assertIn(f'input_model_file = "{models.PRIOR_ABS}"', cfg)
        self.assertIn(f'smiles_file = "{instance.corpusTrainFile.path}"', cfg)
        self.assertIn(f'validation_smiles_file = "{instance.corpusValidFile.path}"', cfg)
        self.assertIn(f'output_model_file = "{out_file}"', cfg)

        # Run TL; adapter writes AUX training log
        produced = instance.run_transfer_learning(device="cpu")
        self.assertEqual(produced, out_file)

        log_mf = instance.trainLogFile
        self.assertIsNotNone(log_mf)
        self.assertTrue(log_mf.file and os.path.isfile(log_mf.file.path))
        with open(log_mf.file.path, "r", encoding="utf-8") as fh:
            log_txt = fh.read()
        self.assertIn("[CMD]", log_txt)
        self.assertTrue(len(log_txt.strip()) > 0)

    def test_best_epoch_persisted_and_active_checkpoint(self):
        instance = self._create_reinvent(reverse("reinvent-net-list"))
        instance.prepareData()
        instance.run_transfer_learning(device="cpu")

        ts = instance.trainingStrategy
        # Now parsed from the REINVENT log; assert both values exist
        self.assertIsNotNone(ts.best_epoch, "best_epoch not set; check REINVENT log contained the expected line")
        self.assertIsNotNone(ts.best_valid_loss, "best_valid_loss not set; check REINVENT log contained the expected line")

        active = instance.get_active_checkpoint_path()
        self.assertTrue(os.path.isfile(active), f"Active checkpoint missing at {active}")


@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class ReinventHierarchyTestCase(SetUpReinventMixIn, APITestCase):
    """Simple parent/child linkage like DrugEx."""

    def test_parent_child(self):
        root = self._create_reinvent(reverse("reinvent-net-list"))
        child = self._create_reinvent(reverse("reinvent-net-list"), initial=root)
        self.assertIsNotNone(child.parent)
        self.assertEqual(child.parent_id, root.id)