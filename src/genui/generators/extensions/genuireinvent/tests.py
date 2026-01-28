import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import override_settings
from django.urls import reverse

from rest_framework import status
from rest_framework.test import APITestCase

from genui.qsar.tests import QSARModelInit
from genui.models.models import Algorithm, AlgorithmMode, ModelFileFormat

from . import models


TEST_EPOCHS = 2


class SetUpReinventMixIn(QSARModelInit):
    """Reusable setup helpers for genuireinvent API/tests.

    Deployment/CI note:
      - These tests should not depend on an installed REINVENT CLI or on user-specific
        absolute paths.
      - We therefore create a temporary MEDIA_ROOT + FILES_DIR and inject a dummy prior
        and dummy checkpoint files as needed.
    """

    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls._admin = User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="admin",
        )

        cls.mode_generator = AlgorithmMode.objects.get_or_create(name="generator")[0]
        cls.alg_reinvent = Algorithm.objects.get_or_create(name="ReinventNet")[0]
        cls.alg_reinvent.validModes.add(cls.mode_generator)
        cls.alg_reinvent.corePackage = "genui.generators.extensions.genuireinvent.genuimodels"
        cls.alg_reinvent.save(update_fields=["corePackage"])

        fmt, _ = ModelFileFormat.objects.get_or_create(
            fileExtension=".pkg",
            defaults={"description": "State of a neural network built with pytorch."},
        )
        if fmt not in cls.alg_reinvent.fileFormats.all():
            cls.alg_reinvent.fileFormats.add(fmt)

    def setUp(self):
        super().setUp()
        self.client.force_login(self._admin)

        # Ensure project ownership for queryset visibility
        if getattr(self.project, "owner_id", None) != self._admin.id:
            self.project.owner = self._admin
            self.project.save()

        # Isolate filesystem writes
        self._tmp_media = tempfile.TemporaryDirectory(prefix="genuireinvent_media_")
        self._tmp_files = tempfile.TemporaryDirectory(prefix="genuireinvent_files_")

        self._old_media_root = settings.MEDIA_ROOT
        self._old_genui_settings = dict(getattr(settings, "GENUI_SETTINGS", {}) or {})

        settings.MEDIA_ROOT = self._tmp_media.name
        settings.GENUI_SETTINGS = {**self._old_genui_settings, "FILES_DIR": self._tmp_files.name}

        # Provide a dummy prior path for TOML generation
        self._old_env_prior = os.environ.get("REINVENT_PRIOR")
        prior_path = os.path.join(self._tmp_files.name, "checkpoints", "prior", "reinvent.prior")
        os.makedirs(os.path.dirname(prior_path), exist_ok=True)
        with open(prior_path, "wb") as fh:
            fh.write(b"DUMMY PRIOR\n")
        os.environ["REINVENT_PRIOR"] = prior_path

    def tearDown(self):
        # Restore settings/env to avoid leaking state to other tests
        settings.MEDIA_ROOT = self._old_media_root
        settings.GENUI_SETTINGS = self._old_genui_settings

        if self._old_env_prior is None:
            os.environ.pop("REINVENT_PRIOR", None)
        else:
            os.environ["REINVENT_PRIOR"] = self._old_env_prior

        try:
            self._tmp_media.cleanup()
        finally:
            self._tmp_files.cleanup()
            super().tearDown()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _create_reinvent_net(self, url, initial=None, *, build=False):
        """POST a ReinventNet via REST. Returns DB instance."""
        payload = {
            "name": "Test Reinvent Network (pretraining)" if not initial else "Test Reinvent Network (finetuning)",
            "description": "test description",
            "project": self.project.id,
            "build": bool(build),  # IMPORTANT: tests should not require REINVENT CLI
            "trainingStrategy": {
                "algorithm": Algorithm.objects.get(name="ReinventNet").id,
                "mode": AlgorithmMode.objects.get(name="generator").id,
                "epochs": TEST_EPOCHS,
                "batch_size": 16,
                "sample_batch_size": 100,
                "save_every_n_epochs": 1,
            },
            "validationStrategy": {"validSetSize": 5, "split_method": "random", "valid_fraction": 0.2},
            "molset": self.molset.id,
        }
        if initial:
            payload["parent"] = initial.id

        resp = self.client.post(url, data=payload, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)
        return models.ReinventNet.objects.get(pk=resp.data["id"])

    def _write_text(self, path: str, text: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def _ensure_net_artifacts(self, net: models.ReinventNet):
        """Create the minimum on-disk artifacts required for staged-learning TOML.

        We intentionally do NOT call reinvent.datapipeline or the reinvent CLI.
        """
        # Minimal SMILES content
        smiles = []
        try:
            smiles = list(getattr(self.molset, "allSmiles", []) or [])
        except Exception:
            smiles = []
        if not smiles:
            smiles = ["CCO", "CCN", "c1ccccc1"]

        train = "\n".join(smiles[: max(1, len(smiles) - 1)]) + "\n"
        valid = "\n".join(smiles[-1:]) + "\n"
        full = train + valid

        self._write_text(net.corpusTrainFile.path, train)
        self._write_text(net.corpusValidFile.path, valid)
        self._write_text(net.corpusFullFile.path, full)
        self._write_text(net.corpusPreviewFile.path, "\n".join(smiles[:3]) + "\n")

        # Dummy checkpoint so ReinventEnvironment.get_*_path works
        ckpt_path = net.checkpointFile.path
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        with open(ckpt_path, "wb") as fh:
            fh.write(b"DUMMY CHECKPOINT\n")

        return net.checkpointFile

    # ---------------------------------------------------------------------
    # Schema-aligned creation helpers (post-migrations)
    # ---------------------------------------------------------------------
    def _create_with_model_fields(self, model_cls, **kwargs):
        field_names = {f.name for f in model_cls._meta.get_fields()}
        filtered = {k: v for k, v in kwargs.items() if k in field_names}
        return model_cls.objects.create(**filtered)

    def _ensure_dataset_links(self, model_cls, kwargs: dict) -> dict:
        fields = {f.name: f for f in model_cls._meta.get_fields()}

        if "project" in fields and "project" not in kwargs:
            kwargs["project"] = self.project

        # some bases use molecules, some molset
        if "molecules" in fields and "molecules" not in kwargs:
            kwargs["molecules"] = self.molset

        if "molset" in fields and "molset" not in kwargs:
            kwargs["molset"] = self.molset

        return kwargs

    def _get_builder_model(self):
        for app_label in ("models", "genui_models", "genui"):
            for cls_name in ("Builder", "ModelBuilder"):
                try:
                    return apps.get_model(app_label, cls_name)
                except Exception:
                    continue
        raise RuntimeError("Could not locate Builder model (tried Builder/ModelBuilder).")

    def _create_builder_for(self, *, model_class_name: str):
        Builder = self._get_builder_model()

        kwargs = {}
        for f in Builder._meta.fields:
            if getattr(f, "primary_key", False):
                continue
            if getattr(f, "auto_created", False):
                continue
            if getattr(f, "auto_now", False) or getattr(f, "auto_now_add", False):
                continue
            if getattr(f, "has_default", lambda: False)() and f.has_default():
                continue
            if getattr(f, "null", False) or getattr(f, "blank", False):
                continue

            name = f.name.lower()

            # FKs
            if getattr(f, "many_to_one", False) and getattr(f, "remote_field", None):
                rel = f.remote_field.model
                rel_name = getattr(rel, "__name__", "")

                if rel_name == "Project":
                    kwargs[f.name] = self.project
                    continue
                if rel_name in ("User", get_user_model().__name__):
                    kwargs[f.name] = self._admin
                    continue
                if rel_name == "Algorithm":
                    kwargs[f.name] = self.alg_reinvent
                    continue
                if rel_name == "AlgorithmMode":
                    kwargs[f.name] = self.mode_generator
                    continue
                continue

            if getattr(f, "choices", None):
                kwargs[f.name] = f.choices[0][0]
                continue

            internal = f.get_internal_type()
            if internal in ("CharField", "TextField"):
                if "class" in name and "model" in name:
                    kwargs[f.name] = model_class_name
                elif "name" in name:
                    kwargs[f.name] = f"builder:{model_class_name}"
                elif "status" in name or "state" in name:
                    kwargs[f.name] = "created"
                else:
                    kwargs[f.name] = "test"
            elif internal in ("IntegerField", "BigIntegerField", "PositiveIntegerField", "SmallIntegerField"):
                kwargs[f.name] = 0
            elif internal in ("FloatField", "DecimalField"):
                kwargs[f.name] = 0.0
            elif internal == "BooleanField":
                kwargs[f.name] = False
            elif internal == "JSONField":
                kwargs[f.name] = {}
            else:
                kwargs[f.name] = "test"

        return Builder.objects.create(**kwargs)

    def _create_model_like(self, model_cls, **kwargs):
        fields = {f.name: f for f in model_cls._meta.get_fields()}

        if "project" in fields and "project" not in kwargs:
            kwargs["project"] = self.project

        if "algorithm" in fields and "algorithm" not in kwargs:
            kwargs["algorithm"] = self.alg_reinvent

        if "mode" in fields and "mode" not in kwargs:
            kwargs["mode"] = self.mode_generator

        if "builder" in fields and "builder" not in kwargs:
            kwargs["builder"] = self._create_builder_for(model_class_name=model_cls.__name__)

        return self._create_with_model_fields(model_cls, **kwargs)

    def _create_dataset_like(self, model_cls, **kwargs):
        kwargs = self._ensure_dataset_links(model_cls, kwargs)
        return self._create_with_model_fields(model_cls, **kwargs)

    def _create_strategy_like(self, model_cls, *, model_instance, **kwargs):
        fields = {f.name: f for f in model_cls._meta.get_fields()}

        if "modelInstance" in fields and "modelInstance" not in kwargs:
            kwargs["modelInstance"] = model_instance

        if "algorithm" in fields and "algorithm" not in kwargs:
            kwargs["algorithm"] = self.alg_reinvent

        if "mode" in fields and "mode" not in kwargs:
            kwargs["mode"] = self.mode_generator

        if "epochs" in fields and "epochs" not in kwargs:
            kwargs["epochs"] = 1

        return self._create_with_model_fields(model_cls, **kwargs)


@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class ReinventStagedLearningTestCase(SetUpReinventMixIn, APITestCase):
    """Staged-learning tests that validate TOML schema and endpoints."""

    def _mk_scheme_env_agent_gen(self, net: models.ReinventNet, *, add_diversity=False):
        scheme = self._create_dataset_like(
            models.ReinventEnvironmentScores,
            aggregation_type="geometric_mean",
        )

        models.UnwantedSmartsScorer.objects.create(
            name="unwanted_alerts",
            weight=1.0,
            scheme=scheme,
            enabled=True,
        )

        df = None
        if add_diversity:
            df = models.ReinventDiversityFilter.objects.create(
                type="ScaffoldSimilarity",
                bucket_size=10,
                minscore=0.4,
                minsimilarity=0.4,
                penalty_multiplier=0.5,
            )

        env = self._create_dataset_like(
            models.ReinventEnvironment,
            name="Test RL Environment",
            prior_net=net,
            agent_net=net,
            diversity_filter=df,
            reward_scheme=scheme,
        )

        train_cfg = self._create_strategy_like(
            models.ReinventAgentTraining,
            model_instance=net,
            batch_size=16,
            unique_sequences=True,
            randomize_smiles=True,
            tb_isim=False,
            use_checkpoint=False,
            purge_memories=False,
            summary_csv_prefix="reinvent",
            learning_type="dap",
            sigma=64.0,
            rate=0.0005,
        )

        agent = self._create_model_like(
            models.ReinventAgent,
            name="Test Reinvent Agent",
            description="agent for staged learning tests",
            environment=env,
            training=train_cfg,
            validation=None,
            output_model=None,
            tb_logdir=os.path.join(settings.MEDIA_ROOT, "tb_rl"),
            json_out_config="_staged_learning.json",
        )

        gen = self._create_model_like(
            models.Reinvent,
            name="Test Staged Learning Run",
            description="reinvent staged learning run for tests",
            environment=env,
            agent=agent,
        )

        return scheme, env, agent, gen

    def test_build_staged_learning_toml_schema_correct(self):
        net = self._create_reinvent_net(reverse("reinvent-net-list"), build=False)
        ckpt_mf = self._ensure_net_artifacts(net)
        self.assertTrue(os.path.isfile(ckpt_mf.path))

        scheme, env, agent, gen = self._mk_scheme_env_agent_gen(net, add_diversity=True)

        models.ReinventStage.objects.create(
            generator=gen,
            order=0,
            termination_type="simple",
            max_score=1.0,
            min_steps=1,
            max_steps=10,
            scoring_source="inline",
            scoring_scheme=scheme,
        )

        toml_path = gen.build_staged_toml(device="cpu")
        self.assertTrue(os.path.isfile(toml_path), f"TOML not written at {toml_path}")

        cfg = open(toml_path, "r", encoding="utf-8").read()

        self.assertIn('run_type = "staged_learning"', cfg)
        self.assertIn('device = "cpu"', cfg)
        self.assertIn('tb_logdir = "', cfg)
        self.assertIn('json_out_config = "_staged_learning.json"', cfg)

        self.assertIn("[parameters]", cfg)
        self.assertIn("use_checkpoint = false", cfg)
        self.assertIn('summary_csv_prefix = "reinvent"', cfg)
        self.assertIn('prior_file = "', cfg)
        self.assertIn('agent_file = "', cfg)
        self.assertIn("batch_size = 16", cfg)

        self.assertIn("[learning_strategy]", cfg)
        self.assertIn('type = "dap"', cfg)
        self.assertIn("sigma = 64.0", cfg)
        self.assertIn("rate = 0.0005", cfg)

        self.assertIn("[diversity_filter]", cfg)
        self.assertIn('type = "ScaffoldSimilarity"', cfg)
        self.assertIn("bucket_size = 10", cfg)
        self.assertIn("minscore = 0.4", cfg)
        self.assertIn("minsimilarity = 0.4", cfg)

        self.assertIn("[[stage]]", cfg)
        self.assertIn('chkpt_file = "', cfg)
        self.assertIn('termination = "simple"', cfg)
        self.assertIn("min_steps = 1", cfg)
        self.assertIn("max_steps = 10", cfg)

        self.assertIn("[stage.scoring]", cfg)
        self.assertIn('type = "geometric_mean"', cfg)

        self.assertIn("[stage.scoring.component.custom_alerts]", cfg)
        self.assertIn('name = "unwanted_alerts"', cfg)
        self.assertIn("params.smarts = [", cfg)

    def test_build_toml_action_endpoint(self):
        net = self._create_reinvent_net(reverse("reinvent-net-list"), build=False)
        _ = self._ensure_net_artifacts(net)

        scheme, env, agent, gen = self._mk_scheme_env_agent_gen(net, add_diversity=False)

        models.ReinventStage.objects.create(
            generator=gen,
            order=0,
            termination_type="simple",
            max_score=1.0,
            min_steps=1,
            max_steps=5,
            scoring_source="inline",
            scoring_scheme=scheme,
        )

        url = reverse("reinvent-build-toml", args=[gen.id])
        resp = self.client.post(url, data={"device": "cpu"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)

        toml_path = resp.data["toml_path"]
        self.assertTrue(os.path.isfile(toml_path))

        cfg = open(toml_path, "r", encoding="utf-8").read()
        self.assertIn('run_type = "staged_learning"', cfg)
        self.assertIn("[parameters]", cfg)
        self.assertIn("[learning_strategy]", cfg)
        self.assertIn("[[stage]]", cfg)
        self.assertIn("[stage.scoring]", cfg)

    @patch("genui.generators.extensions.genuireinvent.models.subprocess.Popen", autospec=True)
    def test_run_staged_learning_cli_smoke(self, popen_mock):
        """
        Unit test: do NOT run the real REINVENT binary.
        We only verify:
          - TOML is produced
          - subprocess output is logged
          - non-zero exit raises RuntimeError
        """

        class _FakeProc:
            def __init__(self, *args, **kwargs):
                self.stdout = ["fake reinvent output\n"]

            def wait(self):
                return 1  # simulate failure

        popen_mock.return_value = _FakeProc()

        net = self._create_reinvent_net(reverse("reinvent-net-list"), build=False)
        _ = self._ensure_net_artifacts(net)

        scheme, env, agent, gen = self._mk_scheme_env_agent_gen(net, add_diversity=False)

        models.ReinventStage.objects.create(
            generator=gen,
            order=0,
            termination_type="simple",
            max_score=1.0,
            min_steps=1,
            max_steps=3,
            scoring_source="inline",
            scoring_scheme=scheme,
        )

        # Ensure run_staged_learning doesn't fail before Popen due to missing binary
        # (it checks settings.REINVENT_BIN / env / shutil.which("reinvent"))
        fake_bin = shutil.which("python") or shutil.which("bash") or "/bin/echo"
        with override_settings(REINVENT_BIN=fake_bin):
            with self.assertRaises(RuntimeError):
                gen.run_staged_learning(device="cpu")

        # Log should be written even on failure
        log_path = agent.get_rl_log_path()
        self.assertTrue(os.path.isfile(log_path))
        log_txt = open(log_path, "r", encoding="utf-8").read()
        self.assertIn("[CMD]", log_txt)
        self.assertIn("fake reinvent output", log_txt)