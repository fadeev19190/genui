# --- ADD THESE IMPORTS near the top of your tests.py ---
import os
import json
import shutil
import datetime
import unittest

from django.apps import apps
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
REINVENT_BIN = "/opt/anaconda3/envs/reinvent4/bin/reinvent"  # set to your working reinvent CLI


class SetUpReinventMixIn(QSARModelInit):
    """
    Minimal setup for creating a ReinventNet via REST (transfer learning),
    and helpers for staged-learning objects aligned with current schema.
    """

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

        os.environ.setdefault("REINVENT_BIN", REINVENT_BIN)

        repo_files = os.path.abspath(
            os.path.join(settings.BASE_DIR, os.pardir, os.pardir, "files")
        )
        media_debug = os.path.join(repo_files, "media_debug")
        os.makedirs(media_debug, exist_ok=True)
        os.makedirs(os.path.join(repo_files, "checkpoints", "prior"), exist_ok=True)

        settings.MEDIA_ROOT = media_debug
        settings.GENUI_SETTINGS = {**settings.GENUI_SETTINGS, "FILES_DIR": repo_files}

        # Sanity: required prior must exist at the absolute path used by extension model code
        prior_abs = models.PRIOR_ABS
        if not os.path.isfile(prior_abs):
            self.fail(
                f"Required prior not found at {prior_abs}. "
                f"Either place it there or change PRIOR_ABS in models.py for tests."
            )

    # ---------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------
    def _snapshot_artifacts(self, label: str):
        src = settings.MEDIA_ROOT
        debug_root = os.path.join(
            os.path.abspath(os.path.join(settings.BASE_DIR, os.pardir, os.pardir)),
            "files",
            "debug_artifacts",
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
            self._snapshot_artifacts(self._testMethodName)
        finally:
            super().tearDown()

    def _create_reinvent_net(self, url, initial=None):
        """
        POST a ReinventNet (transfer learning). Returns DB instance.
        """
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

    # ---------------------------------------------------------------------
    # Schema-aligned creation helpers (post-migrations)
    # ---------------------------------------------------------------------
    def _create_with_model_fields(self, model_cls, **kwargs):
        field_names = {f.name for f in model_cls._meta.get_fields()}
        filtered = {k: v for k, v in kwargs.items() if k in field_names}
        return model_cls.objects.create(**filtered)

    def _ensure_dataset_links(self, model_cls, kwargs: dict) -> dict:
        """
        For DataSet/ActivitySet-derived models after your migrations:
          - project is required (projects.models.Project-linked base save())
          - molecules is required for ActivitySet (your DB error showed molecules_id NOT NULL)
        Uses introspection, so it won't break on small schema changes.
        """
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
        """
        Your DB table is models_model, so the app label is usually 'models'.
        Try common names used across GenUI versions.
        """
        for app_label in ("models", "genui_models", "genui"):
            for cls_name in ("Builder", "ModelBuilder"):
                try:
                    return apps.get_model(app_label, cls_name)
                except Exception:
                    continue
        raise RuntimeError("Could not locate Builder model (tried Builder/ModelBuilder).")

    def _create_builder_for(self, *, model_class_name: str):
        """
        Create a Builder row that satisfies NOT NULL constraints, using introspection.
        This fixes: IntegrityError null value in column 'builder_id' of relation 'models_model'.
        """
        Builder = self._get_builder_model()

        kwargs = {}
        for f in Builder._meta.fields:
            # Skip PK / auto fields
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

                # If we don't know, we can't auto-create safely.
                # Most Builder schemas won't require unknown FK here.
                continue

            # Scalars / choices
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
                # last resort for unknown required fields
                kwargs[f.name] = "test"

        return Builder.objects.create(**kwargs)

    def _create_model_like(self, model_cls, **kwargs):
        """
        Create Model/Generator-derived rows after migrations:
          - ensure project
          - ensure builder
          - ensure algorithm/mode when present (best-effort)
        """
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
        """
        TrainingStrategy-derived rows after migrations:
          - modelInstance is NOT NULL
          - algorithm/mode often NOT NULL
        We intentionally use model_instance=net to avoid circular creation with ReinventAgent.
        """
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
    """
    Post-migrations staged-learning tests aligned with genuireinvent/models.py:
      - DataSet/ActivitySet required fields
      - TrainingStrategy.modelInstance required
      - Model.builder required
    """

    def _mk_scheme_env_agent_gen(self, net: models.ReinventNet, *, add_diversity=False):
        # Reward scheme is ActivitySet => requires project+molecules
        scheme = self._create_dataset_like(
            models.ReinventEnvironmentScores,
            aggregation_type="geometric_mean",
        )

        # At least one component so scoring exists
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

        # Environment is DataSet => requires project+molecules
        env = self._create_dataset_like(
            models.ReinventEnvironment,
            name="Test RL Environment",
            prior_net=net,
            agent_net=net,
            diversity_filter=df,
            reward_scheme=scheme,
        )

        # TrainingStrategy requires modelInstance => use net (avoids circular dependency)
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

        # ReinventAgent is Model => requires builder (helper injects it)
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

        # Reinvent is Generator (usually Model-derived) => also may require builder
        gen = self._create_model_like(
            models.Reinvent,
            name="Test Staged Learning Run",
            description="reinvent staged learning run for tests",
            environment=env,
            agent=agent,
        )

        return scheme, env, agent, gen

    def _ensure_net_checkpoint(self, net: models.ReinventNet):
        """
        Make sure the net has an actual checkpoint file on disk.
        Depending on your build pipeline, the checkpoint may exist only after TL runs.
        """
        # Prepare split/corpus first (needed for TL toml)
        net.prepareData()

        # Run TL once to ensure checkpoint exists and is non-empty
        out = net.run_transfer_learning(device="cpu")
        self.assertTrue(os.path.isfile(out), f"Expected TL checkpoint at {out}")
        self.assertTrue(os.path.getsize(out) > 0, f"Checkpoint is empty at {out}")

        return net.checkpointFile

    def test_build_staged_learning_toml_schema_correct(self):
        net = self._create_reinvent_net(reverse("reinvent-net-list"))
        ckpt_mf = self._ensure_net_checkpoint(net)
        self.assertTrue(os.path.isfile(ckpt_mf.path))

        scheme, env, agent, gen = self._mk_scheme_env_agent_gen(net, add_diversity=True)

        # One stage, inline scoring using scheme
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

        # Top-level (from ReinventAgent.build_staged_toml)
        self.assertIn('run_type = "staged_learning"', cfg)
        self.assertIn('device = "cpu"', cfg)
        self.assertIn('tb_logdir = "', cfg)
        self.assertIn('json_out_config = "_staged_learning.json"', cfg)

        # Parameters section
        self.assertIn("[parameters]", cfg)
        self.assertIn("use_checkpoint = false", cfg)
        self.assertIn('summary_csv_prefix = "reinvent"', cfg)
        self.assertIn('prior_file = "', cfg)  # env.get_prior_path()
        self.assertIn('agent_file = "', cfg)  # env.get_agent_path()
        self.assertIn("batch_size = 16", cfg)

        # Learning strategy
        self.assertIn("[learning_strategy]", cfg)
        self.assertIn('type = "dap"', cfg)
        self.assertIn("sigma = 64.0", cfg)
        self.assertIn("rate = 0.0005", cfg)

        # Diversity filter exists
        self.assertIn("[diversity_filter]", cfg)
        self.assertIn('type = "ScaffoldSimilarity"', cfg)
        self.assertIn("bucket_size = 10", cfg)
        self.assertIn("minscore = 0.4", cfg)
        self.assertIn("minsimilarity = 0.4", cfg)

        # Stage layout
        self.assertIn("[[stage]]", cfg)
        self.assertIn('chkpt_file = "', cfg)
        self.assertIn('termination = "simple"', cfg)
        self.assertIn("min_steps = 1", cfg)
        self.assertIn("max_steps = 10", cfg)

        # Inline scoring block
        self.assertIn("[stage.scoring]", cfg)
        self.assertIn('type = "geometric_mean"', cfg)

        # UnwantedSmarts scorer block
        self.assertIn("[stage.scoring.component.custom_alerts]", cfg)
        self.assertIn('name = "unwanted_alerts"', cfg)
        self.assertIn("params.smarts = [", cfg)

    def test_build_toml_action_endpoint(self):
        net = self._create_reinvent_net(reverse("reinvent-net-list"))
        _ = self._ensure_net_checkpoint(net)

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

    @unittest.skipUnless(
        (os.environ.get("REINVENT_BIN") and os.path.isfile(os.environ.get("REINVENT_BIN")))
        or shutil.which("reinvent"),
        "REINVENT CLI not available (set REINVENT_BIN or ensure `reinvent` is on PATH).",
    )
    def test_run_staged_learning_cli_smoke(self):
        net = self._create_reinvent_net(reverse("reinvent-net-list"))
        _ = self._ensure_net_checkpoint(net)

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

        toml_path = gen.run_staged_learning(device="cpu")
        self.assertTrue(os.path.isfile(toml_path))

        # RL log is written by ReinventAgent.run_staged_learning
        log_path = agent.get_rl_log_path()
        self.assertTrue(os.path.isfile(log_path))
        log_txt = open(log_path, "r", encoding="utf-8").read()
        self.assertIn("[CMD]", log_txt)
        self.assertTrue(len(log_txt.strip()) > 0)