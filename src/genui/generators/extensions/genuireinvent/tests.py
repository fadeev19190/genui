# genuireinvent/tests.py
import os
import shutil
import tempfile
import unittest

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
from .tasks import buildReinventModel, runReinventStagedLearning


TEST_EPOCHS = 2


# ---------------------------------------------------------------------
# Hard dependency checks (NO fakes, NO dummy processes)
# ---------------------------------------------------------------------
def _get_reinvent_bin() -> str | None:
    return (
        getattr(settings, "REINVENT_BIN", None)
        or os.environ.get("REINVENT_BIN")
        or shutil.which("reinvent")
    )


def _assert_reinvent_runtime_available():
    # 1) REINVENT python package (needed for corpus preprocess)
    try:
        from reinvent.datapipeline import preprocess  # noqa: F401
    except Exception as e:
        raise AssertionError(
            "Missing Python dependency: reinvent.datapipeline.preprocess.\n"
            "Install the REINVENT python package (or ensure it is importable in the test environment)."
        ) from e

    # 2) Prior file
    try:
        prior_path = models._resolve_reinvent_prior_path()
    except FileNotFoundError as e:
        raise AssertionError(
            f"{e}\n\n"
            "Fix by either:\n"
            "  - setting REINVENT_PRIOR (env), or\n"
            "  - setting settings.REINVENT_PRIOR_PATH (or GENUI_SETTINGS['REINVENT_PRIOR_PATH']), or\n"
            "  - placing the prior under GENUI_SETTINGS['FILES_DIR']/checkpoints/prior/reinvent.prior\n"
        ) from e

    if not os.path.isfile(prior_path):
        raise AssertionError(f"Resolved REINVENT prior path does not exist: {prior_path}")

    # 3) REINVENT CLI binary
    reinvent_bin = _get_reinvent_bin()
    if not reinvent_bin:
        raise AssertionError(
            "REINVENT CLI binary not found.\n"
            "Fix by either:\n"
            "  - setting settings.REINVENT_BIN, or\n"
            "  - setting REINVENT_BIN env var, or\n"
            "  - ensuring `reinvent` is on PATH."
        )
    if not os.path.isfile(reinvent_bin) and shutil.which(reinvent_bin) is None:
        raise AssertionError(f"REINVENT bin was set but not found on disk/PATH: {reinvent_bin}")

    # If it's a file, ensure executable
    if os.path.isfile(reinvent_bin) and not os.access(reinvent_bin, os.X_OK):
        raise AssertionError(f"REINVENT bin exists but is not executable: {reinvent_bin}")


# ---------------------------------------------------------------------
# Shared setup + helpers (schema-introspective, post-migrations friendly)
# ---------------------------------------------------------------------
class SetUpReinventMixIn(QSARModelInit):
    """
    Setup:
      - creates admin
      - ensures Algorithm + Mode metadata exist
      - uses a temp MEDIA_ROOT so tests don't pollute repo
      - asserts real REINVENT runtime exists (no fakes)
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

        # ensure project ownership for queryset visibility
        if getattr(self.project, "owner_id", None) != self._admin.id:
            self.project.owner = self._admin
            self.project.save(update_fields=["owner"])

        # temp MEDIA_ROOT to avoid repo pollution
        self._tmp_media = tempfile.mkdtemp(prefix="reinvent_test_media_")
        self.addCleanup(lambda: shutil.rmtree(self._tmp_media, ignore_errors=True))

        old_media_root = getattr(settings, "MEDIA_ROOT", None)
        self.addCleanup(lambda: setattr(settings, "MEDIA_ROOT", old_media_root))
        settings.MEDIA_ROOT = self._tmp_media

        # hard-check real runtime (no fakes)
        _assert_reinvent_runtime_available()

    # ----------------- creation helpers (introspection) -----------------
    def _create_with_model_fields(self, model_cls, **kwargs):
        field_names = {f.name for f in model_cls._meta.get_fields()}
        filtered = {k: v for k, v in kwargs.items() if k in field_names}
        return model_cls.objects.create(**filtered)

    def _ensure_dataset_links(self, model_cls, kwargs: dict) -> dict:
        fields = {f.name: f for f in model_cls._meta.get_fields()}
        if "project" in fields and "project" not in kwargs:
            kwargs["project"] = self.project
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
            if getattr(f, "primary_key", False) or getattr(f, "auto_created", False):
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

            # choices
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

    # ----------------- API helpers -----------------
    def _create_reinvent_net_via_api(self, *, build: bool):
        url = reverse("reinvent-net-list")
        payload = {
            "name": "Test Reinvent Network",
            "description": "test",
            "project": self.project.id,
            "build": bool(build),
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
        resp = self.client.post(url, data=payload, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)
        return models.ReinventNet.objects.get(pk=resp.data["id"]), resp.data


# ---------------------------------------------------------------------
# Tests: Transfer learning integration (prepareData + TL subprocess)
# ---------------------------------------------------------------------
@override_settings(
    ROOT_URLCONF="genui.urls",
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class ReinventTransferLearningIntegrationTests(SetUpReinventMixIn, APITestCase):
    def test_prepare_corpus_endpoint_runs_datapipeline_and_writes_files(self):
        net, _ = self._create_reinvent_net_via_api(build=False)

        url = reverse("reinvent-net-prepare-corpus", kwargs={"pk": net.id})
        resp = self.client.post(url, data={}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED, msg=resp.data)

        # Returned paths must exist
        for key in ("train_file", "valid_file", "preview_file", "full_file"):
            p = resp.data.get(key)
            self.assertTrue(p and os.path.isfile(p), f"{key} missing or not a file: {p}")

        # And counts should be >= 0 (valid may be 0 for very small corpora)
        self.assertGreaterEqual(int(resp.data.get("prepared_train", 0)), 0)
        self.assertGreaterEqual(int(resp.data.get("prepared_valid", 0)), 0)

    def test_transfer_learning_direct_call_runs_and_writes_checkpoint_and_log(self):
        net, _ = self._create_reinvent_net_via_api(build=False)

        # 1) corpus
        train_mf, valid_mf = net.prepareData()
        self.assertTrue(os.path.isfile(train_mf.path))
        self.assertTrue(os.path.isfile(valid_mf.path))
        self.assertTrue(os.path.isfile(net.corpusFullFile.path))
        self.assertTrue(os.path.isfile(net.corpusPreviewFile.path))

        # 2) TL
        ckpt_path = net.run_transfer_learning(device="cpu")
        self.assertEqual(ckpt_path, net.checkpointFile.path)
        self.assertTrue(os.path.isfile(ckpt_path), f"Expected checkpoint at {ckpt_path}")
        self.assertGreater(os.path.getsize(ckpt_path), 0, "Checkpoint file is empty")

        # 3) TOML + log should exist
        self.assertTrue(os.path.isfile(net.tlTomlFile.path))
        self.assertTrue(os.path.isfile(net.trainLogFile.path))
        log_txt = open(net.trainLogFile.path, "r", encoding="utf-8").read()
        self.assertIn("[CMD]", log_txt)

        # 4) best_epoch parsing is optional (depends on REINVENT output)
        ts = net.trainingStrategy
        if getattr(ts, "best_epoch", None) is not None:
            self.assertIsInstance(ts.best_epoch, int)
        if getattr(ts, "best_valid_loss", None) is not None:
            self.assertIsInstance(ts.best_valid_loss, float)

    def test_transfer_learning_via_build_task_executes_full_pipeline(self):
        """
        This exercises the actual build pipeline:
          - ReinventNetViewSet create(build=True) enqueues BuildReinventModel
          - builder.getX() calls prepareData()
          - algorithm.fit() triggers CLI TL
        """
        net, data = self._create_reinvent_net_via_api(build=True)

        # If the viewset returns task_id, eager mode executes immediately anyway.
        # We assert the artifacts exist on disk after the build.
        net.refresh_from_db()

        # Builder pipeline should have created corpus + checkpoint + log
        self.assertTrue(os.path.isfile(net.corpusFullFile.path), "Corpus was not created by build pipeline")
        self.assertTrue(os.path.isfile(net.checkpointFile.path), "Checkpoint was not created by build pipeline")
        self.assertTrue(os.path.isfile(net.trainLogFile.path), "Training log was not created by build pipeline")
        self.assertGreater(os.path.getsize(net.checkpointFile.path), 0, "Checkpoint file is empty")


# ---------------------------------------------------------------------
# Tests: Staged learning integration (TOML build + RL subprocess via Celery)
# ---------------------------------------------------------------------
@override_settings(
    ROOT_URLCONF="genui.urls",
    CELERY_TASK_ALWAYS_EAGER=True,
    CELERY_TASK_EAGER_PROPAGATES=True,
)
class ReinventStagedLearningIntegrationTests(SetUpReinventMixIn, APITestCase):
    def _mk_scheme_env_agent_gen(self, net: models.ReinventNet, *, add_diversity: bool = False):
        # Reward scheme (ActivitySet/DataSet-like)
        scheme = self._create_dataset_like(
            models.ReinventEnvironmentScores,
            aggregation_type="geometric_mean",
        )

        # Minimal scoring component so scoring exists
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

        # TrainingStrategy requires modelInstance (avoid circular dependency by using net)
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
            description="agent",
            environment=env,
            training=train_cfg,
            validation=None,
            output_model=None,
            tb_logdir=os.path.join(settings.MEDIA_ROOT, "tb_rl"),
            json_out_config="_staged_learning.json",
        )

        gen = self._create_model_like(
            models.Reinvent,
            name="Test Reinvent Run",
            description="run",
            environment=env,
            agent=agent,
        )

        return scheme, env, agent, gen

    def _ensure_net_checkpoint(self, net: models.ReinventNet):
        net.prepareData()
        ckpt_path = net.run_transfer_learning(device="cpu")
        self.assertTrue(os.path.isfile(ckpt_path))
        self.assertGreater(os.path.getsize(ckpt_path), 0)
        return ckpt_path

    def test_build_staged_toml_action_endpoint(self):
        net, _ = self._create_reinvent_net_via_api(build=False)
        self._ensure_net_checkpoint(net)

        scheme, env, agent, gen = self._mk_scheme_env_agent_gen(net, add_diversity=True)

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

    def test_run_staged_learning_via_celery_task_writes_rl_log(self):
        net, _ = self._create_reinvent_net_via_api(build=False)
        self._ensure_net_checkpoint(net)

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

        # Execute the real task (eager mode => runs inline)
        res = runReinventStagedLearning.delay(gen.id, device="cpu")
        out = res.get()

        toml_path = out.get("toml_path")
        self.assertTrue(toml_path and os.path.isfile(toml_path))

        # RL log is written by ReinventAgent.run_staged_learning
        log_path = out.get("rl_log_path") or agent.get_rl_log_path()
        self.assertTrue(log_path and os.path.isfile(log_path))
        log_txt = open(log_path, "r", encoding="utf-8").read()
        self.assertIn("[CMD]", log_txt)
        self.assertTrue(len(log_txt.strip()) > 0)

    def test_run_staged_learning_endpoint_enqueues_task(self):
        net, _ = self._create_reinvent_net_via_api(build=False)
        self._ensure_net_checkpoint(net)

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

        url = reverse("reinvent-run-staged-learning", args=[gen.id])
        resp = self.client.post(url, data={"device": "cpu"}, format="json")
        self.assertEqual(resp.status_code, status.HTTP_202_ACCEPTED, msg=resp.data)

        # In eager mode the task executes immediately; check RL log exists.
        log_path = agent.get_rl_log_path()
        self.assertTrue(os.path.isfile(log_path))
        txt = open(log_path, "r", encoding="utf-8").read()
        self.assertIn("[CMD]", txt)