# genui/generators/extensions/genuireinvent/views.py

from __future__ import annotations

import logging

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from genui.models.views import ModelViewSet

from . import models, serializers
from .genuimodels import builders
from .tasks import buildReinventModel, runReinventStagedLearning

log = logging.getLogger(__name__)


class ReinventNetViewSet(ModelViewSet):
    """
    CRUD for ReinventNet + action to run REINVENT's preprocessor.
    Artifacts are stored as AUX ModelFiles (hashed paths under media/).
    """
    queryset = models.ReinventNet.objects.order_by("-created")
    serializer_class = serializers.ReinventNetSerializer
    init_serializer_class = serializers.ReinventNetInitSerializer
    owner_relation = "project__owner"
    builder_class = builders.ReinventNetBuilder
    build_task = buildReinventModel

    def get_builder_kwargs(self):
        return {"model_class": models.ReinventNet.__name__}

    @action(detail=True, methods=["post"], url_path="prepare-corpus")
    def prepare_corpus(self, request, pk=None):
        """
        POST /reinvent/networks/{id}/prepare-corpus/
        Runs datapipeline, writes cleaned corpus + train/valid splits to AUX files,
        and returns counts + file references.
        """
        try:
            net = self.get_object()
            train_mf, valid_mf = net.prepareData()

            def _count_lines(path: str) -> int:
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        return sum(1 for ln in fh if ln.strip())
                except FileNotFoundError:
                    return 0

            return Response(
                {
                    "prepared_train": _count_lines(train_mf.path),
                    "prepared_valid": _count_lines(valid_mf.path),
                    "train_file": train_mf.path,
                    "valid_file": valid_mf.path,
                    "preview_file": net.corpusPreviewFile.path,
                    "full_file": net.corpusFullFile.path,
                    "split_method": getattr(getattr(net, "validationStrategy", None), "split_method", None),
                },
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            log.exception("prepare_corpus failed for ReinventNet pk=%s", pk)
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReinventDiversityFilterViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventDiversityFilter.objects.all()
    serializer_class = serializers.ReinventDiversityFilterSerializer

    @action(detail=False, methods=["get"], url_path="available-types")
    def available_types(self, request):
        try:
            types_ = models.ReinventEnvironmentHelper.get_diversity_filters()
            return Response({"types": types_})
        except Exception as e:
            log.exception("available_types failed")
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ScoreModifierViewSet(viewsets.ModelViewSet):
    queryset = models.ScoreModifier.objects.all()
    serializer_class = serializers.ScoreModifierSerializer


class ReinventEnvironmentScoresViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventEnvironmentScores.objects.all()
    serializer_class = serializers.ReinventEnvironmentScoresSerializer


class PropertyScorerViewSet(viewsets.ModelViewSet):
    queryset = models.PropertyScorer.objects.all()
    serializer_class = serializers.PropertyScorerSerializer


class GenUIModelScorerViewSet(viewsets.ModelViewSet):
    queryset = models.GenUIModelScorer.objects.all()
    serializer_class = serializers.GenUIModelScorerSerializer


class UnwantedSmartsScorerViewSet(viewsets.ModelViewSet):
    queryset = models.UnwantedSmartsScorer.objects.all()
    serializer_class = serializers.UnwantedSmartsScorerSerializer


class ReinventEnvironmentViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventEnvironment.objects.all()
    serializer_class = serializers.ReinventEnvironmentSerializer


class ReinventAgentTrainingViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventAgentTraining.objects.all()
    serializer_class = serializers.ReinventAgentTrainingSerializer

    @action(detail=False, methods=["get"], url_path="learning-strategies")
    def learning_strategies(self, request):
        try:
            strategies = models.ReinventEnvironmentHelper.get_learning_strategies()
            return Response({"strategies": strategies})
        except Exception as e:
            log.exception("learning_strategies failed")
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReinventAgentValidationViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventAgentValidation.objects.all()
    serializer_class = serializers.ReinventAgentValidationSerializer


class ReinventAgentViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventAgent.objects.all()
    serializer_class = serializers.ReinventAgentSerializer


class ReinventStageViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventStage.objects.all()
    serializer_class = serializers.ReinventStageSerializer


class ReinventViewSet(viewsets.ModelViewSet):
    queryset = models.Reinvent.objects.order_by("-id")
    serializer_class = serializers.ReinventSerializer
    init_serializer_class = serializers.ReinventInitSerializer
    owner_relation = "project__owner"

    def get_serializer_class(self):
        if self.action in {"create", "update", "partial_update"}:
            return self.init_serializer_class
        return self.serializer_class

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        instance = serializer.save()

        build = bool(request.data.get("build", False))
        if build:
            device = request.data.get("device", "cuda:0")
            async_res = runReinventStagedLearning.delay(instance.id, device=device)
            out = serializers.ReinventSerializer(instance, context=self.get_serializer_context()).data
            out.update({"task_id": async_res.id, "device": device})
            return Response(out, status=status.HTTP_201_CREATED)

        out = serializers.ReinventSerializer(instance, context=self.get_serializer_context()).data
        return Response(out, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"], url_path="build-toml")
    def build_toml(self, request, pk=None):
        reinvent = self.get_object()
        device = request.data.get("device", "cuda:0")
        try:
            toml_path = reinvent.build_staged_toml(device=device)
            return Response({"toml_path": toml_path}, status=status.HTTP_201_CREATED)
        except Exception as e:
            log.exception("build_toml failed for Reinvent pk=%s", pk)
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["post"], url_path="run-staged-learning")
    def run_staged_learning(self, request, pk=None):
        reinvent = self.get_object()
        device = request.data.get("device", "cuda:0")
        try:
            async_res = runReinventStagedLearning.delay(reinvent.id, device=device)
            return Response(
                {"task_id": async_res.id, "reinvent_id": reinvent.id, "device": device},
                status=status.HTTP_202_ACCEPTED,
            )
        except Exception as e:
            log.exception("run_staged_learning enqueue failed for Reinvent pk=%s", pk)
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ModelPerformanceReinventViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = serializers.ModelPerformanceReinventSerializer

    def get_queryset(self):
        qs = models.ModelPerformanceReinvent.objects.all().order_by("created")

        agent_id = self.request.query_params.get("agent")
        if agent_id:
            qs = qs.filter(agent_id=agent_id)

        stage_index = self.request.query_params.get("stage_index")
        if stage_index is not None:
            try:
                qs = qs.filter(stage_index=int(stage_index))
            except (TypeError, ValueError):
                pass

        return qs