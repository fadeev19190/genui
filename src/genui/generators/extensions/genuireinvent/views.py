# genui/generators/extensions/genuireinvent/views.py
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from genui.models.views import ModelViewSet
from . import models, serializers
from .genuimodels import builders
from .tasks import buildReinventModel


class ReinventNetViewSet(ModelViewSet):
    """
    CRUD for ReinventNet + extra action to run REINVENT's preprocessor.
    All artifacts are saved as AUX ModelFiles (no ad-hoc directories).
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
        Runs REINVENT datapipeline, writes cleaned corpus preview to AUX file,
        and returns a simple summary.
        """
        try:
            net = self.get_queryset().get(pk=pk)

            # This populates the AUX file (media/models/...) and returns (train,test) handles.
            net.prepareData()
            train_mf = net.corpusFileTrain  # ModelFile (AUX)

            # Count non-empty lines (preview length)
            prepared = 0
            try:
                with open(train_mf.path, "r", encoding="utf-8") as fh:
                    prepared = sum(1 for ln in fh if ln.strip())
            except FileNotFoundError:
                prepared = 0

            # You can also expose train_mf.file.url if you want a browser-accessible link.
            return Response(
                {
                    "prepared": prepared,
                    "train_file": train_mf.path,     # or: train_mf.file.url
                    "note": train_mf.note,
                },
                status=status.HTTP_201_CREATED,
            )
        except models.ReinventNet.DoesNotExist:
            return Response({"error": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# =====================================================================
# RL / ENVIRONMENT CONFIG
# =====================================================================

class ReinventDiversityFilterViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventDiversityFilter.objects.all()
    serializer_class = serializers.ReinventDiversityFilterSerializer

    @action(detail=False, methods=["get"], url_path="available-types")
    def available_types(self, request):
        """
        GET /reinvent/diversity-filters/available-types/
        Returns the list of DiversityFilter classes discovered from REINVENT.
        """
        types_ = models.ReinventEnvironmentHelper.get_diversity_filters()
        return Response({"types": types_})


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


# =====================================================================
# AGENT CONFIG
# =====================================================================

class ReinventAgentTrainingViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventAgentTraining.objects.all()
    serializer_class = serializers.ReinventAgentTrainingSerializer

    @action(detail=False, methods=["get"], url_path="learning-strategies")
    def learning_strategies(self, request):
        """
        GET /reinvent/agent-training/learning-strategies/
        Returns the list of supported staged-learning strategies (e.g., "dap").
        """
        strategies = models.ReinventEnvironmentHelper.get_learning_strategies()
        return Response({"strategies": strategies})


class ReinventAgentValidationViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventAgentValidation.objects.all()
    serializer_class = serializers.ReinventAgentValidationSerializer


class ReinventAgentViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventAgent.objects.all()
    serializer_class = serializers.ReinventAgentSerializer


# =====================================================================
# STAGED LEARNING (REINVENT RL RUNNER)
# =====================================================================

class ReinventStageViewSet(viewsets.ModelViewSet):
    queryset = models.ReinventStage.objects.all()
    serializer_class = serializers.ReinventStageSerializer


class ReinventViewSet(viewsets.ModelViewSet):
    """
    CRUD for staged-learning configs + actions to build TOML and run RL.
    """
    queryset = models.Reinvent.objects.all()
    serializer_class = serializers.ReinventSerializer

    @action(detail=True, methods=["post"], url_path="build-toml")
    def build_toml(self, request, pk=None):
        """
        POST /reinvent/runs/{id}/build-toml/
        Build the staged-learning TOML and save it as AUX ModelFile.
        """
        reinvent = self.get_object()
        device = request.data.get("device", "cuda:0")
        try:
            toml_path = reinvent.build_staged_toml(device=device)
            return Response(
                {"toml_path": toml_path},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=["post"], url_path="run-staged-learning")
    def run_staged_learning(self, request, pk=None):
        """
        POST /reinvent/runs/{id}/run-staged-learning/
        Invoke the external REINVENT binary to perform staged learning.
        """
        reinvent = self.get_object()
        device = request.data.get("device", "cuda:0")
        try:
            toml_path = reinvent.run_staged_learning(device=device)
            return Response(
                {"toml_path": toml_path},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# =====================================================================
# PERFORMANCE LOGGING
# =====================================================================

class ModelPerformanceReinventViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only access to RL performance logs.

    Optional query params:
      - agent: filter by ReinventAgent id
      - stage_index: filter by RL stage index
    """
    serializer_class = serializers.ModelPerformanceReinventSerializer

    def get_queryset(self):
        qs = models.ModelPerformanceReinvent.objects.all().order_by("created")

        agent_id = self.request.query_params.get("agent")
        if agent_id:
            qs = qs.filter(agent_id=agent_id)

        stage_index = self.request.query_params.get("stage_index")
        if stage_index is not None:
            try:
                stage_index_int = int(stage_index)
                qs = qs.filter(stage_index=stage_index_int)
            except ValueError:
                # ignore bad values; return unfiltered by stage_index
                pass

        return qs