# genui/generators/extensions/genuireinvent/views.py
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from genui.models.views import ModelViewSet
from . import models, serializers
from .genuimodels import builders
from .tasks import buildReinventModel  # keep commented until builder is ready

class ReinventNetViewSet(ModelViewSet):
    """
    CRUD for ReinventNet + an extra action to prepare corpora via REINVENT's preprocessor.
    Scoped by a project (query param) and by owner (self.owner_relation).
    """
    queryset = models.ReinventNet.objects.order_by("-created")
    serializer_class = serializers.ReinventNetSerializer
    init_serializer_class = serializers.ReinventNetInitSerializer
    owner_relation = "project__owner"
    builder_class = builders.ReinventNetBuilder
    build_task = buildReinventModel  # don't trigger Celery until the builder is production-ready

    def get_builder_kwargs(self):
        return {"model_class": models.ReinventNet.__name__}

    @action(detail=True, methods=["post"], url_path="prepare-corpus")
    def prepare_corpus(self, request, pk=None):
        """
        POST /reinvent/networks/{id}/prepare-corpus/
        Генерирует очищенный SMILES-корпус в GENUI/files/corpora.
        """
        try:
            net = self.get_queryset().get(pk=pk)

            # Твоя текущая сигнатура prepareData(self) -> str
            cleaned_path = net.prepareData()

            return Response(
                {
                    "prepared": sum(1 for _ in net._read_smiles_from_path(cleaned_path)),
                    "train_file": cleaned_path,
                },
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response({"error": repr(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)