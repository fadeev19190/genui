# genui/generators/extensions/genuireinvent/views.py
from rest_framework import status
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