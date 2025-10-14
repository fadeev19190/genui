from celery import shared_task

from genui.compounds.models import MolSet, ActivityTypes, Activity
from genui.utils.extensions.tasks.progress import ProgressRecorder
from genui.utils.inspection import getObjectAndModuleFromFullName

from django.db import connections
from django.conf import settings as dj_settings

from . import models
# from .models import ReinventEnvironment, ReinventEnvironmentScores
from .torchutils import cleanup

@shared_task(name="BuildReinventModel", bind=True, queue='gpu')
def buildReinventModel(self, model_id, builder_class, model_class):
    # get the builder
    try:
        model_class = getattr(models, model_class)
        instance = model_class.objects.get(pk=model_id)
        builder_class = getObjectAndModuleFromFullName(builder_class)[0]
        recorder = ProgressRecorder(self)

        if hasattr(instance, 'parent'):
            builder = builder_class(
                instance,
                instance.parent,
                progress=recorder
            )
        else:
            builder = builder_class(
                instance,
                progress=recorder
            )

    # build the model
        try:
            builder.build()
            return {
                "errors": [repr(x) for x in builder.errors],
                "ReinventModelName": instance.name,
                "ReinventModelID": instance.id,
            }
        except Exception:
            raise
        finally:
            try:
                cleanup()
            finally:
                if not getattr(dj_settings, "CELERY_TASK_ALWAYS_EAGER", False):
                    connections.close_all()
    except Exception as e:
        raise e

@shared_task
def run_reinventnet(model_id: int) -> str:
    net = models.ReinventNet.get(pk=model_id)
    return net.getModel()