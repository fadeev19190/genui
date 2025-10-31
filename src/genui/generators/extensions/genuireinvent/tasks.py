from celery import shared_task
from types import SimpleNamespace

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
    try:
        model_cls = getattr(models, model_class)
        instance = model_cls.objects.get(pk=model_id)
        builder_cls = getObjectAndModuleFromFullName(builder_class)[0]
        recorder = ProgressRecorder(self)

        if hasattr(instance, 'parent') and instance.parent_id:
            builder = builder_cls(instance, instance.parent, progress=recorder)
        else:
            builder = builder_cls(instance, progress=recorder)

        # run the build inline (eager) or in the worker
        builder.build()

        # return a Task-like object so the view can safely do "task.id"
        return SimpleNamespace(
            id=getattr(self.request, "id", None),
            result={
                "errors": [repr(x) for x in builder.errors],
                "ReinventModelName": instance.name,
                "ReinventModelID": instance.id,
            },
        )
    finally:
        cleanup()
