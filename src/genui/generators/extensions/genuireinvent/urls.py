from django.urls import path, include
from rest_framework import routers

from genui.utils.extensions.tasks.views import ModelTasksView
from genui.models.views import ModelFileView, ModelPerformanceListView

from . import models, views

router = routers.DefaultRouter()
router.register(r"reinvent/networks", views.ReinventNetViewSet, basename="reinvent-net")
router.register(r"reinvent/diversity-filters", views.ReinventDiversityFilterViewSet)
router.register(r"reinvent/score-modifiers", views.ScoreModifierViewSet)
router.register(r"reinvent/env-schemes", views.ReinventEnvironmentScoresViewSet)
router.register(r"reinvent/property-scorers", views.PropertyScorerViewSet)
router.register(r"reinvent/model-scorers", views.GenUIModelScorerViewSet)
router.register(r"reinvent/unwanted-smarts", views.UnwantedSmartsScorerViewSet)
router.register(r"reinvent/environments", views.ReinventEnvironmentViewSet)
router.register(r"reinvent/agent-training", views.ReinventAgentTrainingViewSet)
router.register(r"reinvent/agent-validation", views.ReinventAgentValidationViewSet)
router.register(r"reinvent/agents", views.ReinventAgentViewSet)
router.register(r"reinvent/runs", views.ReinventViewSet)
router.register(r"reinvent/stages", views.ReinventStageViewSet)
router.register(r"reinvent/performance", views.ModelPerformanceReinventViewSet, basename="reinvent-performance")

routes = [
    path(
        "reinvent/networks/<int:pk>/tasks/all/",
        ModelTasksView.as_view(model_class=models.ReinventNet),
    ),
    path(
        "reinvent/networks/<int:pk>/tasks/started/",
        ModelTasksView.as_view(started_only=True, model_class=models.ReinventNet),
    ),
    path(
        "reinvent/networks/<int:pk>/performance/",
        ModelPerformanceListView.as_view(model_class=models.ReinventNet),
        name="reinvent_net_perf_view",
    ),
    path(
        "reinvent/networks/<int:pk>/files/",
        ModelFileView.as_view(model_class=models.ReinventNet),
        name="reinvent-net-model-files-list",
    ),
]

urlpatterns = [
    path("", include(routes)),
    path("", include(router.urls)),
]