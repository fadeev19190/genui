# genui/generators/extensions/genuireinvent/urls.py

from django.urls import path, include
from rest_framework import routers
from genui.utils.extensions.tasks.views import ModelTasksView
from genui.models.views import ModelFileView, ModelPerformanceListView
from . import models, views

router = routers.DefaultRouter()
router.register(r'reinvent/networks', views.ReinventNetViewSet, basename='reinvent-net')  # <-- ВАЖНО: reinvent-net

routes = [
    path('reinvent/networks/<int:pk>/tasks/all/', ModelTasksView.as_view(model_class=models.ReinventNet)),
    path('reinvent/networks/<int:pk>/tasks/started/', ModelTasksView.as_view(started_only=True, model_class=models.ReinventNet)),
    path('reinvent/networks/<int:pk>/performance/', ModelPerformanceListView.as_view(), name="reinvent_net_perf_view"),
    path('reinvent/networks/<int:pk>/files/', ModelFileView.as_view(model_class=models.ReinventNet), name="reinvent-net-model-files-list"),
]

urlpatterns = [
    path('', include(routes)),
    path('', include(router.urls)),
]