from django.contrib import admin

import genui.generators.extensions.genuireinvent.models
from . import models

@admin.register(models.Generator)
class GeneratorAdmin(admin.ModelAdmin):
    pass

@admin.register(genui.generators.extensions.genuireinvent.models.ReinventNet)
class ReinventNetAdmin(admin.ModelAdmin):
    pass

@admin.register(genui.generators.extensions.genuireinvent.models.ReinventAgent)
class ReinventAgentAdmin(admin.ModelAdmin):
    pass
