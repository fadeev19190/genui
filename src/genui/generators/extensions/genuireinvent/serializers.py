# genui/src/genui/generators/extensions/genuireinvent/serializers.py

from django.db.models import Q
from rest_framework import serializers

from genui.compounds.models import MolSet
from genui.compounds.serializers import MolSetSerializer
from genui.models.models import (
    ModelPerformanceMetric,
    Algorithm,           # FIX: import Algorithm from the right place
)
from genui.models.serializers import (
    ValidationStrategySerializer,
    TrainingStrategySerializer,
    TrainingStrategyInitSerializer,
    ModelSerializer,
    ValidationStrategyInitSerializer,
)
from . import models
from genui.projects.models import Project


# ---------- Validation ----------

class ReinventValidationStrategySerializer(ValidationStrategySerializer):
    class Meta:
        model = models.ReinventNetValidation
        fields = ValidationStrategySerializer.Meta.fields + ("validSetSize",)


class ReinventValidationStrategyInitSerializer(ValidationStrategyInitSerializer):
    # Allow passing metrics explicitly; otherwise we will auto-fill them in create()
    metrics = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=ModelPerformanceMetric.objects.all(),
        required=False
    )

    class Meta:
        model = models.ReinventNetValidation
        fields = ReinventValidationStrategySerializer.Meta.fields   # includes "validSetSize"


# ---------- Training ----------

class ReinventTrainingStrategySerializer(TrainingStrategySerializer):
    """Expose REINVENT-specific hyperparameters in the strategy representation."""
    class Meta:
        model = models.ReinventNetTraining
        fields = TrainingStrategySerializer.Meta.fields + (
            "epochs",
            "batch_size",
            "sample_batch_size",      # NEW
            "save_every_n_epochs",    # NEW
        )


class ReinventTrainingStrategyInitSerializer(TrainingStrategyInitSerializer):
    """Init serializer must also accept REINVENT hyperparameters."""
    class Meta:
        model = models.ReinventNetTraining
        fields = TrainingStrategyInitSerializer.Meta.fields + (
            "epochs",
            "batch_size",
            "sample_batch_size",      # NEW
            "save_every_n_epochs",    # NEW
        )


# ---------- Model (ReinventNet) ----------

class ReinventNetSerializer(ModelSerializer):
    molset = MolSetSerializer(many=False, required=False, allow_null=True)
    # Override fields to use our specialized serializers,
    # but DO NOT re-list them in Meta.fields (base ModelSerializer already includes them).
    trainingStrategy = ReinventTrainingStrategySerializer(many=False)
    validationStrategy = ReinventValidationStrategySerializer(many=False, required=False)
    parent = serializers.SerializerMethodField("get_parent")

    class Meta:
        model = models.ReinventNet
        # Keep base fields, drop 'performance' if present, and add 'molset' and 'parent'
        fields = [f for f in ModelSerializer.Meta.fields if f != "performance"] + ["molset", "parent"]
        read_only_fields = [f for f in ModelSerializer.Meta.read_only_fields if f != "performance"]

    def get_parent(self, obj):
        """Return a lightweight parent representation to avoid deep recursion."""
        if obj.parent_id:
            return {"id": obj.parent_id, "name": getattr(obj.parent, "name", None)}
        return None


class ReinventNetInitSerializer(ReinventNetSerializer):
    molset = serializers.PrimaryKeyRelatedField(
        many=False, queryset=MolSet.objects.all(), required=False, allow_null=True
    )
    trainingStrategy = ReinventTrainingStrategyInitSerializer(many=False)
    validationStrategy = ReinventValidationStrategyInitSerializer(many=False, required=False)
    parent = serializers.PrimaryKeyRelatedField(
        many=False, queryset=models.ReinventNet.objects.all(), required=False, allow_null=True
    )

    class Meta:
        model = models.ReinventNet
        fields = ReinventNetSerializer.Meta.fields
        read_only_fields = ReinventNetSerializer.Meta.read_only_fields

    def create(self, validated_data, **kwargs):
        """
        Create ReinventNet and attach training/validation strategies.
        """
        # Pop nested payloads so the base create() doesn't choke on them
        ts_data = validated_data.pop("trainingStrategy")
        vs_data = validated_data.pop("validationStrategy", None)
        parent = validated_data.pop("parent", None)
        molset = validated_data.pop("molset", None)

        # Create the model instance; explicitly pass molset if provided
        instance = super().create(validated_data, molset=molset, **kwargs)

        # Link parent if present
        if parent:
            instance.parent = parent
            instance.save(update_fields=["parent"])

        # Create training strategy with REINVENT hyperparameters
        trainingStrategy = models.ReinventNetTraining.objects.create(
            modelInstance=instance,
            algorithm=ts_data["algorithm"],
            mode=ts_data["mode"],
            epochs=ts_data.get("epochs", models.ReinventNetTraining._meta.get_field("epochs").default),
            batch_size=ts_data.get("batch_size", models.ReinventNetTraining._meta.get_field("batch_size").default),
            sample_batch_size=ts_data.get("sample_batch_size",
                                          models.ReinventNetTraining._meta.get_field("sample_batch_size").default),
            # NEW
            save_every_n_epochs=ts_data.get("save_every_n_epochs",
                                            models.ReinventNetTraining._meta.get_field("save_every_n_epochs").default),
            # NEW
        )
        # Persist algorithm-specific parameters (if any) via helper from base ModelSerializer
        self.saveParameters(trainingStrategy, ts_data)

        # Create validation strategy (optional)
        if vs_data is not None:
            validationStrategy = models.ReinventNetValidation.objects.create(
                modelInstance=instance,
                validSetSize=vs_data.get(
                    "validSetSize",
                    models.ReinventNetValidation._meta.get_field("validSetSize").default
                ),
            )
            # Use provided metrics or auto-select based on chosen mode/algorithm
            if "metrics" in vs_data and vs_data["metrics"]:
                validationStrategy.metrics.set(vs_data["metrics"])
            else:
                # Select metrics valid for the chosen mode and algorithm
                validationStrategy.metrics.set(
                    ModelPerformanceMetric.objects
                    .filter(validModes=ts_data["mode"])
                    .filter(Q(validAlgorithms=ts_data["algorithm"]) | Q(validAlgorithms__isnull=True))
                    .distinct()
                )
            validationStrategy.save()

        # Do NOT create any generator object here unless such a class actually exists.
        return instance


# class ScoringFunctionSerializer(serializers.HyperlinkedModelSerializer):
#     project = serializers.PrimaryKeyRelatedField(many=False, queryset=Project.objects.all())
#
#     class Meta:
#         model = models.ScoringMethod
#         fields = ('id', 'name', 'description', 'created', 'updated', 'project')
#         read_only_fields = ('id', 'created', 'updated', )