# genui/generators/extensions/genuireinvent/serializers.py
from django.db.models import Q
from rest_framework import serializers

from genui.compounds.models import MolSet
from genui.compounds.serializers import MolSetSerializer
from genui.models.models import ModelPerformanceMetric
from genui.models.serializers import (
    ValidationStrategySerializer,
    TrainingStrategySerializer,
    TrainingStrategyInitSerializer,
    ModelSerializer,
    ValidationStrategyInitSerializer,
)
from . import models


class ReinventValidationStrategySerializer(ValidationStrategySerializer):
    class Meta:
        model = models.ReinventNetValidation
        fields = ValidationStrategySerializer.Meta.fields + ("validSetSize", "split_method", "valid_fraction", "random_seed", "temporal_cutoff")


class ReinventValidationStrategyInitSerializer(ValidationStrategyInitSerializer):
    # Optional explicit metrics
    metrics = serializers.PrimaryKeyRelatedField(
        many=True, queryset=ModelPerformanceMetric.objects.all(), required=False
    )

    class Meta:
        model = models.ReinventNetValidation
        fields = ReinventValidationStrategySerializer.Meta.fields


class ReinventTrainingStrategySerializer(TrainingStrategySerializer):
    class Meta(TrainingStrategySerializer.Meta):
        model = models.ReinventNetTraining

        _base_fields = tuple(getattr(TrainingStrategySerializer.Meta, "fields", ()))
        _extra_fields = (
            "epochs",
            "batch_size",
            "sample_batch_size",
            "save_every_n_epochs",
            "best_epoch", "best_valid_loss",)
        fields = _base_fields + _extra_fields

        _base_ro = tuple(getattr(TrainingStrategySerializer.Meta, "read_only_fields", ()))
        read_only_fields = _base_ro + ("best_epoch", "best_valid_loss", "started_at", "finished_at", "device")


class ReinventTrainingStrategyInitSerializer(TrainingStrategyInitSerializer):
    class Meta(TrainingStrategyInitSerializer.Meta):
        model = models.ReinventNetTraining

        _base_fields = tuple(getattr(TrainingStrategyInitSerializer.Meta, "fields", ()))
        _extra_fields = (
            "epochs",
            "batch_size",
            "sample_batch_size",
            "save_every_n_epochs",
        )
        fields = _base_fields + _extra_fields


class ReinventNetSerializer(ModelSerializer):
    molset = MolSetSerializer(many=False, required=False, allow_null=True)
    trainingStrategy = ReinventTrainingStrategySerializer(many=False)
    validationStrategy = ReinventValidationStrategySerializer(many=False, required=False)
    parent = serializers.SerializerMethodField()

    bestEpoch = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = models.ReinventNet
        # include bestEpoch in fields
        fields = [f for f in ModelSerializer.Meta.fields if f != "performance"] + [
            "molset", "parent", "bestEpoch",
        ]
        # keep performance read-only behavior, add bestEpoch
        read_only_fields = [f for f in ModelSerializer.Meta.read_only_fields if f != "performance"] + [
            "bestEpoch",
        ]

    def get_parent(self, obj):
        if obj.parent_id:
            return {"id": obj.parent_id, "name": getattr(obj.parent, "name", None)}
        return None

    def get_bestEpoch(self, obj):
        ts = getattr(obj, "trainingStrategy", None)
        return getattr(ts, "best_epoch", None)



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
        IMPORTANT: Do NOT pop trainingStrategy before super().create().
        The base ModelSerializer uses it to resolve and set builder_id.
        """
        # keep the nested payloads intact for super().create()
        instance = super().create(
            validated_data,
            molset=validated_data.get("molset"),
            **kwargs
        )

        # parent (optional)
        parent = validated_data.get("parent")
        if parent:
            instance.parent = parent
            instance.save(update_fields=["parent"])

        # training strategy — create concrete ReinventNetTraining record
        ts_data = validated_data["trainingStrategy"]
        trainingStrategy = models.ReinventNetTraining.objects.create(
            modelInstance=instance,
            algorithm=ts_data["algorithm"],
            mode=ts_data["mode"],
            epochs=ts_data.get(
                "epochs",
                models.ReinventNetTraining._meta.get_field("epochs").default
            ),
            batch_size=ts_data.get(
                "batch_size",
                models.ReinventNetTraining._meta.get_field("batch_size").default
            ),
            sample_batch_size=ts_data.get(
                "sample_batch_size",
                models.ReinventNetTraining._meta.get_field("sample_batch_size").default
            ),
            save_every_n_epochs=ts_data.get(
                "save_every_n_epochs",
                models.ReinventNetTraining._meta.get_field("save_every_n_epochs").default
            ),
        )
        self.saveParameters(trainingStrategy, ts_data)

        # validation strategy (optional)
        vs_data = validated_data.get("validationStrategy")
        if vs_data:
            validationStrategy = models.ReinventNetValidation.objects.create(
                modelInstance=instance,
                validSetSize=vs_data.get(
                    "validSetSize",
                    models.ReinventNetValidation._meta.get_field("validSetSize").default
                ),
            )
            # pick default metrics by mode/algorithm if not specified
            metrics = vs_data.get("metrics")
            if metrics:
                validationStrategy.metrics.set(metrics)
            else:
                validationStrategy.metrics.set(
                    ModelPerformanceMetric.objects
                    .filter(validModes=ts_data["mode"])
                    .filter(Q(validAlgorithms=ts_data["algorithm"]) | Q(validAlgorithms__isnull=True))
                    .distinct()
                )
            validationStrategy.save()

        return instance

# =====================================================================
# 3. RL / ENVIRONMENT CONFIG SERIALIZERS
# =====================================================================

class ReinventDiversityFilterSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventDiversityFilter
        fields = "__all__"


class ScoreModifierSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ScoreModifier
        fields = "__all__"


class ReinventEnvironmentScoresSerializer(serializers.ModelSerializer):
    """
    Aggregation scheme. Exposes related scoring components via read-only
    nested lists for convenience.
    """

    property_scorers = serializers.PrimaryKeyRelatedField(
        many=True,
        read_only=True,
        source="propertyscorer_set",
    )
    model_scorers = serializers.PrimaryKeyRelatedField(
        many=True,
        read_only=True,
        source="genuimodelscorer_set",
    )
    unwanted_smarts_scorers = serializers.PrimaryKeyRelatedField(
        many=True,
        read_only=True,
        source="unwantedsmartsscorer_set",
    )

    class Meta:
        model = models.ReinventEnvironmentScores
        fields = (
            "id",
            "aggregation_type",
            "property_scorers",
            "model_scorers",
            "unwanted_smarts_scorers",
        )


class PropertyScorerSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.PropertyScorer
        fields = "__all__"


class GenUIModelScorerSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.GenUIModelScorer
        fields = "__all__"


class UnwantedSmartsScorerSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.UnwantedSmartsScorer
        fields = "__all__"


class ReinventEnvironmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventEnvironment
        fields = "__all__"


# =====================================================================
# 4. RL AGENT CONFIG SERIALIZERS
# =====================================================================

class ReinventAgentTrainingSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventAgentTraining
        fields = "__all__"


class ReinventAgentValidationSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventAgentValidation
        fields = "__all__"


class ReinventAgentSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventAgent
        fields = "__all__"


# =====================================================================
# 5. STAGED LEARNING / RL RUN SERIALIZERS
# =====================================================================

class ReinventStageSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventStage
        fields = "__all__"


class ReinventSerializer(serializers.ModelSerializer):
    """
    Minimal serializer for the staged-learning runner.
    Relationships are exposed as primary keys; you can wire up
    nested serializers or separate endpoints as needed.
    """

    class Meta:
        model = models.Reinvent
        fields = "__all__"


# =====================================================================
# 6. PERFORMANCE LOGGING
# =====================================================================

class ModelPerformanceReinventSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ModelPerformanceReinvent
        fields = "__all__"