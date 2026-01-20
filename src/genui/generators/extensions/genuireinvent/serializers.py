# genui/generators/extensions/genuireinvent/serializers.py

from __future__ import annotations

from django.db.models import Q
from django.utils.translation import gettext_lazy as _
from rest_framework import serializers

from genui.compounds.models import MolSet
from genui.compounds.serializers import MolSetSerializer
from genui.models.models import ModelPerformanceMetric
from genui.models.serializers import (
    ModelSerializer,
    TrainingStrategyInitSerializer,
    TrainingStrategySerializer,
    ValidationStrategyInitSerializer,
    ValidationStrategySerializer,
)
from genui.projects.models import Project

from . import models


# =====================================================================
# 1) REINVENT NET (TL) STRATEGIES
# =====================================================================

class ReinventValidationStrategySerializer(ValidationStrategySerializer):
    class Meta:
        model = models.ReinventNetValidation
        fields = (
            ValidationStrategySerializer.Meta.fields
            + ("validSetSize", "split_method", "valid_fraction", "random_seed", "temporal_cutoff")
        )


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
            "best_epoch",
            "best_valid_loss",
        )
        fields = _base_fields + _extra_fields

        _base_ro = tuple(getattr(TrainingStrategySerializer.Meta, "read_only_fields", ()))
        read_only_fields = _base_ro + (
            "best_epoch",
            "best_valid_loss",
            "started_at",
            "finished_at",
            "device",
        )


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


# =====================================================================
# 2) REINVENT NET SERIALIZERS
# =====================================================================

class ReinventNetSerializer(ModelSerializer):
    molset = MolSetSerializer(many=False, required=False, allow_null=True)

    trainingStrategy = ReinventTrainingStrategySerializer(many=False)
    validationStrategy = ReinventValidationStrategySerializer(many=False, required=False)

    parent = serializers.SerializerMethodField()
    bestEpoch = serializers.SerializerMethodField(read_only=True)
    bestValidLoss = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = models.ReinventNet
        # keep base fields, but don't expose performance here (usually read-only/handled elsewhere)
        fields = [f for f in ModelSerializer.Meta.fields if f != "performance"] + [
            "molset",
            "parent",
            "bestEpoch",
            "bestValidLoss",
        ]
        read_only_fields = [f for f in ModelSerializer.Meta.read_only_fields if f != "performance"] + [
            "bestEpoch",
            "bestValidLoss",
        ]

    def get_parent(self, obj):
        if obj.parent_id:
            return {"id": obj.parent_id, "name": getattr(obj.parent, "name", None)}
        return None

    def get_bestEpoch(self, obj):
        ts = getattr(obj, "trainingStrategy", None)
        return getattr(ts, "best_epoch", None)

    def get_bestValidLoss(self, obj):
        ts = getattr(obj, "trainingStrategy", None)
        return getattr(ts, "best_valid_loss", None)


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
        Create ReinventNet + concrete training/validation strategies.

        IMPORTANT:
        - Do not remove trainingStrategy from validated_data before super().create(),
          because the base ModelSerializer may use it to resolve builder_id.
        """
        ts_data = dict(validated_data.get("trainingStrategy") or {})
        vs_data = dict(validated_data.get("validationStrategy") or {})
        parent = validated_data.get("parent")
        molset = validated_data.get("molset")

        instance = super().create(validated_data, molset=molset, **kwargs)

        if parent:
            instance.parent = parent
            instance.save(update_fields=["parent"])

        # Create concrete ReinventNetTraining
        if ts_data:
            trainingStrategy = models.ReinventNetTraining.objects.create(
                modelInstance=instance,
                algorithm=ts_data["algorithm"],
                mode=ts_data["mode"],
                epochs=ts_data.get(
                    "epochs",
                    models.ReinventNetTraining._meta.get_field("epochs").default,
                ),
                batch_size=ts_data.get(
                    "batch_size",
                    models.ReinventNetTraining._meta.get_field("batch_size").default,
                ),
                sample_batch_size=ts_data.get(
                    "sample_batch_size",
                    models.ReinventNetTraining._meta.get_field("sample_batch_size").default,
                ),
                save_every_n_epochs=ts_data.get(
                    "save_every_n_epochs",
                    models.ReinventNetTraining._meta.get_field("save_every_n_epochs").default,
                ),
            )
            self.saveParameters(trainingStrategy, ts_data)

        # Create concrete ReinventNetValidation (optional)
        if vs_data:
            validationStrategy = models.ReinventNetValidation.objects.create(
                modelInstance=instance,
                validSetSize=vs_data.get(
                    "validSetSize",
                    models.ReinventNetValidation._meta.get_field("validSetSize").default,
                ),
                split_method=vs_data.get(
                    "split_method",
                    models.ReinventNetValidation._meta.get_field("split_method").default,
                ),
                valid_fraction=vs_data.get(
                    "valid_fraction",
                    models.ReinventNetValidation._meta.get_field("valid_fraction").default,
                ),
                random_seed=vs_data.get(
                    "random_seed",
                    models.ReinventNetValidation._meta.get_field("random_seed").default,
                ),
                temporal_cutoff=vs_data.get("temporal_cutoff", None),
            )

            metrics = vs_data.get("metrics")
            if metrics:
                validationStrategy.metrics.set(metrics)
            else:
                # default metrics by algorithm/mode
                validationStrategy.metrics.set(
                    ModelPerformanceMetric.objects
                    .filter(validModes=ts_data.get("mode"))
                    .filter(
                        Q(validAlgorithms=ts_data.get("algorithm"))
                        | Q(validAlgorithms__isnull=True)
                    )
                    .distinct()
                )

            validationStrategy.save()

        return instance


# =====================================================================
# 3) RL / ENVIRONMENT CONFIG SERIALIZERS
# =====================================================================

class ReinventDiversityFilterSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ReinventDiversityFilter
        fields = "__all__"


class ScoreModifierSerializer(serializers.Serializer):
    """
    Polymorphic serializer for ScoreModifier subclasses:
      - ClippedScore
      - SmoothHump
    """

    TYPE_CLIPPED = "ClippedScore"
    TYPE_HUMP = "SmoothHump"
    TYPE_CHOICES = (TYPE_CLIPPED, TYPE_HUMP)

    id = serializers.IntegerField(read_only=True)
    type = serializers.ChoiceField(choices=[(t, t) for t in TYPE_CHOICES])

    # DataSet fields (inherited)
    project = serializers.PrimaryKeyRelatedField(queryset=Project.objects.all())
    name = serializers.CharField(max_length=256)
    description = serializers.CharField(max_length=10000, required=False, allow_blank=True, allow_null=True)

    created = serializers.DateTimeField(read_only=True)
    updated = serializers.DateTimeField(read_only=True)

    # ---- ClippedScore fields ----
    upper = serializers.FloatField(required=False, allow_null=True)
    lower = serializers.FloatField(required=False, allow_null=True)
    high = serializers.FloatField(required=False, allow_null=True)
    low = serializers.FloatField(required=False, allow_null=True)
    smooth = serializers.BooleanField(required=False)

    # ---- SmoothHump fields ----
    sigma = serializers.FloatField(required=False, allow_null=True)

    def _downcast(self, obj):
        for attr in ("clippedscore", "smoothhump"):
            try:
                child = getattr(obj, attr)
            except Exception:
                child = None
            if child is not None:
                return child
        return obj

    def _model_for_type(self, t: str):
        if t == self.TYPE_CLIPPED:
            return models.ClippedScore
        if t == self.TYPE_HUMP:
            return models.SmoothHump
        raise serializers.ValidationError({"type": _("Unknown modifier type.")})

    def _base_repr(self, obj) -> dict:
        return {
            "id": obj.pk,
            "type": obj.__class__.__name__,
            "project": getattr(obj, "project_id", None),
            "name": getattr(obj, "name", None),
            "description": getattr(obj, "description", None),
            "created": getattr(obj, "created", None),
            "updated": getattr(obj, "updated", None),
        }

    def to_representation(self, obj):
        obj = self._downcast(obj)
        data = self._base_repr(obj)

        if isinstance(obj, models.ClippedScore):
            data.update(
                upper=obj.upper,
                lower=obj.lower,
                high=obj.high,
                low=obj.low,
                smooth=obj.smooth,
                sigma=None,
            )
        elif isinstance(obj, models.SmoothHump):
            data.update(
                upper=obj.upper,
                lower=obj.lower,
                high=None,
                low=None,
                smooth=None,
                sigma=obj.sigma,
            )
        else:
            data.update(
                upper=getattr(obj, "upper", None),
                lower=getattr(obj, "lower", None),
                high=getattr(obj, "high", None),
                low=getattr(obj, "low", None),
                smooth=getattr(obj, "smooth", None),
                sigma=getattr(obj, "sigma", None),
            )

        return data

    def validate(self, attrs):
        t = attrs.get("type")

        if t == self.TYPE_CLIPPED:
            upper = attrs.get("upper")
            lower = attrs.get("lower")
            if upper is None:
                raise serializers.ValidationError({"upper": _("This field is required for ClippedScore.")})
            if lower is not None and lower >= upper:
                raise serializers.ValidationError({"lower": _("lower must be < upper.")})
            return attrs

        if t == self.TYPE_HUMP:
            # model defaults handle missing values
            return attrs

        raise serializers.ValidationError({"type": _("Invalid modifier type.")})

    def create(self, validated_data):
        t = validated_data.pop("type")
        ModelCls = self._model_for_type(t)

        allowed = {f.name for f in ModelCls._meta.fields}
        kwargs = {k: v for k, v in validated_data.items() if k in allowed}

        return ModelCls.objects.create(**kwargs)

    def update(self, instance, validated_data):
        instance = self._downcast(instance)

        t = validated_data.pop("type", None)
        if t and t != instance.__class__.__name__:
            raise serializers.ValidationError(
                {"type": _("Changing modifier type is not supported. Create a new modifier instead.")}
            )

        for k, v in validated_data.items():
            if hasattr(instance, k):
                setattr(instance, k, v)

        instance.save()
        return instance


class ReinventEnvironmentScoresSerializer(serializers.ModelSerializer):
    property_scorers = serializers.PrimaryKeyRelatedField(
        many=True, read_only=True, source="propertyscorer_set"
    )
    model_scorers = serializers.PrimaryKeyRelatedField(
        many=True, read_only=True, source="genuimodelscorer_set"
    )
    unwanted_smarts_scorers = serializers.PrimaryKeyRelatedField(
        many=True, read_only=True, source="unwantedsmartsscorer_set"
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
    prior_path = serializers.SerializerMethodField(read_only=True)
    agent_path = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = models.ReinventEnvironment
        fields = "__all__"

    def get_prior_path(self, obj):
        try:
            return obj.get_prior_path()
        except Exception:
            return None

    def get_agent_path(self, obj):
        try:
            return obj.get_agent_path()
        except Exception:
            return None


# =====================================================================
# 4) RL AGENT CONFIG SERIALIZERS
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
# 5) STAGED LEARNING / RL RUN SERIALIZERS
# =====================================================================

class ReinventStageSerializer(serializers.ModelSerializer):
    resolved_checkpoint_path = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = models.ReinventStage
        fields = "__all__"

    def get_resolved_checkpoint_path(self, obj):
        try:
            # mirror model logic; safe default path is internal to build_staged_toml
            return obj.resolve_checkpoint_path(default_path="")
        except Exception:
            return None


class ReinventSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Reinvent
        fields = "__all__"


class ReinventInitSerializer(ReinventSerializer):
    # write-through fields (stored on linked agent)
    tb_logdir = serializers.CharField(required=False, allow_blank=True, write_only=True)
    json_out_config = serializers.CharField(required=False, allow_blank=True, write_only=True)

    # mirror TL workflow: allow build=true on POST
    build = serializers.BooleanField(required=False, default=False, write_only=True)
    device = serializers.CharField(required=False, default="cuda:0", write_only=True)

    class Meta(ReinventSerializer.Meta):
        fields = "__all__"

    def create(self, validated_data):
        tb_logdir = validated_data.pop("tb_logdir", None)
        json_out_config = validated_data.pop("json_out_config", None)
        validated_data.pop("build", None)
        validated_data.pop("device", None)

        instance = super().create(validated_data)

        # write-through to agent
        agent = getattr(instance, "agent", None)
        if agent and (tb_logdir is not None or json_out_config is not None):
            fields = []
            if tb_logdir is not None:
                agent.tb_logdir = tb_logdir
                fields.append("tb_logdir")
            if json_out_config is not None:
                agent.json_out_config = json_out_config
                fields.append("json_out_config")
            if fields:
                agent.save(update_fields=fields)

        # staged learning MUST have at least one stage
        try:
            if not instance.stages.exists():
                models.ReinventStage.objects.create(
                    generator=instance,
                    order=1,
                    termination_type="simple",
                    max_score=1.0,
                    min_steps=1,
                    max_steps=100,
                    scoring_source="inline",
                )
        except Exception:
            pass

        return instance


# =====================================================================
# 6) PERFORMANCE LOGGING
# =====================================================================

class ModelPerformanceReinventSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ModelPerformanceReinvent
        fields = "__all__"