"""
genuisetup

Registers models from the genuireinvent extension into the default GenUI group.
"""

PARENT = "genui.generators"


def setup(*args, **kwargs):
    from genui.utils.init import createGroup
    from . import models

    createGroup(
        "GenUI_Users",
        [
            # Transfer learning / prior network
            models.ReinventNet,
            models.ReinventNetTraining,
            models.ReinventNetValidation,
            models.ModelPerformanceReinvent,
            # RL environment + scoring
            models.ReinventEnvironment,
            models.ReinventEnvironmentScores,
            models.ReinventDiversityFilter,
            models.ScoreModifier,
            models.ClippedScore,
            models.SmoothHump,
            models.ScoringMethod,
            models.PropertyScorer,
            models.GenUIModelScorer,
            models.UnwantedSmartsScorer,
            # RL agent + staged learning generator
            models.ReinventAgent,
            models.ReinventAgentTraining,
            models.ReinventAgentValidation,
            models.Reinvent,
            models.ReinventStage,
        ],
        force=kwargs.get("force", False),
    )