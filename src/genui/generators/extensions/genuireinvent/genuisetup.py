"""
genuisetup

Registers models from the genuireinvent extension into the default GenUI group
and ensures the REINVENT prior exists locally.
"""
import hashlib
import os
import tempfile
import urllib.request

PARENT = "genui.generators"


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_reinvent_prior(*, force: bool = False) -> str:
    """
    Ensures REINVENT prior exists at:
      GENUI_SETTINGS['FILES_DIR']/checkpoints/prior/reinvent.prior

    URL + MD5 are taken from env vars (or defaults below).
    """
    from django.conf import settings

    files_dir = getattr(settings, "GENUI_SETTINGS", {}).get("FILES_DIR")
    if not files_dir:
        raise RuntimeError("GENUI_SETTINGS['FILES_DIR'] is not set")

    url = os.environ.get(
        "REINVENT_PRIOR_URL",
        "https://zenodo.org/records/15641297/files/reinvent.prior?download=1",
    )
    expected_md5 = os.environ.get(
        "REINVENT_PRIOR_MD5",
        "f268eb072f4fca69ca9434768d3cd461",
    )

    rel = os.path.join("checkpoints", "prior", "reinvent.prior")
    target = os.path.join(files_dir, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # already valid?
    if not force and os.path.isfile(target):
        try:
            if _md5_file(target) == expected_md5:
                return target
        except Exception:
            pass  # re-download

    # download to temp and atomically replace
    fd, tmp_path = tempfile.mkstemp(prefix="reinvent_prior_", dir=os.path.dirname(target))
    os.close(fd)

    try:
        with urllib.request.urlopen(url) as r, open(tmp_path, "wb") as out:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        got = _md5_file(tmp_path)
        if got != expected_md5:
            raise RuntimeError(f"Downloaded prior has wrong md5: got {got}, expected {expected_md5}")

        os.replace(tmp_path, target)
        return target

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def setup(*args, **kwargs):
    from genui.utils.init import createGroup
    from . import models

    # Ensure prior exists during setup
    force = bool(kwargs.get("force", False))
    prior_path = ensure_reinvent_prior(force=force)

    # Optional: print/log something (keep it minimal)
    # If you want, you can use Django's command stdout instead, but it's not passed here.
    # print(f"REINVENT prior OK: {prior_path}")

    createGroup(
        "GenUI_Users",
        [
            models.ReinventNet,
            models.ReinventNetTraining,
            models.ReinventNetValidation,
            models.ModelPerformanceReinvent,
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
            models.ReinventAgent,
            models.ReinventAgentTraining,
            models.ReinventAgentValidation,
            models.Reinvent,
            models.ReinventStage,
        ],
        force=force,
    )