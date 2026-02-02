"""
genuisetup

Created by: Martin Sicho
On: 4/28/20, 9:34 AM
"""
import hashlib
import importlib
import os
import tempfile
import urllib.request

from django.core.management import BaseCommand


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

    Downloads it from settings.REINVENT_PRIOR_URL and verifies settings.REINVENT_PRIOR_MD5.
    Uses atomic replace to avoid leaving partial files behind.
    """
    from django.conf import settings

    files_dir = getattr(settings, "GENUI_SETTINGS", {}).get("FILES_DIR")
    if not files_dir:
        raise RuntimeError("GENUI_SETTINGS['FILES_DIR'] is not set")

    url = getattr(settings, "REINVENT_PRIOR_URL", None)
    expected_md5 = getattr(settings, "REINVENT_PRIOR_MD5", None)
    if not url or not expected_md5:
        raise RuntimeError("REINVENT_PRIOR_URL / REINVENT_PRIOR_MD5 are not set")

    rel = os.path.join("checkpoints", "prior", "reinvent.prior")
    target = os.path.join(files_dir, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # Already present + valid?
    if not force and os.path.isfile(target):
        try:
            if _md5_file(target) == expected_md5:
                return target
        except Exception:
            # If hashing fails for any reason, we re-download
            pass

    # Download to temp first, then verify, then atomic replace
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

        os.replace(tmp_path, target)  # atomic on POSIX
        return target
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


class Command(BaseCommand):
    help = "Sets up the genui app and the extensions."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force all updates (also re-download REINVENT prior).",
        )

        parser.add_argument(
            "--strict",
            action="store_true",
            help="Do not ignore some exceptions (also fail if prior download fails).",
        )

    def handle(self, *args, **options):
        # 1) Ensure REINVENT prior exists (only if URL+MD5 are configured)
        try:
            from django.conf import settings

            if getattr(settings, "REINVENT_PRIOR_URL", None) and getattr(settings, "REINVENT_PRIOR_MD5", None):
                prior_path = ensure_reinvent_prior(force=bool(options["force"]))
                self.stdout.write(self.style.SUCCESS(f'REINVENT prior OK: {prior_path}'))
            else:
                self.stderr.write(self.style.WARNING(
                    "REINVENT prior download skipped (REINVENT_PRIOR_URL/REINVENT_PRIOR_MD5 not set)."
                ))
        except Exception as exp:
            if bool(options["strict"]):
                raise
            self.stderr.write(self.style.WARNING(f"REINVENT prior download failed (continuing): {exp}"))

        # 2) Existing setup logic
        apps = []
        try:
            from django.conf import settings
            apps = settings.GENUI_SETTINGS["APPS"]
        except Exception:
            self.stderr.write(self.style.WARNING(
                "Failed to load GENUI_SETTINGS from settings.py. Loading internal modules only..."
            ))
            from genui import apps as genui_apps
            apps = genui_apps.all_()

        for app in apps:
            try:
                setupmodule = importlib.import_module(f"{app}.genuisetup")
                setupmodule.setup(
                    force=bool(options["force"]),
                    strict=bool(options["strict"]),
                )
            except ModuleNotFoundError as exp:
                if not options["strict"]:
                    self.stderr.write(self.style.WARNING(
                        f'Failed to find the genuisetup module for app or extension: "{app}". '
                        f"No setup will be done. Reason: {exp}"
                    ))
                    continue
                raise
            self.stdout.write(self.style.SUCCESS(f'Successful setup for: "{app}"'))