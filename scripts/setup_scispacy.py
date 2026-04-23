"""Install the scispaCy biomedical NER model (en_core_sci_lg).

The model tarball is hosted on allenai's S3 bucket (not PyPI) and its URL
is tied to the scispaCy release version, so we install it from a post-sync
script rather than pinning it in pyproject.toml.

Idempotent: if the model is already importable, does nothing.

Usage:
    uv run python scripts/setup_scispacy.py
    uv run python scripts/setup_scispacy.py --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys

MODEL_NAME = "en_core_sci_lg"
MODEL_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
    "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
)


def _model_available() -> bool:
    """True iff en_core_sci_lg can be imported by spaCy in this env."""
    try:
        import spacy  # noqa: F401
        import en_core_sci_lg  # noqa: F401
        return True
    except ImportError:
        return False


def _install_model() -> int:
    """Install the pinned model tarball into the active interpreter."""
    print(f"[setup_scispacy] Installing {MODEL_NAME} from {MODEL_URL}")
    print(f"[setup_scispacy] (~400 MB download, one-off)")
    uv_cmd = ["uv", "pip", "install", MODEL_URL]
    rc = subprocess.call(uv_cmd)
    if rc == 0:
        return 0
    print(
        f"[setup_scispacy] `uv pip install` failed (exit {rc}); "
        f"trying `python -m pip install` as a fallback.",
        file=sys.stderr,
    )
    pip_cmd = [sys.executable, "-m", "pip", "install", MODEL_URL]
    return subprocess.call(pip_cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the scispaCy en_core_sci_lg model (idempotent)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reinstall the model even if already available.",
    )
    args = parser.parse_args()

    if _model_available() and not args.force:
        print(f"[setup_scispacy] {MODEL_NAME} already installed -- nothing to do.")
        print(f"[setup_scispacy] (Use --force to reinstall.)")
        return 0

    rc = _install_model()
    if rc != 0:
        print(
            f"\n[setup_scispacy] Install failed (exit {rc}). The pipeline will "
            f"still run, but UMLS query expansion will be disabled.",
            file=sys.stderr,
        )
        return rc

    if _model_available():
        print(f"[setup_scispacy] {MODEL_NAME} installed and importable.")
        print(
            f"[setup_scispacy] First NER call will also cache ~1 GB of UMLS KB "
            f"data to ~/.scispacy/ (one-off)."
        )
        return 0

    print(
        f"[setup_scispacy] pip reported success but {MODEL_NAME} is still not "
        f"importable. You may need to restart the Python interpreter.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
