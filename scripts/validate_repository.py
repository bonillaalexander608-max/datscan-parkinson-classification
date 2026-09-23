"""Static checks that do not require competition data or trained weights."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def validate_notebooks() -> None:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if len(notebooks) != 8:
        raise AssertionError(f"Expected 8 notebooks, found {len(notebooks)}")

    forbidden = (
        r"C:\Users",
        "nn=NearestNeighbors",
    )

    for path in notebooks:
        raw = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in raw:
                raise AssertionError(f"Forbidden token {token!r} in {path}")

        notebook = json.loads(raw)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        compile(code, str(path), "exec")

        if "from src.config import DATA_ROOT" not in code:
            raise AssertionError(f"Portable DATA_ROOT import missing in {path}")


def validate_metadata() -> None:
    metadata_paths = (
        ROOT / "src/submission/cnn3d_asymmetry_metadata.json",
        ROOT / "src/submission/efficientnet_multiview_model_metadata.json",
        ROOT / "src/submission/ensemble_metadata.json",
    )
    for path in metadata_paths:
        json.loads(path.read_text(encoding="utf-8"))

    ensemble = json.loads(metadata_paths[-1].read_text(encoding="utf-8"))
    top_level = ensemble["effective_top_level_weights"]
    if abs(sum(top_level.values()) - 1.0) > 1e-10:
        raise AssertionError("Effective top-level ensemble weights do not sum to 1")


def validate_requirements() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    if "scikit-image" not in requirements:
        raise AssertionError("scikit-image is required by inference_modelo_2.py")


if __name__ == "__main__":
    validate_notebooks()
    validate_metadata()
    validate_requirements()
    print("Repository static validation passed.")
