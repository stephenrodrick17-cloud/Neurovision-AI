"""
Runs Step 2 (Leakage-Free Baselines) and Step 3 (Quantitative Grad-CAM Evaluation on BraTS Masks)
sequentially in clean isolated Python subprocesses.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("======================================================================", flush=True)
    print("EXECUTING STEP 2: LEAKAGE-FREE BASELINES", flush=True)
    print("======================================================================", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "paper_evidence" / "run_leakage_free_baselines.py")],
        cwd=str(ROOT),
        check=True,
    )

    print("\n======================================================================", flush=True)
    print("EXECUTING STEP 3: GRAD-CAM LOCALIZATION EVALUATION ON BraTS MASKS", flush=True)
    print("======================================================================", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "paper_evidence" / "run_gradcam_brats_evaluation.py")],
        cwd=str(ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
