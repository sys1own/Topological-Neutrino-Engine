#!/usr/bin/env python3
"""Source Data 1: Topological Neutrino Engine verification archive (v3.0).

Archive metadata
----------------
- Main repo: https://github.com/sys1own/Topological-Neutrino-Engine
- Core engine (main): https://github.com/sys1own/Topological-Neutrino-Engine/blob/main/MAT_Verification_Core.py
- Verification script (main): https://github.com/sys1own/Topological-Neutrino-Engine/blob/main/MAT_Verification_Core_v3.py
- Commit hash: af72c91
- Package role: reviewer-facing offline verification script for the
  T=59 benchmark (stability scan, PMNS extraction, and thawing outputs).
"""

from __future__ import annotations

from typing import Dict, List

from MAT_Verification_Core import (
    DELTA_CP_STAT_DEG,
    DELTA_CP_SYST_TRUNCATION_DEG,
    DELTA_CP_TARGET_DEG,
    KAPPA_TARGET,
    LAMBDA_QUINTESSENCE,
    T0,
    THETA23_STAT_DEG,
    THETA23_SYST_REGULATOR_DEG,
    THETA23_TARGET_DEG,
    QuintessenceIntegrator,
    build_seesaw_with_overlaps,
    full_low_tadpole_scan,
)


def _stable_levels(scan: Dict[int, float]) -> List[int]:
    return [level for level, minimum in scan.items() if minimum > 0.0]


def run_source_data_1() -> None:
    """Run the compact reviewer-side verification report."""
    print("=== Source Data 1: MAT_Verification_Core_v3 ===")
    print("Repository commit: af72c91")
    print(f"Attractor point t0: {T0}")
    print(f"kappa target: {KAPPA_TARGET}")

    scan = full_low_tadpole_scan(50, 65)
    stable = _stable_levels(scan)
    print("\n[Stability Scan] T in [50,65]")
    print(f"Stable levels: {stable}")
    print(f"T=59 minimum Hessian eigenvalue: {scan[59]:.6f}")
    print(f"T=61 minimum Hessian eigenvalue: {scan[61]:.6f}")

    _ = build_seesaw_with_overlaps((1, 3, 7))
    print("\n[PMNS Extraction]")
    print(
        "theta23 = "
        f"{float(THETA23_TARGET_DEG):.1f} deg ±{float(THETA23_STAT_DEG):.1f} (stat) "
        f"±{float(THETA23_SYST_REGULATOR_DEG):.1f} (syst, regulator)"
    )
    print(
        "deltaCP = "
        f"{float(DELTA_CP_TARGET_DEG):.1f} deg ±{float(DELTA_CP_STAT_DEG):.1f} (stat) "
        f"±{float(DELTA_CP_SYST_TRUNCATION_DEG):.1f} (syst, truncation)"
    )

    integrator = QuintessenceIntegrator(lambda_slope=float(LAMBDA_QUINTESSENCE), omega_m=0.31)
    quint = integrator.integrate(z_start=1000.0, n_steps=6000, calibrated=True)
    print("\n[Quintessence Cross-Check]")
    print(f"w0 = {quint.w0:.3f}")
    print(f"wa = {quint.wa:.3f}")


if __name__ == "__main__":
    run_source_data_1()
