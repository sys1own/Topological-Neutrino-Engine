"""MAT verification core utilities.

This module mirrors the numerical backbone described in the manuscript:
1) Frobenius -> integral symplectic basis map M
2) Attractor evaluation of kappa = Re[Pi1/Pi0] at t0 = i*39/(2*pi)
3) Heat-kernel-resummed torsion map Phi(n)
4) Minimal Type-I seesaw + PMNS extraction driven by overlap residues

The code is intentionally lightweight and deterministic so reviewers can run it
as a transparent computational checklist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


# -----------------------------
# Global constants (manuscript)
# -----------------------------
DELTA_RS = 0.051293
KAPPA_TARGET = 4.114
T0 = 1j * 39.0 / (2.0 * math.pi)

# Overlap residues quoted in manuscript Appendix A.1
R12 = 0.041
R13 = 0.018
R23 = 0.037


# -----------------------------------------------------------
# 1) Symplectic basis matrix M (Frobenius -> integral cycles)
# -----------------------------------------------------------
def symplectic_basis_matrix() -> np.ndarray:
    """Return the 4x4 mirror-quintic symplectic change-of-basis matrix M.

    It maps
        Pi_integral = M @ Pi_frobenius.
    """
    two_pi_i = 2.0 * math.pi * 1j
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, two_pi_i ** -1, 0.0, 0.0],
            [-25.0 / 12.0, -(11.0 / 2.0) * (two_pi_i ** -1), -5.0 * (two_pi_i ** -2), 0.0],
            [
                (200.0 * float(np.real(np.zeta(3) if hasattr(np, 'zeta') else 1.2020569031595942)))
                * (two_pi_i ** -3),
                (25.0 / 12.0) * (two_pi_i ** -1),
                0.0,
                5.0 * (two_pi_i ** -3),
            ],
        ],
        dtype=np.complex128,
    )


# --------------------------------------
# 2) Attractor evaluation for varpi1/varpi0
# --------------------------------------
def mirror_coordinate(t: complex) -> complex:
    """Mirror coordinate z = exp(2*pi*i*t)."""
    return np.exp(2.0 * math.pi * 1j * t)


def frobenius_varpi0(z: complex, n_terms: int = 100) -> complex:
    """Compute varpi_0(z) = sum_n (5n)!/(n!)^5 z^n up to n_terms."""
    total = 0.0 + 0.0j
    log_z = np.log(z)
    for n in range(n_terms + 1):
        log_coeff = math.lgamma(5 * n + 1) - 5.0 * math.lgamma(n + 1)
        total += np.exp(log_coeff + n * log_z)
    return total


def frobenius_varpi1(z: complex, n_terms: int = 100) -> complex:
    """Compute varpi_1 using Frobenius logarithmic completion.

    varpi_1 = varpi_0*log(z) + 5*sum_{m>=1} a_m (H_{5m}-H_m) z^m,
    where a_m=(5m)!/(m!)^5.
    """

    def harmonic_number(m: int) -> float:
        return sum(1.0 / k for k in range(1, m + 1))

    v0 = frobenius_varpi0(z, n_terms=n_terms)
    regular = 0.0 + 0.0j
    log_z = np.log(z)
    for m in range(1, n_terms + 1):
        log_coeff = math.lgamma(5 * m + 1) - 5.0 * math.lgamma(m + 1)
        term = np.exp(log_coeff + m * log_z)
        regular += 5.0 * (harmonic_number(5 * m) - harmonic_number(m)) * term
    return v0 * np.log(z) + regular


def attractor_kappa(n_terms: int = 100) -> float:
    """Evaluate kappa = Re[Pi1/Pi0] at t0 in the integral basis."""
    z0 = mirror_coordinate(T0)
    varpi0 = frobenius_varpi0(z0, n_terms=n_terms)
    varpi1 = frobenius_varpi1(z0, n_terms=n_terms)

    # Integral basis components needed for kappa.
    # Pi = M @ (varpi0,varpi1,varpi2,varpi3)^T and Pi0/Pi1 depend only on first two rows.
    two_pi_i = 2.0 * math.pi * 1j
    pi0 = varpi0
    pi1 = (two_pi_i ** -1) * varpi1
    return float(np.real(pi1 / pi0))


# ------------------------------
# 3) Heat-kernel torsion map Phi(n)
# ------------------------------
def phi_torsion(n: int, delta_rs: float = DELTA_RS, kappa: float = KAPPA_TARGET) -> float:
    """Heat-kernel-resummed torsion residue map Phi(n) from Eq. (23)-style form."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    inv_n2 = 1.0 / (n * n)
    shape = (1.0 - (n * n) / 49.0) ** 2
    exponent = -delta_rs * (inv_n2 + kappa * math.log(n) * inv_n2 * shape)
    return float(math.exp(exponent))


def mass_from_winding(n: int, delta_rs: float = DELTA_RS, kappa: float = KAPPA_TARGET) -> float:
    """Return m(n)=n^2*Phi(n) in meV units of manuscript convention."""
    return (n * n) * phi_torsion(n=n, delta_rs=delta_rs, kappa=kappa)


# ----------------------------------------
# 4) Minimal PMNS + Type-I seesaw machinery
# ----------------------------------------
@dataclass
class SeesawResult:
    m_light: np.ndarray
    pmns: np.ndarray
    angles_deg: Dict[str, float]


def overlap_matrix(r12: float = R12, r13: float = R13, r23: float = R23) -> np.ndarray:
    """Build a symmetric overlap matrix Omega from residue overlaps."""
    omega = np.array(
        [
            [1.0, r12, r13],
            [r12, 1.0, r23],
            [r13, r23, 1.0],
        ],
        dtype=float,
    )
    return omega


def type_i_seesaw_light_mass(m_dirac: np.ndarray, m_heavy: np.ndarray) -> np.ndarray:
    """Compute m_nu = - m_D M_R^{-1} m_D^T."""
    return -m_dirac @ np.linalg.inv(m_heavy) @ m_dirac.T


def pmns_from_mass_matrix(m_nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize symmetric Majorana mass matrix, returning eigenvalues and mixing U."""
    # For real-symmetric demo matrices, eigh is sufficient and stable.
    evals, evecs = np.linalg.eigh(m_nu)
    order = np.argsort(np.abs(evals))
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs


def mixing_angles_from_pmns(u: np.ndarray) -> Dict[str, float]:
    """Extract (theta12, theta13, theta23) in PDG-like convention (degrees)."""
    s13 = abs(u[0, 2])
    c13 = math.sqrt(max(0.0, 1.0 - s13**2))

    s12 = abs(u[0, 1]) / c13 if c13 > 0 else 0.0
    s23 = abs(u[1, 2]) / c13 if c13 > 0 else 0.0

    theta13 = math.degrees(math.asin(min(1.0, max(0.0, s13))))
    theta12 = math.degrees(math.asin(min(1.0, max(0.0, s12))))
    theta23 = math.degrees(math.asin(min(1.0, max(0.0, s23))))
    return {"theta12": theta12, "theta13": theta13, "theta23": theta23}


def build_seesaw_with_overlaps(
    windings: Iterable[int] = (1, 3, 7),
    heavy_scale: float = 1.0e3,
) -> SeesawResult:
    """Construct a simple seesaw model where overlaps source flavor off-diagonals.

    - Dirac scales follow winding-derived masses m(n)=n^2 Phi(n).
    - Heavy Majorana block is overlap-weighted to encode r12,r13,r23.
    """
    n1, n2, n3 = tuple(windings)
    m_dirac_diag = np.diag([mass_from_winding(n1), mass_from_winding(n2), mass_from_winding(n3)])

    omega = overlap_matrix()
    m_heavy = heavy_scale * omega

    m_light = type_i_seesaw_light_mass(m_dirac_diag, m_heavy)
    eigvals, u_pmns = pmns_from_mass_matrix(m_light)
    angles = mixing_angles_from_pmns(u_pmns)
    return SeesawResult(m_light=eigvals, pmns=u_pmns, angles_deg=angles)


def demo() -> None:
    """Quick textual verification run."""
    np.set_printoptions(precision=6, suppress=True)

    m = symplectic_basis_matrix()
    kappa_est = attractor_kappa(n_terms=100)
    seesaw = build_seesaw_with_overlaps()

    print("Symplectic matrix M:\n", m)
    print(f"\nAttractor t0 = {T0}")
    print(f"kappa (computed, N=100) = {kappa_est:.6f}")
    print(f"kappa (target manuscript) = {KAPPA_TARGET:.6f}")
    print("\nPhi(1), Phi(3), Phi(7):", [phi_torsion(n) for n in (1, 3, 7)])
    print("m(n)=n^2 Phi(n) [meV-like]:", [mass_from_winding(n) for n in (1, 3, 7)])
    print("\nLight neutrino eigenvalues (relative units):", seesaw.m_light)
    print("PMNS-like angles [deg]:", seesaw.angles_deg)


if __name__ == "__main__":
    demo()"""MAT verification core utilities.

This module mirrors the numerical backbone described in the manuscript:
1) Frobenius -> integral symplectic basis map M
2) Attractor evaluation of kappa = Re[Pi1/Pi0] at t0 = i*39/(2*pi)
3) Heat-kernel-resummed torsion map Phi(n)
4) Minimal Type-I seesaw + PMNS extraction driven by overlap residues

The code is intentionally lightweight and deterministic so reviewers can run it
as a transparent computational checklist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


# -----------------------------
# Global constants (manuscript)
# -----------------------------
DELTA_RS = 0.051293
KAPPA_TARGET = 4.114
T0 = 1j * 39.0 / (2.0 * math.pi)

# Overlap residues quoted in manuscript Appendix A.1
R12 = 0.041
R13 = 0.018
R23 = 0.037


# -----------------------------------------------------------
# 1) Symplectic basis matrix M (Frobenius -> integral cycles)
# -----------------------------------------------------------
def symplectic_basis_matrix() -> np.ndarray:
    """Return the 4x4 mirror-quintic symplectic change-of-basis matrix M.

    It maps
        Pi_integral = M @ Pi_frobenius.
    """
    two_pi_i = 2.0 * math.pi * 1j
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, two_pi_i ** -1, 0.0, 0.0],
            [-25.0 / 12.0, -(11.0 / 2.0) * (two_pi_i ** -1), -5.0 * (two_pi_i ** -2), 0.0],
            [
                (200.0 * float(np.real(np.zeta(3) if hasattr(np, 'zeta') else 1.2020569031595942)))
                * (two_pi_i ** -3),
                (25.0 / 12.0) * (two_pi_i ** -1),
                0.0,
                5.0 * (two_pi_i ** -3),
            ],
        ],
        dtype=np.complex128,
    )


# --------------------------------------
# 2) Attractor evaluation for varpi1/varpi0
# --------------------------------------
def mirror_coordinate(t: complex) -> complex:
    """Mirror coordinate z = exp(2*pi*i*t)."""
    return np.exp(2.0 * math.pi * 1j * t)


def frobenius_varpi0(z: complex, n_terms: int = 100) -> complex:
    """Compute varpi_0(z) = sum_n (5n)!/(n!)^5 z^n up to n_terms."""
    total = 0.0 + 0.0j
    log_z = np.log(z)
    for n in range(n_terms + 1):
        log_coeff = math.lgamma(5 * n + 1) - 5.0 * math.lgamma(n + 1)
        total += np.exp(log_coeff + n * log_z)
    return total


def frobenius_varpi1(z: complex, n_terms: int = 100) -> complex:
    """Compute varpi_1 using Frobenius logarithmic completion.

    varpi_1 = varpi_0*log(z) + 5*sum_{m>=1} a_m (H_{5m}-H_m) z^m,
    where a_m=(5m)!/(m!)^5.
    """

    def harmonic_number(m: int) -> float:
        return sum(1.0 / k for k in range(1, m + 1))

    v0 = frobenius_varpi0(z, n_terms=n_terms)
    regular = 0.0 + 0.0j
    log_z = np.log(z)
    for m in range(1, n_terms + 1):
        log_coeff = math.lgamma(5 * m + 1) - 5.0 * math.lgamma(m + 1)
        term = np.exp(log_coeff + m * log_z)
        regular += 5.0 * (harmonic_number(5 * m) - harmonic_number(m)) * term
    return v0 * np.log(z) + regular


def attractor_kappa(n_terms: int = 100) -> float:
    """Evaluate kappa = Re[Pi1/Pi0] at t0 in the integral basis."""
    z0 = mirror_coordinate(T0)
    varpi0 = frobenius_varpi0(z0, n_terms=n_terms)
    varpi1 = frobenius_varpi1(z0, n_terms=n_terms)

    # Integral basis components needed for kappa.
    # Pi = M @ (varpi0,varpi1,varpi2,varpi3)^T and Pi0/Pi1 depend only on first two rows.
    two_pi_i = 2.0 * math.pi * 1j
    pi0 = varpi0
    pi1 = (two_pi_i ** -1) * varpi1
    return float(np.real(pi1 / pi0))


# ------------------------------
# 3) Heat-kernel torsion map Phi(n)
# ------------------------------
def phi_torsion(n: int, delta_rs: float = DELTA_RS, kappa: float = KAPPA_TARGET) -> float:
    """Heat-kernel-resummed torsion residue map Phi(n) from Eq. (23)-style form."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    inv_n2 = 1.0 / (n * n)
    shape = (1.0 - (n * n) / 49.0) ** 2
    exponent = -delta_rs * (inv_n2 + kappa * math.log(n) * inv_n2 * shape)
    return float(math.exp(exponent))


def mass_from_winding(n: int, delta_rs: float = DELTA_RS, kappa: float = KAPPA_TARGET) -> float:
    """Return m(n)=n^2*Phi(n) in meV units of manuscript convention."""
    return (n * n) * phi_torsion(n=n, delta_rs=delta_rs, kappa=kappa)


# ----------------------------------------
# 4) Minimal PMNS + Type-I seesaw machinery
# ----------------------------------------
@dataclass
class SeesawResult:
    m_light: np.ndarray
    pmns: np.ndarray
    angles_deg: Dict[str, float]


def overlap_matrix(r12: float = R12, r13: float = R13, r23: float = R23) -> np.ndarray:
    """Build a symmetric overlap matrix Omega from residue overlaps."""
    omega = np.array(
        [
            [1.0, r12, r13],
            [r12, 1.0, r23],
            [r13, r23, 1.0],
        ],
        dtype=float,
    )
    return omega


def type_i_seesaw_light_mass(m_dirac: np.ndarray, m_heavy: np.ndarray) -> np.ndarray:
    """Compute m_nu = - m_D M_R^{-1} m_D^T."""
    return -m_dirac @ np.linalg.inv(m_heavy) @ m_dirac.T


def pmns_from_mass_matrix(m_nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize symmetric Majorana mass matrix, returning eigenvalues and mixing U."""
    # For real-symmetric demo matrices, eigh is sufficient and stable.
    evals, evecs = np.linalg.eigh(m_nu)
    order = np.argsort(np.abs(evals))
    evals = evals[order]
    evecs = evecs[:, order]
    return evals, evecs


def mixing_angles_from_pmns(u: np.ndarray) -> Dict[str, float]:
    """Extract (theta12, theta13, theta23) in PDG-like convention (degrees)."""
    s13 = abs(u[0, 2])
    c13 = math.sqrt(max(0.0, 1.0 - s13**2))

    s12 = abs(u[0, 1]) / c13 if c13 > 0 else 0.0
    s23 = abs(u[1, 2]) / c13 if c13 > 0 else 0.0

    theta13 = math.degrees(math.asin(min(1.0, max(0.0, s13))))
    theta12 = math.degrees(math.asin(min(1.0, max(0.0, s12))))
    theta23 = math.degrees(math.asin(min(1.0, max(0.0, s23))))
    return {"theta12": theta12, "theta13": theta13, "theta23": theta23}


def build_seesaw_with_overlaps(
    windings: Iterable[int] = (1, 3, 7),
    heavy_scale: float = 1.0e3,
) -> SeesawResult:
    """Construct a simple seesaw model where overlaps source flavor off-diagonals.

    - Dirac scales follow winding-derived masses m(n)=n^2 Phi(n).
    - Heavy Majorana block is overlap-weighted to encode r12,r13,r23.
    """
    n1, n2, n3 = tuple(windings)
    m_dirac_diag = np.diag([mass_from_winding(n1), mass_from_winding(n2), mass_from_winding(n3)])

    omega = overlap_matrix()
    m_heavy = heavy_scale * omega

    m_light = type_i_seesaw_light_mass(m_dirac_diag, m_heavy)
    eigvals, u_pmns = pmns_from_mass_matrix(m_light)
    angles = mixing_angles_from_pmns(u_pmns)
    return SeesawResult(m_light=eigvals, pmns=u_pmns, angles_deg=angles)


def demo() -> None:
    """Quick textual verification run."""
    np.set_printoptions(precision=6, suppress=True)

    m = symplectic_basis_matrix()
    kappa_est = attractor_kappa(n_terms=100)
    seesaw = build_seesaw_with_overlaps()

    print("Symplectic matrix M:\n", m)
    print(f"\nAttractor t0 = {T0}")
    print(f"kappa (computed, N=100) = {kappa_est:.6f}")
    print(f"kappa (target manuscript) = {KAPPA_TARGET:.6f}")
    print("\nPhi(1), Phi(3), Phi(7):", [phi_torsion(n) for n in (1, 3, 7)])
    print("m(n)=n^2 Phi(n) [meV-like]:", [mass_from_winding(n) for n in (1, 3, 7)])
    print("\nLight neutrino eigenvalues (relative units):", seesaw.m_light)
    print("PMNS-like angles [deg]:", seesaw.angles_deg)


if __name__ == "__main__":
    demo()
