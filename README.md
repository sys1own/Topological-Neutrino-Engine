# Topological Neutrino Engine (TNE) - Verification Suite

This repository contains the core computational utilities for the paper **"Topological Neutrino Engine: Attractor Residues and the Exclusion of Inverted Hierarchy on the Regular Mirror-Quintic Branch"**. 

The suite provides a high-precision verification of the **Monodromy Attractor Theory (MAT)** pipeline, deriving neutrino masses, mixing angles, and dark energy residues from the $L=59$ branch of the mirror-quintic Calabi-Yau threefold.

## Core Features

### 1. Theorem G.1 Compliance (Spectral-Gap Convergence)
The engine does not use an arbitrary torsion constant. Instead, it implements a **Spectral-Gap Filter** that calculates the Ray-Singer torsion residue ($\delta_{RS}$) as a convergent sum of heat-kernel weights.
* **Verification:** The script demonstrates that the first 10 local Hessian modes capture $>99.8\%$ of the global topological information.

### 2. Monodromy Ordering & IH Exclusion
The suite enforces the **Monodromy Parity Rule**. It numerically demonstrates why the Inverted Hierarchy (IH) is topologically prohibited on the $L=59$ branch.
* **Mechanism:** Any attempt to permute winding numbers into an IH configuration ($n_3 < n_1$) drives the Kähler metric determinant ($\det \Omega$) toward the conifold singularity ($\det \Omega \to 0$), triggering a `ConifoldSingularityError`.

### 3. The L=59 Stability Sieve
The code includes a branch-stability scanner. It verifies that for all flux-stabilized branches where $L < 59$, the Hessian of the effective potential contains at least one non-positive eigenvalue, identifying $L=59$ as the **minimal unique stable branch** for a three-generation model.

---

## Installation & Requirements

The suite requires `mpmath` for 50-decimal precision to accurately track the Frobenius truncation residues.

```bash
pip install numpy mpmath
```

## Usage

To run the full verification suite and reproduce the residues found in **Table 11** and **Table 14** of the manuscript:

```bash
python verify_residues.py
```

### Expected Output
Upon execution, the console will provide a step-by-step audit:
1. **Symplectic Basis Map:** Verification of the $M$-matrix transformation from Frobenius to integral cycles.
2. **Spectral Capture:** A table showing the convergence of $\delta_{RS}$ toward the $99.8\%$ threshold.
3. **Stability Sieve:** Confirmation of the $L=59$ branch stability vs. lower-$L$ instabilities.
4. **PMNS Residues:** Final derivation of mixing angles ($\theta_{12}, \theta_{13}, \theta_{23}$) and mass sums.

---

## 📊 Summary of Theoretical Predictions

| Parameter | Prediction ($L=59$) | Status |
| :--- | :--- | :--- |
| $\sum m_{\nu}$ | $58.6 \pm 2.8$ meV | **Falsifiable Target** |
| Hierarchy | Normal Only | **Topologically Locked** |
| $\theta_{23}$ | $49.2^\circ$ (Upper Octant) | **Residue Output** |
| $(w_0, w_a)$ | $(-0.929, -0.112)$ | **Cosmological Residue** |
