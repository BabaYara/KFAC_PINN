# Discrepancies Between `pinn_kfac.py` and KFAC for PINNs Paper

This document outlines the identified discrepancies between the current implementation in `pinn_kfac.py` and the methods described in the accompanying paper "Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks" (`2025_05_20_e8fffb9338419e358febg.tex`).

## 1. Calculation of Interior Weight Factors (`A_omega`, `B_omega`)

**Paper Reference:** Equation 9

**Description:**
The paper (Eq. 9) defines the Kronecker factors for the PDE interior loss term ($L_\Omega$) as:
*   $\boldsymbol{A}_{\Omega}^{(l)} \approx\left(\frac{1}{N S} \sum_{n, s=1}^{N, S} \boldsymbol{Z}_{n, s}^{(l-1)} \boldsymbol{Z}_{n, s}^{(l-1)^{\top}}\right)$
*   $\boldsymbol{B}_{\Omega}^{(l)} \approx\left(\frac{1}{N} \sum_{n, s=1}^{N, S} \boldsymbol{g}_{n, s}^{(l)} \boldsymbol{g}_{n, s}^{(l)}{ }^{\top}\right)$

Where:
*   $N$ is the number of data points.
*   $S$ is the number of components derived from the Taylor-mode/Forward Laplacian expansion (e.g., $S = d+2$ for a 2nd order PDE in $d$ dimensions, including $u$, its $d$ first partial derivatives, and the Laplacian $\Delta u$).
*   $\boldsymbol{Z}_{n,s}^{(l-1)}$ is the $s$-th component of the augmented input to layer $l$ (e.g., $\boldsymbol{z}^{(l-1)}, \partial_{x_i} \boldsymbol{z}^{(l-1)}, \Delta_{\boldsymbol{x}} \boldsymbol{z}^{(l-1)}$).
*   $\boldsymbol{g}_{n,s}^{(l)}$ is the Jacobian of the PDE operator's output (e.g., $\Delta u_n$) with respect to the $s$-th component of the augmented output of layer $l$ ($\boldsymbol{Z}_{n,s}^{(l)}$). This requires propagating derivatives through the network using techniques like the "Forward Laplacian" (Sec 3.1, Appendix C.2).

**`pinn_kfac.py` Implementation (`_factor_terms` and `step` methods):**
*   `A_om` is calculated as: `(a_i.T @ a_i) / a_i.shape[0]`.
    *   `a_i` represents the standard activations (inputs `h` to the linear layer) from `_forward_cache`. This corresponds to $\boldsymbol{z}^{(l-1)}$ from the standard KFAC formulation (Eq. 4 in the paper), not the augmented $\boldsymbol{Z}_{n,s}^{(l-1)}$.
*   `B_om` is calculated as: `(d_i.T @ d_i) / d_i.shape[0]`.
    *   `d_i` represents the gradients of the *loss function* $L_\Omega$ with respect to the layer's post-activations (outputs of the activation function following the linear layer). This is analogous to $\mathrm{J}_{\boldsymbol{z}^{(l)}} L_\Omega$, not $\boldsymbol{g}_{n,s}^{(l)}$ from Eq. 9.
*   The `forward_laplacian` function is currently used only to compute the final PDE residual for the loss value, not to generate the augmented states $\boldsymbol{Z}_{n,s}^{(l-1)}$ or the Jacobians $\boldsymbol{g}_{n,s}^{(l)}$ needed for the specialized KFAC factors in Eq. 9.

**Discrepancy:**
The implementation computes `A_omega` and `B_omega` using standard activations and standard loss gradients, respectively, which aligns with the KFAC formulation for typical supervised learning (like the boundary term $L_{\partial\Omega}$), rather than the specialized formulation for the PDE term $L_\Omega$ (Eq. 9) that requires augmented states and Jacobians of the PDE operator. The "Forward Laplacian" or Taylor-mode differentiation is not used to compute the components for these factors.

## 2. Absence of KFAC* Update Rule

**Paper Reference:** Section 3.4, "Learning rate and momentum"

**Description:**
The paper describes two variants for updating parameters:
1.  **KFAC:** Uses a manually set momentum parameter ($\mu$) and a line search to find the learning rate ($\alpha_\star$).
2.  **KFAC***: Employs a heuristic from the original KFAC paper [38] to automatically determine both the learning rate $\alpha_\star$ and momentum $\mu_\star$ by minimizing a quadratic model of the loss. This involves solving a 2x2 linear system for $\binom{\alpha_{\star}}{\mu_{\star}}$, requiring Gramian-vector products.

**`pinn_kfac.py` Implementation (`step` method):**
The current implementation includes:
*   A momentum update: `mw = self.momentum * lf.mw + gw` (where `self.momentum` is a fixed hyperparameter).
*   A line search for the learning rate `best_alpha` over a predefined set of scaled learning rates.

**Discrepancy:**
The KFAC* update rule, which provides an automated way to set both learning rate and momentum based on local curvature information, is not implemented. The code only implements the first "KFAC" variant.

## 3. Missing $1/S$ Scaling Factor for $A_\Omega$ (and potentially $B_\Omega$)

**Paper Reference:** Equation 9

**Description:**
Equation 9 in the paper specifies the scaling for $\boldsymbol{A}_{\Omega}^{(l)}$ as $\frac{1}{NS}$ and for $\boldsymbol{B}_{\Omega}^{(l)}$ as $\frac{1}{N}$ (with an additional sum over $S$ components within the $B_\Omega$ term).
*   $\boldsymbol{A}_{\Omega}^{(l)} \approx\left(\frac{1}{N S} \sum_{n, s=1}^{N, S} \boldsymbol{Z}_{n, s}^{(l-1)} \boldsymbol{Z}_{n, s}^{(l-1)^{\top}}\right)$
*   $\boldsymbol{B}_{\Omega}^{(l)} \approx\left(\frac{1}{N} \sum_{n, s=1}^{N, S} \boldsymbol{g}_{n, s}^{(l)} \boldsymbol{g}_{n, s}^{(l)}{ }^{\top}\right)$

The text following Eq. 9 states: "applies the expectation approximation from §2.2 over the batch and shared axes". If the expectation is taken over $N \times S$ total samples for both factors, then $B_\Omega$ might also be expected to have a $1/(NS)$ scaling. However, strictly following the formula, $A_\Omega$ has $1/(NS)$ and $B_\Omega$ has $1/N \sum_s (\dots)$.

**`pinn_kfac.py` Implementation (`step` method):**
*   `A_om` update is scaled by `1 / a_i.shape[0]`, where `a_i.shape[0]` is $N$ (number of interior points).
*   `B_om` update is scaled by `1 / d_i.shape[0]`, where `d_i.shape[0]` is $N$.

**Discrepancy:**
The current implementation scales both `A_om` and `B_om` by $1/N$.
The paper's Eq. 9 requires $A_\Omega$ to be scaled by $1/(NS)$.
This discrepancy is linked to Discrepancy #1: since the $S$ different components (from Taylor-mode/Forward Laplacian) are not being computed for the factors, the $1/S$ scaling is consequently also absent. If Eq. 9 were to be fully implemented, $A_\Omega$ would require the $1/(NS)$ scaling. The scaling for $B_\Omega$ would be $1/N$ but would include a sum over $S$ terms, or if interpreted as a full expectation over $N \times S$ samples, it would also be $1/(NS)$.

---

This summary should help in guiding the efforts to align the implementation with the paper.
