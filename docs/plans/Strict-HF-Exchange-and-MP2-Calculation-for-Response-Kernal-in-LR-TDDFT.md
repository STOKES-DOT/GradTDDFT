# Strict HF Exchange and MP2 Calculation for Response Kernal in LR-TDDFT

**Date**: 2026-04-28  
**Status**: Theory note  
**Scope**: This note records the working theory discussion on strict HF exchange response, local HF/PT2 mixing kernels, and the conceptual difference between ORCA-style CIS(D) corrections and the current TD-GradDFT PT2-like response-kernel approach.

> The filename and title keep the requested wording, including `Kernal`.

---

## 1. Orbital-Rotation Coordinates in LR-TDDFT

In LR-TDDFT/TDA, the response matrix is naturally expressed in occupied-virtual orbital-rotation coordinates.

Let the MO coefficient matrix satisfy

\[
C^\dagger S C = I
\]

An infinitesimal orbital rotation can be parameterized as

\[
C(\kappa)=C e^\kappa
\]

with

\[
\kappa^\dagger=-\kappa
\]

or, in the real closed-shell case,

\[
\kappa^T=-\kappa
\]

The physically relevant response directions are occupied-virtual rotations:

\[
\kappa_{ai}
\]

which mix an occupied orbital \(i\) with a virtual orbital \(a\):

\[
\phi_i(\kappa)
=
\phi_i
+
\sum_a \kappa_{ai}\phi_a
+
O(\kappa^2)
\]

\[
\phi_a(\kappa)
=
\phi_a
-
\sum_i \kappa_{ai}\phi_i
+
O(\kappa^2)
\]

Thus a compound index

\[
I=(ia)
\]

corresponds to one single-excitation direction. The LR-TDDFT matrix elements are orbital Hessian elements in these directions:

\[
A_{ia,jb}
\sim
\frac{\partial^2 E}{\partial\kappa_{ai}\partial\kappa_{bj}}
\]

---

## 2. Strict HF Exchange Kernel in LR-TDDFT

For a global hybrid or TDHF-like exchange term

\[
E_x^{HF,hyb}=a_x E_x^{HF}
\]

with constant \(a_x\), the exact exchange contribution to the LR-TDDFT/TDA matrices is strict and standard:

\[
A^{HF}_{ia,jb}
=
-a_x(ij|ab)
\]

\[
B^{HF}_{ia,jb}
=
-a_x(ib|aj)
\]

These terms come from the second derivative of the orbital-dependent HF exchange energy with respect to occupied-virtual orbital rotations:

\[
K^{HF}_{ia,jb}
=
\frac{\partial^2 E_x^{HF}}
{\partial\kappa_{ai}\partial\kappa_{bj}}
\]

Therefore, for conventional global hybrids such as PBE0, B3LYP, or TDHF, there is no missing local derivative in this HF contribution. The nonlocal exchange kernel above is the strict LR form.

---

## 3. Local or Neural HF Mixing Is a Different Object

If the exchange fraction is local or neural,

\[
a_x = a_\theta(u(\mathbf r))
\]

then replacing it by a density-weighted average,

\[
\alpha_{\mathrm{eff}}
=
\frac{\int \rho(\mathbf r)a_\theta(\mathbf r)\,d\mathbf r}
{\int \rho(\mathbf r)\,d\mathbf r}
\]

and using

\[
A^{HF}_{ia,jb}
=
-\alpha_{\mathrm{eff}}(ij|ab)
\]

\[
B^{HF}_{ia,jb}
=
-\alpha_{\mathrm{eff}}(ib|aj)
\]

is strict only for the effective global-hybrid approximation, not for the true local/neural hybrid functional.

For a local mixing energy

\[
E_x^{mix}
=
\int d\mathbf r\;
a(u(\mathbf r))h_x^{loc}(\mathbf r)
\]

the strict orbital-response kernel should be decomposed as

\[
K^{x,mix}_{IJ}
=
K^{aa}_{IJ}
+
K^{ah}_{IJ}
+
K^{hh}_{IJ}
\]

where \(I=(ia)\) and \(J=(jb)\).

The meta-GGA-like coefficient term is

\[
K^{aa}_{IJ}
=
\int d\mathbf r\;
h_x^{loc}(\mathbf r)
\sum_{\mu\nu}
a_{,\mu\nu}(\mathbf r)
\delta u_\mu^I(\mathbf r)
\delta u_\nu^J(\mathbf r)
\]

The cross term is

\[
K^{ah}_{IJ}
=
\int d\mathbf r\;
\sum_\mu a_{,\mu}(\mathbf r)
\Big[
\delta u_\mu^I(\mathbf r)\delta h_x^J(\mathbf r)
+
\delta u_\mu^J(\mathbf r)\delta h_x^I(\mathbf r)
\Big]
\]

The orbital-dependent HF-density term is

\[
K^{hh}_{IJ}
=
\int d\mathbf r\;
a(\mathbf r)\delta^2 h_x^{IJ}(\mathbf r)
\]

with

\[
\delta h_x^I(\mathbf r)
=
\frac{\partial h_x^{loc}(\mathbf r)}
{\partial\kappa_I}
\]

\[
\delta^2 h_x^{IJ}(\mathbf r)
=
\frac{\partial^2 h_x^{loc}(\mathbf r)}
{\partial\kappa_I\partial\kappa_J}
\]

In the constant coefficient limit,

\[
a(\mathbf r)=a_x
\]

the derivative terms of \(a\) vanish:

\[
a_{,\mu}=0,\qquad a_{,\mu\nu}=0
\]

and the kernel should reduce to the usual nonlocal HF exchange kernel.

---

## 4. Exchange-Hole Gauge for Local HF Energy Density

One possible strict gauge for local HF exchange energy density is the exchange-hole gauge:

\[
h_{x,\sigma}^{loc}(\mathbf r)
=
-\frac12
\int d\mathbf r'
\frac{|\gamma_\sigma(\mathbf r,\mathbf r')|^2}
{|\mathbf r-\mathbf r'|}
\]

where

\[
\gamma_\sigma(\mathbf r,\mathbf r')
=
\sum_{i\in occ,\sigma}
\phi_{i\sigma}(\mathbf r)
\phi_{i\sigma}^*(\mathbf r')
\]

The first orbital response is

\[
\delta h_{x,\sigma}^{I}(\mathbf r)
=
-\int d\mathbf r'
\frac{
\gamma_\sigma(\mathbf r,\mathbf r')
\delta\gamma_\sigma^I(\mathbf r,\mathbf r')
}
{|\mathbf r-\mathbf r'|}
\]

The second response is

\[
\delta^2 h_{x,\sigma}^{IJ}(\mathbf r)
=
-\int d\mathbf r'
\frac{
\delta\gamma_\sigma^I(\mathbf r,\mathbf r')
\delta\gamma_\sigma^J(\mathbf r,\mathbf r')
}
{|\mathbf r-\mathbf r'|}
\]

with the occupied-virtual transition density matrix

\[
\delta\gamma_\sigma^{(ia)}(\mathbf r,\mathbf r')
=
\phi_{a\sigma}(\mathbf r)\phi_{i\sigma}(\mathbf r')
+
\phi_{i\sigma}(\mathbf r)\phi_{a\sigma}(\mathbf r')
\]

This gauge-dependent construction is appropriate for a local-hybrid surrogate, but the most invariant strict HF object remains the nonlocal exchange operator or the OEP/TD-OEP kernel.

---

## 5. PT2/MP2 Local Mixing Kernel

The analogous local PT2 mixing energy is

\[
E_c^{mix}
=
\int d\mathbf r\;
b(u(\mathbf r))p_c^{loc}(\mathbf r)
\]

where

- \(b(u)\) is the local PT2 mixing coefficient
- \(p_c^{loc}\) is a local MP2/PT2 energy-density surrogate

The strict local-surrogate response kernel should be decomposed as

\[
K^{pt2,mix}_{IJ}
=
K^{bb}_{IJ}
+
K^{bp}_{IJ}
+
K^{pp}_{IJ}
\]

with

\[
K^{bb}_{IJ}
=
\int d\mathbf r\;
p_c^{loc}(\mathbf r)
\sum_{\mu\nu}
b_{,\mu\nu}(\mathbf r)
\delta u_\mu^I(\mathbf r)
\delta u_\nu^J(\mathbf r)
\]

\[
K^{bp}_{IJ}
=
\int d\mathbf r\;
\sum_\mu b_{,\mu}(\mathbf r)
\Big[
\delta u_\mu^I(\mathbf r)\delta p_c^J(\mathbf r)
+
\delta u_\mu^J(\mathbf r)\delta p_c^I(\mathbf r)
\Big]
\]

\[
K^{pp}_{IJ}
=
\int d\mathbf r\;
b(\mathbf r)\delta^2 p_c^{IJ}(\mathbf r)
\]

where

\[
\delta p_c^I(\mathbf r)
=
\frac{\partial p_c^{loc}(\mathbf r)}
{\partial\kappa_I}
\]

\[
\delta^2 p_c^{IJ}(\mathbf r)
=
\frac{\partial^2 p_c^{loc}(\mathbf r)}
{\partial\kappa_I\partial\kappa_J}
\]

Thus PT2 should be treated in direct analogy to local HF mixing: the coefficient \(b(u)\) is differentiated like a meta-GGA coefficient, while the local PT2 energy-density surrogate must also be differentiated with respect to orbital rotations.

---

## 6. Local Exact-Pair PT2 Gauge Used in TD-GradDFT

The current TD-GradDFT local PT2 feature can be summarized as

\[
p_c^{loc}(\mathbf r)
=
\sum_{iajb}
\rho_{ia}(\mathbf r)
V_{jb}(\mathbf r)
W_{iajb}
\]

with

\[
\rho_{ia}(\mathbf r)
=
\phi_i(\mathbf r)\phi_a(\mathbf r)
\]

\[
W_{iajb}
=
\frac{2(ia|jb)-(ib|ja)}
{\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}
\]

\[
V_{jb}(\mathbf r)
=
\sum_{pqrs}
\chi_p(\mathbf r)\chi_q(\mathbf r)
(pq|rs)
C_{rj}C_{sb}
\]

The first response is

\[
\delta p_c^I(\mathbf r)
=
\sum_{iajb}
\Big[
\delta\rho_{ia}^I(\mathbf r)V_{jb}(\mathbf r)W_{iajb}
+
\rho_{ia}(\mathbf r)\delta V_{jb}^I(\mathbf r)W_{iajb}
+
\rho_{ia}(\mathbf r)V_{jb}(\mathbf r)\delta W_{iajb}^I
\Big]
\]

The second response contains all product-rule terms:

\[
\delta^2 p_c^{IJ}(\mathbf r)
=
\sum_{iajb}
\delta^{IJ}
\big(
\rho_{ia}V_{jb}W_{iajb}
\big)(\mathbf r)
\]

Expanded explicitly, this includes:

\[
\delta^{IJ}\rho\,V\,W
+
\rho\,\delta^{IJ}V\,W
+
\rho\,V\,\delta^{IJ}W
+
\delta^I\rho\,\delta^J V\,W
+
\delta^J\rho\,\delta^I V\,W
\]

\[
+
\delta^I\rho\,V\,\delta^J W
+
\delta^J\rho\,V\,\delta^I W
+
\rho\,\delta^I V\,\delta^J W
+
\rho\,\delta^J V\,\delta^I W
\]

This is the local-surrogate analogue of differentiating the HF exchange energy density. It is not the same as the full frequency-dependent PT2 response kernel.

---

## 7. Difference from ORCA-Style CIS(D)

ORCA-style double-hybrid excited-state PT2 is conventionally implemented as a post-response CIS(D)-type correction:

\[
\Omega_I^{DH}
=
\Omega_I^{TDDFT/TDA}
+
a_c\Delta_I^{CIS(D)}
\]

or in spin-component-scaled form:

\[
\Omega_I^{DH}
=
\Omega_I^{TDDFT/TDA}
+
c_{OS}\Delta_I^{OS}
+
c_{SS}\Delta_I^{SS}
\]

This correction is applied after the TDDFT/TDA singles problem is solved. It is state-specific and depends on the singles amplitudes and the perturbative coupling to the doubles manifold.

In wavefunction language, the CIS(D) first-order corrected state can be written schematically as

\[
|\Psi_I^{CIS(D)}\rangle
\approx
\sum_{ia}C_{ia}^{I}|\Phi_i^a\rangle
+
\sum_{ijab}
\frac{
U_{ij}^{ab}(I)
}
{
\Omega_I^{(0)}
+
\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b
}
|\Phi_{ij}^{ab}\rangle
\]

where \(U_{ij}^{ab}(I)\) is the coupling between the CIS singles state and the double excitation manifold.

Most practical CIS(D) implementations use this structure mainly to compute an energy correction:

\[
\Delta E_I^{CIS(D)}
\sim
\sum_{ijab}
\frac{|U_{ij}^{ab}(I)|^2}
{\Omega_I^{(0)}+\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b}
+
\Delta E_I^{indirect}
\]

They do not usually rediagonalize a singles+doubles response problem.

---

## 8. TD-GradDFT PT2-Like Kernel Interpretation

The current TD-GradDFT PT2 approach has a different theoretical role:

\[
(A/B)^{model}
=
(A/B)^{sl+HF}
+
K^{PT2-like}
\]

Then the response problem is solved with this modified kernel:

\[
(A/B)^{model}(X_I,Y_I)
=
\Omega_I(X_I,Y_I)
\]

Therefore, the PT2-like term changes not only the excitation energy \(\Omega_I\), but also the singles response amplitudes \(X_I,Y_I\), transition densities, oscillator strengths, and the shape of excited-state potential-energy curves.

This is not an explicit CIS(D) doubles wavefunction correction:

\[
|\Psi_I\rangle
\ne
|S_I\rangle + |D_I^{(1)}\rangle
\]

because TD-GradDFT does not explicitly construct doubles amplitudes \(T_{ij}^{ab,I}\) or a doubles manifold.

The more precise interpretation is:

\[
\boxed{
\text{TD-GradDFT uses a learned local PT2-like kernel to produce a doubles-dressed singles response state.}
}
\]

In contrast:

\[
\boxed{
\text{ORCA/CIS(D) applies a state-specific perturbative doubles correction after solving the singles response problem.}
}
\]

---

## 9. Current Approximation in Code

In the current implementation, the strict local response tensor is computed by applying JAX Hessian to the pointwise local energy with respect to local density variables only:

\[
H_{\mu\nu}(\mathbf r)
=
\frac{\partial^2 e_{xc}^{loc}}
{\partial u_\mu\partial u_\nu}
\]

The local HF and PT2 fields are passed as external point arguments:

\[
h_x^{loc}(\mathbf r),\qquad p_c^{loc}(\mathbf r)
\]

so their own orbital responses are frozen.

For local-projected direct terms, the retained pieces are

\[
H^{HF,current}_{\mu\nu}(\mathbf r)
=
h_x^{loc}(\mathbf r)a_{,\mu\nu}(\mathbf r)
\]

\[
H^{PT2,current}_{\mu\nu}(\mathbf r)
=
p_c^{loc}(\mathbf r)b_{,\mu\nu}(\mathbf r)
\]

Thus the current local-projected response keeps only:

\[
K^{aa}\quad\text{for HF}
\]

\[
K^{bb}\quad\text{for PT2}
\]

and omits:

\[
K^{ah},K^{hh},K^{bp},K^{pp}
\]

The nonlocal HF exchange kernel is still included separately through the effective global exchange fraction. That part is strict for the effective global-hybrid approximation, but it is not the full local-hybrid response for a spatially varying neural \(a_\theta(\mathbf r)\).

---

## 10. Summary

For strict global hybrid HF exchange in LR-TDDFT:

\[
A^{HF}_{ia,jb}=-a_x(ij|ab),\qquad
B^{HF}_{ia,jb}=-a_x(ib|aj)
\]

This part is theoretically strict.

For neural/local HF mixing:

\[
K^{x,mix}
=
K^{aa}+K^{ah}+K^{hh}
\]

For neural/local PT2 mixing:

\[
K^{pt2,mix}
=
K^{bb}+K^{bp}+K^{pp}
\]

ORCA-style CIS(D) and TD-GradDFT PT2-like response kernels belong to different theoretical paradigms:

- ORCA/CIS(D): post-response, state-specific, perturbative doubles energy correction
- TD-GradDFT PT2-like kernel: pre-response, learned local kernel contribution that modifies the singles response eigenproblem

The current TD-GradDFT implementation uses a frozen-field local surrogate:

\[
K^{HF/PT2,current}
\subset
K^{x/pt2,mix}
\]

and the missing terms are exactly those involving the orbital response of \(h_x^{loc}\) and \(p_c^{loc}\).
