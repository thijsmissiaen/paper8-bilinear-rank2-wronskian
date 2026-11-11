# Bilinear Rank-2 Factorization of Dirichlet Energy Kernels, Reverse-Triangle and Sieve-Horizon Obstructions, Preissmann–Montgomery–Vaughan $L^2$ Bounds, and Failure of Pointwise Gallagher Transfer

**DOI (Zenodo / CERN):** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23645887.svg)](https://doi.org/10.5281/zenodo.23645887) [`10.5281/zenodo.23645887`](https://doi.org/10.5281/zenodo.23645887)  
**Author:** Thijs Missiaen ([ORCID: 0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810))  
**Manuscript:** [`paper/paper8_bilinear_rank2_wronskian.pdf`](paper/paper8_bilinear_rank2_wronskian.pdf) (11 pages, AMS-LaTeX)  
**Verification Suite:** [`scripts/verify_paper8_bilinear_rank2_wronskian.py`](scripts/verify_paper8_bilinear_rank2_wronskian.py) (`mpmath.mp.dps = 50`, `mpmath.iv.dps = 35`)  
**MSC 2020:** Primary 11M06, 11N36; Secondary 11M26, 42A05

## Overview

In the symmetric Riemann–Siegel model of Riemann's $\Xi$-function at $z = t - \mathrm{i}d$ ($d = \sigma - \frac{1}{2} > 0{}$), governed by the canonical normalization

$$
\Xi(z) = 8\int_0^{\infty} \Phi(u)\cos(2zu)\mathrm{d}u = 4\int_{-\infty}^{\infty} \Phi(u)\cos(2zu)\mathrm{d}u,
$$

$$
W(d,t) = \frac{1}{2}\partial_d\lvert\Xi(t - \mathrm{i}d)\rvert^2 = 16\int_0^{\infty} y\sinh(2dy)I(y,t)\mathrm{d}y, \qquad \widetilde{W}(d,t) := \frac{W(d,t)}{d},
$$

this paper establishes the exact rank-2 bilinear identities for integrated Dirichlet cross-term kernels and proves the arithmetic and analytic obstructions that prevent phase-blind reverse-triangle bounds and Sobolev-transferred $L^2$ window averages from certifying pointwise zero-freeness in $0 < d < \frac{1}{2}$.

## Summary of Main Results

### 1. Squarefree-Divisor Sieve $\widetilde{\Lambda} _ r(n)$ versus Bombieri–Selberg $\Lambda _ r^{\mathrm{BS}}(n)$, and Breakdown for $n > X$ (Theorem 2.3)

Under the polynomial mollifier $M _ X^{(r)}(s) = \sum _ {k \le X}\mu(k)P _ r(\frac{\ln(X/k)}{\ln X})k^{-s}$ with $P _ r(x) = 1 - (1-x)^r$, the Dirichlet coefficients of $\zeta(s)M _ X^{(r)}(s)$ satisfy $b _ n^{(r)}(X) = \widetilde{\Lambda} _ r(n)/(\ln X)^r$ on $2 \le n \le X$, where $\widetilde{\Lambda} _ r(n) := -\sum _ {k \mid n}\mu(k)(\ln k)^r$ generates $(-1)^{r+1}\zeta(s)(\zeta^{-1})^{(r)}(s)$. We prove parity alternation, vanishing for $\omega(n) > r$, the sharp radical bound $\lvert\widetilde{\Lambda} _ r(n)\rvert \le (\ln\mathrm{rad}(n))^r$, and unsigned Stirling asymptotics, distinguish $\widetilde{\Lambda} _ r(n)$ from Bombieri–Selberg's $\Lambda _ r^{\mathrm{BS}}(n) := \sum _ {d \mid n}\mu(d)(\ln(n/d))^r \ge 0$ ($\widetilde{\Lambda} _ 2 = \Lambda _ 2^{\mathrm{BS}} - 2(\Lambda \ast \Lambda){}$), and prove that $b _ n^{(r)}(X) = \widetilde{\Lambda} _ r(n)/(\ln X)^r$ fails at $\mathcal{O}(1)$ scale for $n > X$ ($b _ p^{(r)}(X) = 1$ for primes $p > X$; $b _ {30}^{(2)}(10) = -0.79869655 \neq 0$ despite $\widetilde{\Lambda} _ 2(30) = 0{}$).

### 2. Exact Bilinear Rank-2 Factorization and $\mathcal{O}(N)$ Reduction (Theorem 3.1)

Integrating the transverse hyperbolic modes over $u \in [0, d]$ factors the cross-term weight identically as

$$
w_{mn}^{\mathrm{int}}(d,t) \equiv \frac{2\bigl(\cosh(d\omega_m(t))\cosh(d\omega_n(t)) - 1\bigr)}{d^2},
$$

reducing $\mathcal{O}(N^2)$ off-diagonal double sums to $\mathcal{O}(N)$ one-dimensional sums without approximation.

### 3. Euclidean Representation and the $\sinh(dL/2)$ Reverse-Triangle Obstruction (Theorem 4.1)

Adding the critical-line boundary term cancels the $-1$ offset pair by pair and yields $\frac{1}{4}\lvert S _ N(d,t)\rvert^2 = \lVert\psi _ 1(d,t) + \Psi _ {\ge 2}(d,t)\rVert^2$ in $\mathbb{R}^2$. However, $\lVert\psi _ 1(d,t)\rVert^2 = \sinh^2(dL/2) + \cos^2\phi _ 1(t)$ attains its minimum $\lVert\psi _ 1(d,t)\rVert _ {\min} = \sinh(dL/2)$ (not $\cosh(dL/2){}$) whenever $\cos\phi _ 1(t) = 0$. Because $\Sigma _ {\cosh}^{(r)}(d,t)/\sinh(dL/2) \asymp N(t)^{1/2-d}/\ln N(t) \to +\infty$ for $0 < d < \frac{1}{2}$ at $X = N(t)$, the phase-blind lower bound $\max(0, \sinh(dL/2) - \Sigma _ {\cosh}^{(r)}(d,t))$ vanishes identically for all large $t$.

### 4. Preissmann–Montgomery–Vaughan $L^2$ Window Bounds versus Pointwise Gallagher Divergence (Theorem 5.1)

Using Preissmann's true sharp Hilbert constant (Preissmann 1984, *Enseign. Math.* **30**, 95–113)

$$
c_{\mathrm{MV}} = \pi\sqrt{1 + \frac{2}{3}\sqrt{\frac{6}{5}}} = 4.13247436208154\dots,
$$

the exponent $1 + 2d > 1$ yields mean-square suppression on vertical windows $\Delta T \gg N(T)^{1-2d}$ (recovering the Bohr–Landau 1914 and Carlson 1920 zero-density mechanism $N(\sigma, T) = o(T){}$), whereas in Gallagher's Sobolev inequality the cross-term $\sqrt{\int _ I \lvert F\rvert^2 \int _ I \lvert F'\rvert^2}$ diverges at least as $\mathcal{O}(N(T)^{1-2d})$ for every $\Delta T > 0$, reflecting Kronecker–Voronin phase alignment.

## Repository Structure

- [`paper/paper8_bilinear_rank2_wronskian.tex`](paper/paper8_bilinear_rank2_wronskian.tex) — AMS-LaTeX source manuscript.
- [`paper/paper8_bilinear_rank2_wronskian.pdf`](paper/paper8_bilinear_rank2_wronskian.pdf) — Compiled 11-page PDF manuscript.
- [`scripts/verify_paper8_bilinear_rank2_wronskian.py`](scripts/verify_paper8_bilinear_rank2_wronskian.py) — Standalone 50-decimal `mpmath` and 35-digit `mpmath.iv` verification and figure-generation suite.
- [`figures/`](figures/) — Generated vector PDF and 300 DPI PNG figures (`fig1_bilinear_rank2_and_pure_square`, `fig2_exponent_jump_montgomery_vaughan`).
- [`.agents/skills/bilinear-rank2-wronskian/SKILL.md`](.agents/skills/bilinear-rank2-wronskian/SKILL.md) — AI agent mathematical reference skill.

## Reproducible Verification

Run the standalone verification script from the repository root:

```bash
python scripts/verify_paper8_bilinear_rank2_wronskian.py
```

The script certifies the canonical Wigner integral residual ($1.54 \times 10^{-33}{}$), Bombieri's recursion and $\widetilde{\Lambda} _ 2 = \Lambda _ 2^{\mathrm{BS}} - 2(\Lambda \ast \Lambda)$ ($1.18 \times 10^{-49}{}$), the $n > X$ sieve breakdown, the rank-2 $\mathcal{O}(N)$ reduction ($1.13 \times 10^{-51}{}$), the $\sinh(dL/2)$ reverse-triangle collapse, Preissmann's interval enclosure $[4.1324743620815, 4.1324743620815]$, and Gallagher Sobolev divergence, exiting with code `0`.

## License

Released under the MIT License.
