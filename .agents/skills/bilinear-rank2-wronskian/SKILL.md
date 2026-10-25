---
name: bilinear-rank2-wronskian
description: Scientific knowledge skill for Paper VIII of the Transverse Wronskian series on Riemann's Xi-function. Contains the proven analytical identities for the (mn)^{-d} <= 2^{-d} hyperbolic diagonal advantage, order-r generalized von Mangoldt scaling and Stirling-Dirichlet convolution laws, the exact bilinear rank-2 factorization w_{mn}^{int}(d,t) = 2(cosh(d*omega_m)cosh(d*omega_n)-1)/d^2 reducing O(N(t)^2) off-diagonal sums to O(N(t)) one-dimensional sums with an Euler-Maclaurin phase transition N(t)^{1-4d} at d = 1/4 (sigma = 3/4), the pure-square completion ||psi_1 + Psi_{>=2}||^2 that cancels the -1 boundary term pair by pair against the critical-line ground state and eliminates the 1 + sqrt(2) root penalty, and the 1 + 2d > 1 exponent jump in L^2 combined with Preissmann's sharp Montgomery-Vaughan Hilbert constant c_MV = (4/3)*sqrt(6) and Gallagher-Sobolev pointwise transfer up to t = 10^8.
---

# Bilinear Rank-2 Factorization of Integrated Wronskian Kernels, Pure-Square Completion, and the $1+2d > 1$ Montgomery–Vaughan Exponent Jump in the Critical Strip (Paper VIII)

## 1. Scope and Mathematical Architecture

This skill encapsulates the analytical theorems, exact arithmetic identities, directed-rounding `mpmath.iv` interval-arithmetic certifications, and GPU continuum verification procedures of **Paper VIII** in the series on the Transverse Wronskian of Riemann's $\Xi$-Function (author: Thijs Missiaen, ORCID: [0009-0007-3678-0810](https://orcid.org/0009-0007-3678-0810)). Paper VIII resolves the off-diagonal cross-term problem in the Riemann–Siegel mirror sum of length $N(t) = \lfloor\sqrt{t/2\pi}\rfloor$ across four analytical pillars:

1. **The $(mn)^{-d} \le 2^{-d}$ Hyperbolic Diagonal Advantage and Order $r$ Generalized von Mangoldt Scaling (Section 2)**: In the parity-reduced transverse gradient $\frac{1}{2d}\partial_d\widetilde{\mathcal{Z}} _ N(d,t)$, every off-diagonal pair $1 \le m < n \le N(t)$ has $mn \ge 2$ and satisfies the universal hyperbolic diagonal advantage $\mathcal{H} _ {mn}^+(d,t)/\mathcal{D} _ {1,1}(d,t) < (1 - \frac{\ln(mn)}{\ln(t/2\pi)})(mn)^{-d} < (mn)^{-d} \le 2^{-d} < 1$ for all $d > 0$. Under the smooth order $r$ Möbius filter $P_r(x) = 1 - (1-x)^r$, the sieved coefficients obey $b_1^{(r)}(X) \equiv 1$ and $b_n^{(r)}(X) = \frac{\Lambda_r(n)}{(\ln X)^r}$ ($2 \le n \le X$), alongside the parity alternation law $\lvert\Lambda_r(n)\rvert \equiv (-1)^{\omega(n)-1}\Lambda_r(n) \ge 0$, the Stirling–Dirichlet convolution asymptotics $\sum _ {n \le x}\lvert\Lambda_r(n)\rvert \sim r x(\ln x)^{r-1}$ via unsigned Stirling numbers of the first kind $c _ {r,m} = \frac{s(r,m)}{\Gamma(r)}$, and the universal radical bound $\lvert\Lambda_r(n)\rvert \le (\ln\mathrm{rad}(n))^r \le (\ln n)^r$.
2. **Exact Bilinear Rank-2 Factorization of Integrated Wronskian Kernels and the Phase Transition at $d = \frac{1}{4}$ (Section 3)**: Under the hyperbolic low-pass filter $\mathcal{F} _ {\mathrm{int}}(x) = \frac{\tanh(x/2)}{x/2} \in (0, 1)$ of the integrated sieved energy clamp $\widetilde{\mathcal{E}} _ {\mathcal{M}}(d,t) = \frac{1}{d^2}\lvert\Xi _ {\mathcal{M}}(t)\rvert^2 + \frac{2}{d^2}\int_0^d u \widetilde{W} _ {\mathcal{M}}(u,t) \mathrm{d}u$, the integrated Gabcke–Olver remainder is damped relative to the leading diagonal by $\cosh^{-2}(\frac{1}{4}dL) < 1$, and the off-diagonal weight factors identically without approximation into a bilinear rank-2 form:

$$w _ {mn}^{\mathrm{int}}(d,t) \equiv \frac{2\left(\cosh\left(d\omega_m(t)\right)\cosh\left(d\omega_n(t)\right) - 1\right)}{d^2} \qquad \left(\omega_n(t) = \frac{1}{2}\ln\frac{t}{2\pi n^2}\right).$$

This reduces the $\binom{N(t)}{2}$ double sums $S _ {1,\mathrm{off}}^{\mathrm{int},(r)}$ and $S _ {2,\mathrm{off}}^{\mathrm{int},(r)}$ to $\mathcal{O}(N(t))$ one-dimensional sums $\Sigma _ {\cosh}^{(r)}(d,t)$ and $\Sigma_0^{(r)}(t)$ (certifying all $7{,}954{,}066$ cross-pairs at $t = 10^8$, $N = 3989$, in $19.7 \mathrm{ms}$) and reveals the Euler–Maclaurin phase transition $S _ {2,\mathrm{off}}^{\mathrm{int},(r)}/(\mathcal{D} _ {1,1}^{\mathrm{int}})^2 \propto N(t)^{1-4d}(\ln N(t))^{2(r-1)}$ at $d = \frac{1}{4}$ ($\sigma = \frac{3}{4}$).
3. **Pure-Square Completion, Pair-by-Pair $-1$ Boundary Cancellation, and Identical Elimination of $1 + \sqrt{2}$ (Section 4)**: When the critical-line ground-state energy $\mathcal{G} _ 0(d,t) = \frac{1}{d^2}\lvert\Xi _ {\mathcal{M},\mathrm{main}}(t)\rvert^2$ is combined with the Wronskian integral $\mathcal{I} _ {\mathrm{int}}(d,t)$, the $-1$ boundary term in $w _ {mn}^{\mathrm{int}}(d,t) = \frac{2}{d^2}(\cosh(d\omega_m)\cosh(d\omega_n) - 1)$ cancels identically pair by pair against $\frac{2}{d^2}\cos\phi_m\cos\phi_n$ at $u = 0$. Consequently, the entire higher-shell $(m,n \ge 2)$ sector forms a non-negative pure square with a PLUS sign ($\mathcal{E} _ {\ge 2}(d,t) \equiv +\frac{1}{d^2}\lVert\Psi _ {\ge 2}(d,t)\rVert^2 \ge 0$), transforming $1 - 2Y - Y^2 = 0$ into the completed square $(1 - Y)^2 = 1 - 2Y + Y^2 \ge 0$, eliminating the root penalty $1 + \sqrt{2} = 2.41421356\dots$ identically, and driving $X _ {\mathrm{krit}}^{\mathrm{sq},(3)}(0.10, 10^4) = 9.992470 < 10 \ll N(10^4) = 39$ ($\theta _ {\mathrm{sq}}^{(3)} = 0.312220 < \frac{1}{3}$) and $X _ {\mathrm{krit}}^{\mathrm{sq},(3)}(0.20, 10^4) = 19.442212 < 39$ ($\theta _ {\mathrm{sq}}^{(3)} = 0.402504 < \frac{1}{2}$).
4. **The $1 + 2d > 1$ Exponent Jump in $L^2$ and Preissmann–Montgomery–Vaughan Hilbert Bounds up to $t = 10^8$ (Section 5)**: Whereas the phase-blind $L^1$ sum $\sum _ {n=2}^{N(t)} \lvert b_n\rvert n^{-(1/2+d)} \asymp N(t)^{1/2-d}$ diverges for $0 < d < \frac{1}{2}$ because $\frac{1}{2} + d < 1$, the quadratic $L^2$ diagonal $\mathcal{E} _ {\mathrm{diag}}^{\mathrm{fwd}}(d,t) = \sum _ {n=2}^{N(t)} b_n^2 n^{-(1+2d)} < \sum _ {n=2}^{\infty} b_n^2 n^{-(1+2d)} < \infty$ possesses the convergent exponent $1 + 2d > 1$ across the entire critical strip $d > 0$, while the reflected diagonal $\mathcal{E} _ {\mathrm{diag}}^{\mathrm{ref}}(d,t) < \frac{\max b_n^2}{2d}N(t)^{-2d} \to 0$ decays polynomially to zero. Combining this with Preissmann's sharp Montgomery–Vaughan Hilbert constant $c _ {\mathrm{MV}} = \frac{4}{3}\sqrt{6} \approx 3.2659863237$ on sub-Riemann–Siegel windows $\Delta T _ {\mathrm{MV}}(d,t) \asymp N(t)^{1-2d} = o(t^{1/2})$ ($\mathcal{W} _ {\mathrm{MV}}^{\mathrm{fwd}}(d,t) = \sum _ {n=2}^{N(t)} b_n^2 n^{-2d}$) and Gallagher's one-dimensional Sobolev inequality proves that $X _ {\mathrm{MV},\infty}^{(r)}(d) = \mathcal{O}(1)$ is finite and $t$-independent, forcing $\theta _ {\mathrm{MV}}^{(r)}(d,t) = \mathcal{O}(1/\ln t) \to 0$ as $t \to \infty$.

## 2. Canonical Identities and Theorems

### 2.1 Hyperbolic Diagonal Advantage and Order $r$ Generalized von Mangoldt Scaling (Theorem 2.1 & Proposition 2.2)

For $t > 4\pi$, $L := \ln(t/2\pi)$, $N(t) = \lfloor\sqrt{t/2\pi}\rfloor$, $\omega_n(t) = \frac{1}{2}\ln\frac{t}{2\pi n^2}$, and $\omega _ {mn}^+(t) = \frac{1}{2}\ln\frac{t}{2\pi mn}$, the hyperbolic main weight $\mathcal{H} _ {mn}^+(d,t) := \frac{1}{d}\omega _ {mn}^+(t)\sinh(2d\omega _ {mn}^+(t))$ and ground-state diagonal $\mathcal{D} _ {1,1}(d,t) := \frac{1}{d}\omega_1(t)\sinh(2d\omega_1(t))$ obey the exact decomposition:

$$\sinh\left(2d\omega _ {mn}^+(t)\right) \equiv (mn)^{-d}\sinh\left(2d\omega_1(t)\right) - \sinh\left(d\ln(mn)\right)\left(\frac{t}{2\pi}\right)^{-d},$$

whence for every off-diagonal pair $1 \le m < n \le N(t)$ ($mn \ge 2$) and every $d > 0$:

$$\frac{\mathcal{H} _ {mn}^+(d,t)}{\mathcal{D} _ {1,1}(d,t)} < \left(1 - \frac{\ln(mn)}{\ln(t/2\pi)}\right)(mn)^{-d} < (mn)^{-d} \le 2^{-d} < 1.$$

For the smooth order $r$ Möbius mollifier $P_r(x) = 1 - (1-x)^r$, the Dirichlet convolution coefficients $b_n^{(r)}(X) = \sum _ {k \mid n,  k \le X} \mu(k)P_r(\frac{\ln(X/k)}{\ln X})$ satisfy $b_1^{(r)}(X) \equiv 1$ and, for $2 \le n \le X$:

$$b_n^{(r)}(X) = \frac{\Lambda_r(n)}{(\ln X)^r}, \qquad \Lambda_r(n) := -\sum _ {k \mid n} \mu(k)(\ln k)^r, \qquad \lvert\Lambda_r(n)\rvert \equiv (-1)^{\omega(n)-1}\Lambda_r(n) \le \left(\ln\mathrm{rad}(n)\right)^r \le (\ln n)^r,$$

with $\Lambda_r(n) = 0$ whenever $\omega(n) > r$ and $\sum _ {n \le x}\lvert\Lambda_r(n)\rvert \sim \left(\sum _ {m=1}^r \frac{s(r,m)}{\Gamma(r)}\right)x(\ln x)^{r-1} = r x (\ln x)^{r-1}$.

### 2.2 Exact Bilinear Rank-2 Factorization and Phase Transition at $d = \frac{1}{4}$ (Theorems 3.1 & 3.2)

Under the integral operator $\frac{2}{d^2}\int_0^d u (\cdot) \mathrm{d}u$, each hyperbolic mode $\mathrm{sinc} _ {\mathrm{hyp}}(u\alpha)$ is multiplied by the hyperbolic low-pass filter $\mathcal{F} _ {\mathrm{int}}(d\alpha) = \frac{\tanh(d\alpha/2)}{d\alpha/2} \in (0, 1)$, and the integrated cross-term weight factors identically as:

$$w _ {mn}^{\mathrm{int}}(d,t) \equiv \frac{2\left(\cosh\left(d\omega_m(t)\right)\cosh\left(d\omega_n(t)\right) - 1\right)}{d^2}.$$

Consequently, the $\mathcal{O}(N(t)^2)$ double sums $S _ {1,\mathrm{off}}^{\mathrm{int},(r)}(d,t)$ and $S _ {2,\mathrm{off}}^{\mathrm{int},(r)}(d,t)$ collapse to the $\mathcal{O}(N(t))$ one-dimensional sums $\Sigma _ {\cosh}^{(r)}(d,t) := \sum _ {n=2}^{N(t)} \frac{\lvert\Lambda_r(n)\rvert}{\sqrt{n}}\cosh(d\omega_n(t))$ and $\Sigma_0^{(r)}(t) := \sum _ {n=2}^{N(t)} \frac{\lvert\Lambda_r(n)\rvert}{\sqrt{n}}$:

$$S _ {1,\mathrm{off}}^{\mathrm{int},(r)}(d,t) \equiv \frac{4}{d^2}\left[\cosh\left(d\omega_1(t)\right)\Sigma _ {\cosh}^{(r)}(d,t) - \Sigma_0^{(r)}(t)\right], \qquad S _ {2,\mathrm{off}}^{\mathrm{int},(r)}(d,t) \equiv \frac{2}{d^2}\left[\left(\Sigma _ {\cosh}^{(r)}(d,t)\right)^2 - \left(\Sigma_0^{(r)}(t)\right)^2\right] - B_2^{\mathrm{int},(r)}(d,t),$$

and obey the Euler–Maclaurin scaling $\frac{S _ {2,\mathrm{off}}^{\mathrm{int},(r)}(d,t)}{(\mathcal{D} _ {1,1}^{\mathrm{int}}(d,t))^2} \sim \frac{256r^2 d^4(1-2d^2)}{(1-4d^2)^2}N(t)^{1-4d}(\ln N(t))^{2(r-1)}$ (with pure $\cosh$-envelope ratio $\frac{S _ {2,\mathrm{tot}}^{\cosh,(r)}(d,t)}{(\mathcal{D} _ {1,1}^{\mathrm{int}}(d,t))^2} \sim \frac{32r^2 d^2}{(1-4d^2)^2}N(t)^{1-4d}(\ln N(t))^{2(r-1)}$), exhibiting a sharp phase transition at $d = \frac{1}{4}$ ($\sigma = \frac{3}{4}$).

### 2.3 Pure-Square Completion and Elimination of $1 + \sqrt{2}$ (Theorem 4.1)

Defining the Euclidean harmonic vectors $\psi_n(d,t) := \frac{b_n}{\sqrt{n}}\left(\sqrt{2}\cosh(d\omega_n(t))\cos\phi_n(t),  \sqrt{2}\sinh(d\omega_n(t))\sin\phi_n(t)\right) \in \mathbb{R}^2$ and $\Psi _ {\ge 2}(d,t) := \sum _ {n=2}^{N(t)} \psi_n(d,t)$, the $+1$ ground-state contribution at $u = 0$ cancels the $-1$ boundary term in $w _ {mn}^{\mathrm{int}}(d,t)$ pair by pair:

$$\widetilde{\mathcal{E}} _ {\mathcal{M},\mathrm{main}}(d,t) \equiv \frac{1}{d^2}\left\lVert\psi_1(d,t) + \Psi _ {\ge 2}(d,t)\right\rVert^2 = \frac{1}{d^2}\lVert\psi_1\rVert^2 + \frac{2}{d^2}\langle\psi_1, \Psi _ {\ge 2}\rangle + \frac{1}{d^2}\left\lVert\Psi _ {\ge 2}(d,t)\right\rVert^2 \ge \frac{1}{d^2}\left(\max\left(0,  \lVert\psi_1\rVert - \lVert\Psi _ {\ge 2}\rVert\right)\right)^2,$$

eliminating the factor $1 + \sqrt{2} = 2.414213562373095\dots$ identically: $\mathcal{Q} _ r^{\mathrm{cosh,sq}}(d,t) = \frac{\Sigma _ {\cosh}^{(r)}(d,t)}{\cosh(d\omega_1(t))} \equiv \frac{1}{1+\sqrt{2}}\mathcal{Q} _ r^{\mathrm{cosh,old}}(d,t)$.

### 2.4 The $1 + 2d > 1$ Exponent Jump in $L^2$ and Preissmann–Montgomery–Vaughan Bounds (Theorems 5.1 & 5.2)

For the higher harmonic shells $Z _ {\ge 2}(d,t) = \sum _ {n=2}^{N(t)} b_n(n^{-(1/2+d)}\mathrm{e}^{-\mathrm{i} t\ln n} + \rho _ \Gamma(d,t)n^{-(1/2-d)}\mathrm{e}^{-\mathrm{i}(2\theta(t)-t\ln n)})$, the forward and reflected diagonal $L^2$ energies satisfy for every $d > 0$:

$$\mathcal{E} _ {\mathrm{diag}}^{\mathrm{fwd}}(d,t) := \sum _ {n=2}^{N(t)} \frac{b_n^2}{n^{1+2d}} < \frac{\mathcal{S} _ {\Lambda_r^2}(d)}{(\ln X)^{2r}} < \infty, \qquad \mathcal{E} _ {\mathrm{diag}}^{\mathrm{ref}}(d,t) := \rho _ \Gamma(d,t)^2\sum _ {n=2}^{N(t)} \frac{b_n^2}{n^{1-2d}} < \frac{\max b_n^2}{2d} N(t)^{-2d} \longrightarrow 0.$$

By the generalized Hilbert inequality with Preissmann's sharp constant $c _ {\mathrm{MV}} = \frac{4}{3}\sqrt{6} \approx 3.2659863237 < \pi\sqrt{2}$, the mean square over the boxcar window $I = [t - \Delta T/2, t + \Delta T/2]$ obeys $\frac{1}{\Delta T}\int_I \lvert D(\tau)\rvert^2 \mathrm{d}\tau \le \mathcal{E} _ {\mathrm{diag}}(d,t) + \frac{2c _ {\mathrm{MV}}}{\Delta T}(\mathcal{W} _ {\mathrm{MV}}(d,t) + \frac{1}{2}\mathcal{E} _ {\mathrm{diag}}(d,t))$ (with $2c _ {\mathrm{MV}}$ replaced by $c _ {\mathrm{MV}}$ under Fejér tent smoothing) and $\mathcal{W} _ {\mathrm{MV}}^{\mathrm{fwd}}(d,t) = \sum _ {n=2}^{N(t)} b_n^2 n^{-2d} \asymp N(t)^{1-2d} = o(t^{1/2})$, forcing $X _ {\mathrm{MV},\infty}^{(r)}(d) = \exp([\mathcal{S} _ {\Lambda_r^2}(d)]^{1/2r}) = \mathcal{O}(1)$ and $\theta _ {\mathrm{MV}}^{(r)}(d,t) \sim \frac{[\mathcal{S} _ {\Lambda_r^2}(d)]^{1/2r}}{\ln(t/2\pi)} \to 0$ as $t \to \infty$.

## 3. Reproducible Verification Workflow

Run the standalone verification and figure-generation suite from the repository root:

```bash
python scripts/verify_paper8_bilinear_rank2_wronskian.py
```
