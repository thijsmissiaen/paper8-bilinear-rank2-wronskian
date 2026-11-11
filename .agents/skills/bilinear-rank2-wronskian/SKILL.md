---
name: bilinear-rank2-wronskian
description: Mathematical reference and anti-overclaim skill for Paper VIII. Covers canonical Wronskian normalization, the distinction between the squarefree-divisor sieve tilde_Lambda_r(n) and Bombieri-Selberg's Lambda_r^{BS}(n), the n > X sieve-horizon breakdown, exact bilinear rank-2 factorization of integrated cross-term kernels, the sinh(dL/2) reverse-triangle obstruction, Preissmann's true Hilbert constant pi*sqrt(1 + (2/3)*sqrt(6/5)) = 4.1324743621, and why Montgomery-Vaughan L^2 window averages yield Bohr-Landau/Carlson zero-density bounds rather than pointwise Gallagher positivity.
---

# Bilinear Rank-2 Dirichlet Energy Kernels, Sieve-Horizon and Reverse-Triangle Obstructions, and Preissmann–Montgomery–Vaughan $L^2$ Bounds (Paper VIII)

## 1. Canonical Wronskian Normalization

Throughout the series ($s = \frac{1}{2} + d + \mathrm{i}t$, $z = t - \mathrm{i}d$, $d \ge 0{}$):

$$
\Xi(z) = 8\int_0^{\infty} \Phi(u)\cos(2zu)\mathrm{d}u = 4\int_{-\infty}^{\infty} \Phi(u)\cos(2zu)\mathrm{d}u,
$$

$$
W(d,t) = \frac{1}{2}\partial_d\lvert\Xi(t - \mathrm{i}d)\rvert^2 = R I_t - I R_t = 16\int_0^{\infty} y\sinh(2dy)I(y,t)\mathrm{d}y, \qquad \widetilde{W}(d,t) := \frac{W(d,t)}{d}.
$$

## 2. Genuine Surviving Mathematical Identities

### 2.1 Squarefree-Divisor Sieve $\widetilde{\Lambda} _ r(n)$ vs Bombieri–Selberg $\Lambda _ r^{\mathrm{BS}}(n)$ (Theorem 2.3)

- For $P _ r(x) = 1 - (1-x)^r$, $\zeta(s)M _ X^{(r)}(s) = \sum _ {n=1}^{\infty} b _ n^{(r)}(X)n^{-s}$ satisfies $b _ n^{(r)}(X) = \widetilde{\Lambda} _ r(n)/(\ln X)^r$ **only for $2 \le n \le X$**, where $\widetilde{\Lambda} _ r(n) := -\sum _ {k \mid n}\mu(k)(\ln k)^r$ generates $(-1)^{r+1}\zeta(s)(\zeta^{-1})^{(r)}(s)$.
- $\widetilde{\Lambda} _ r(n)$ vanishes for $\omega(n) > r$, satisfies $(-1)^{\omega(n)-1}\widetilde{\Lambda} _ r(n) > 0$ for $1 \le \omega(n) \le r$, obeys $\lvert\widetilde{\Lambda} _ r(n)\rvert \le (\ln\mathrm{rad}(n))^r$, and has unsigned Stirling asymptotics $\sum _ {n \le x}\lvert\widetilde{\Lambda} _ r(n)\rvert \sim r x(\ln x)^{r-1}$.
- By contrast, Bombieri–Selberg's generalized von Mangoldt function $\Lambda _ r^{\mathrm{BS}}(n) := \sum _ {d \mid n}\mu(d)(\ln(n/d))^r$ generates $(-1)^r\zeta^{(r)}(s)/\zeta(s)$, satisfies $\Lambda _ r^{\mathrm{BS}} = \Lambda _ {r-1}^{\mathrm{BS}}\ln + \Lambda _ {r-1}^{\mathrm{BS}} \ast \Lambda$ and $0 \le \Lambda _ r^{\mathrm{BS}}(n) \le (\ln n)^r$, and relates to $\widetilde{\Lambda} _ 2$ via $\widetilde{\Lambda} _ 2 = \Lambda _ 2^{\mathrm{BS}} - 2(\Lambda \ast \Lambda)$ (at $n = 8 = 2^3$, $\widetilde{\Lambda} _ 2(8) = (\ln 2)^2$ whereas $\Lambda _ 2^{\mathrm{BS}}(8) = 5(\ln 2)^2{}$).

### 2.2 Exact Bilinear Rank-2 Factorization (Theorem 3.1)

Under $\frac{2}{d^2}\int _ 0^d u(\cdot)\mathrm{d}u$, the hyperbolic cross-term weight factors identically as

$$
w_{mn}^{\mathrm{int}}(d,t) \equiv \frac{2\bigl(\cosh(d\omega_m(t))\cosh(d\omega_n(t)) - 1\bigr)}{d^2},
$$

which reduces $\mathcal{O}(N^2)$ off-diagonal double sums to $\mathcal{O}(N)$ prefix sums via $\sum _ {1 \le m < n \le N} a _ m a _ n(\cosh(d\omega _ m)\cosh(d\omega _ n)-1) = \frac{1}{2}[(\sum a _ n\cosh(d\omega _ n))^2 - (\sum a _ n)^2 - \sum a _ n^2\sinh^2(d\omega _ n)]$.

### 2.3 Euclidean Harmonic Representation (Theorem 4.1(i))

Combining $w _ {mn}^{\mathrm{int}}(d,t)$ with the $d = 0$ boundary term cancels the $-1$ offset pair by pair and yields $\frac{1}{4}\lvert S _ N(d,t)\rvert^2 = \lVert\psi _ 1(d,t) + \Psi _ {\ge 2}(d,t)\rVert^2$ in $\mathbb{R}^2$.

## 3. Critical Red-Team Obstructions (Never Repeat Falsified Claims)

1. **Sieve-Horizon Breakdown for $n > X$:** For every prime $p > X$, $b _ p^{(r)}(X) = 1$ (undamped $\mathcal{O}(1){}$), and for composite $n > X$ missing divisors $k > X$ destroy $\omega(n) > r$ vanishing ($b _ {30}^{(2)}(10) = -0.79869655 \neq 0$). One **cannot** assume $b _ n^{(r)}(X) = \widetilde{\Lambda} _ r(n)/(\ln X)^r$ on $2 \le n \le N(t)$ when $X < N(t)$.
2. **Reverse-Triangle $\cosh$ vs $\sinh$ Error:** $\lVert\psi _ 1(d,t)\rVert^2 = \sinh^2(dL/2) + \cos^2\phi _ 1(t)$ has **minimum** $\sinh(dL/2)$ (attained whenever $\cos\phi _ 1(t) = 0$), **not** $\cosh(dL/2)$. Furthermore, omitting $\max(0, \cdot)$ before squaring $(\lVert\psi _ 1\rVert - \lVert\Psi _ {\ge 2}\rVert)$ is invalid. Since $\Sigma _ {\cosh}^{(r)}(d,t)/\sinh(dL/2) \asymp N(t)^{1/2-d}/\ln N(t) \to +\infty$ on $0 < d < \frac{1}{2}$, the true phase-blind lower bound collapses to $0$.
3. **True Preissmann Constant and Citation:** Preissmann (1984, *Enseign. Math. (2)* **30**, 95–113) proved $c _ {\mathrm{MV}} = \pi\sqrt{1 + \frac{2}{3}\sqrt{\frac{6}{5}}} = 4.13247436208154\dots$ (never cite $\frac{4}{3}\sqrt{6}$ or *J. Number Theory* **19**).
4. **Failure of Pointwise Gallagher Transfer:** While $1 + 2d > 1$ damps mean-square windows $\Delta T \gg N(T)^{1-2d}$ (giving Bohr–Landau / Carlson zero-density $N(\sigma, T) = o(T){}$), Gallagher's Sobolev term $\sqrt{\int _ I \lvert F\rvert^2 \int _ I \lvert F'\rvert^2}$ is bounded below by $2c _ {\mathrm{MV}}\sqrt{\mathcal{W} _ {\mathrm{MV}}\mathcal{W} _ {\mathrm{MV}}'} \asymp N(T)^{1-2d} \to +\infty$ for **every** $\Delta T > 0$.

## 4. Verification Command

```bash
python scripts/verify_paper8_bilinear_rank2_wronskian.py
```
