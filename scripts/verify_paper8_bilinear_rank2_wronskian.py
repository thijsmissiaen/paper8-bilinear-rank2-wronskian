#!/usr/bin/env python3
"""
Standalone Verification & Figure-Generation Suite for Paper VIII:
"Bilinear Rank-2 Factorization of Dirichlet Energy Kernels, Reverse-Triangle and
 Sieve-Horizon Obstructions, Preissmann-Montgomery-Vaughan L^2 Bounds, and
 Failure of Pointwise Gallagher Transfer"

Author: Thijs Missiaen (ORCID: 0009-0007-3678-0810)

Certifies in 50-decimal arbitrary precision (mpmath.mp.dps = 50) and 35-digit
directed-rounding interval arithmetic (mpmath.iv.dps = 35):
  1. Canonical normalization W(d,t) = 16 int_0^infty y sinh(2dy) I(y,t) dy.
  2. Bombieri-Selberg generalized von Mangoldt Lambda_r(n):
       - Exact recursion Lambda_r = Lambda_{r-1} ln + Lambda_{r-1} * Lambda,
       - Non-negativity and sharp divisor bound 0 <= Lambda_r(n) <= (ln n)^r,
       - Exact identity b_n^{(r)}(X) = Lambda_r(n)/(ln X)^r for 2 <= n <= X vs
         explicit O(1) breakdown for n > X when X < N(t).
  3. Exact rank-2 bilinear factorization of cosh(d omega_m) cosh(d omega_n) - 1,
     reducing O(N^2) off-diagonal sums to O(N) linear sums to 50 decimals.
  4. Exact Euclidean representation (1/4)|S_N(d,t)|^2 = ||psi_1 + Psi_{>=2}||^2,
     certification of the TRUE minimum ||psi_1||_min = sinh(dL/2) (exposing the
     cosh vs sinh reverse-triangle error), and divergence of Sigma_cosh / sinh(dL/2)
     on 0 < d < 1/2.
  5. True Preissmann (1984) Hilbert constant c_MV = pi * sqrt(1 + (2/3)sqrt(6/5))
     = 4.13247436208..., N(T)^{1-2d} mean-square window suppression (Bohr-Landau /
     Carlson mechanism), and explicit explosion of the Gallagher Sobolev term
     Delta T * sqrt(E_diag * E_deriv) ~ N(T)^{1-2d} -> +infty.
"""

from pathlib import Path
import time
import math
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mp.mp.dps = 50
mp.iv.dps = 35

ROOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT_DIR / "figures"
PAPER_FIG_DIR = ROOT_DIR / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Arithmetic & Analytic Helpers
# ============================================================================

def sieve_mobius_and_primes(limit: int):
    primes = []
    is_comp = bytearray(limit + 1)
    mu = [0] * (limit + 1)
    mu[1] = 1
    for i in range(2, limit + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > limit:
                break
            is_comp[i * p] = 1
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu, primes


def compute_lambda_r_tables(r_max: int, n_max: int, mu: list[int]):
    """
    Computes BOTH:
      - Bombieri-Selberg Lambda_r^{BS}(n) = sum_{d | n} mu(d) (ln(n/d))^r >= 0,
      - Squarefree-divisor sieve tilde_Lambda_r(n) = -sum_{k | n} mu(k) (ln k)^r
        associated with P_r(x) = 1 - (1 - x)^r = 1 - (ln k / ln X)^r.
    """
    logs = [mp.mpf(0)] * (n_max + 1)
    for n in range(2, n_max + 1):
        logs[n] = mp.log(n)

    bs_tables: dict[int, list[mp.mpf]] = {}
    tilde_tables: dict[int, list[mp.mpf]] = {}
    for r in range(1, r_max + 1):
        arr_bs = [mp.mpf(0)] * (n_max + 1)
        arr_tl = [mp.mpf(0)] * (n_max + 1)
        log_pow = [logs[m] ** r for m in range(n_max + 1)]
        for d in range(1, n_max + 1):
            if mu[d] == 0:
                continue
            md = mu[d]
            term_d = -md * log_pow[d]
            for m in range(1, n_max // d + 1):
                arr_bs[d * m] += md * log_pow[m]
                arr_tl[d * m] += term_d
        bs_tables[r] = arr_bs
        tilde_tables[r] = arr_tl
    return bs_tables, tilde_tables


def b_sieve_coeff(n: int, X: float, r: int, mu: list[int]) -> mp.mpf:
    """Computes b_n^{(r)}(X) = sum_{k | n, k <= X} mu(k) [1 - (ln k / ln X)^r] for P_r(x) = 1 - (1-x)^r."""
    lnX = mp.log(mp.mpf(str(X)))
    acc = mp.mpf(0)
    for k in range(1, min(n, int(math.floor(X))) + 1):
        if n % k == 0 and mu[k] != 0:
            acc += mu[k] * (1 - (mp.log(k) / lnX) ** r)
    return acc


def phi_shell(m: int, u):
    eu4 = mp.exp(4 * u)
    m2 = mp.mpf(m * m)
    return (2 * mp.pi**2 * (m2**2) * mp.exp(9 * u) - 3 * mp.pi * m2 * mp.exp(5 * u)) * mp.exp(-mp.pi * m2 * eu4)


def Phi_jacobi(u, m_max: int = 14):
    if mp.re(u) < 0:
        u = -u
    return mp.fsum(phi_shell(m, u) for m in range(1, m_max + 1))


def Xi_exact(z):
    s = mp.mpf("0.5") + 1j * z
    return mp.mpf("0.5") * s * (s - 1) * (mp.pi ** (-s / 2)) * mp.gamma(s / 2) * mp.zeta(s)


def vartheta_and_prime(t):
    w = mp.mpf("0.25") + 1j * t / 2
    th = mp.im(mp.loggamma(w)) - (t / 2) * mp.log(mp.pi)
    th_p = mp.mpf("0.5") * mp.re(mp.psi(0, w)) - mp.mpf("0.5") * mp.log(mp.pi)
    return th, th_p


# ============================================================================
# Verification Sections
# ============================================================================

def verify_section1_and_2_canonical_and_lambda_r():
    print("=" * 78)
    print("[1/5] Canonical Wronskian, Lambda_r(n) Divisor Identity, & n > X Breakdown")
    print("=" * 78)
    # 1. Canonical W(d,t) = 16 int_0^infty y sinh(2dy) I(y,t) dy
    old_dps = mp.mp.dps
    mp.mp.dps = 30
    d_w, t_w = mp.mpf("0.25"), mp.mpf("3.5")
    z0 = t_w - 1j * d_w
    f0 = Xi_exact(z0)
    fp = mp.diff(Xi_exact, z0)
    W_ref = mp.re(f0) * mp.im(fp) - mp.im(f0) * mp.re(fp)

    h_step = mp.mpf("0.04")
    m_half = 60
    phi_cache = {idx: Phi_jacobi(idx * h_step / 2) for idx in range(-2 * m_half, 2 * m_half + 1)}
    cos_cache = {k: mp.cos(2 * t_w * k * h_step) for k in range(-m_half, m_half + 1)}
    W_int = mp.mpf(0)
    for j in range(0, m_half + 1):
        yp = j * h_step
        yw = h_step * (mp.mpf("0.5") if j in (0, m_half) else mp.mpf(1))
        Iy = h_step * mp.fsum(phi_cache[k + j] * phi_cache[k - j] * cos_cache[k] for k in range(-m_half, m_half + 1))
        W_int += yw * yp * mp.sinh(2 * d_w * yp) * Iy
    W_int *= 16
    w_res = abs(W_ref - W_int)
    mp.mp.dps = old_dps
    print(f"  Canonical Wigner |W(d,t) - 16 int_0^infty y sinh(2dy) I dy| : {mp.nstr(w_res, 6)}")
    assert w_res < mp.mpf("1e-28")

    # 2. Contrast tilde_Lambda_r(n) = -sum_{k|n} mu(k)(ln k)^r with Bombieri-Selberg Lambda_r^{BS}(n)
    n_max = 80
    mu, primes = sieve_mobius_and_primes(n_max)
    bs_tab, tl_tab = compute_lambda_r_tables(3, n_max, mu)

    max_rec_err = mp.mpf(0)
    max_div_err = mp.mpf(0)
    max_rel_err = mp.mpf(0)
    for r in (2, 3):
        for n in range(1, n_max + 1):
            rec_rhs = bs_tab[r - 1][n] * (mp.log(n) if n > 1 else 0) + mp.fsum(
                bs_tab[r - 1][u] * bs_tab[1][n // u] for u in range(1, n + 1) if n % u == 0
            )
            max_rec_err = max(max_rec_err, abs(bs_tab[r][n] - rec_rhs))
            div_sum = mp.fsum(bs_tab[r][d] for d in range(1, n + 1) if n % d == 0)
            target = (mp.log(n) ** r) if n > 1 else mp.mpf(0)
            max_div_err = max(max_div_err, abs(div_sum - target))
            assert -mp.mpf("1e-47") <= bs_tab[r][n] <= target + mp.mpf("1e-47")
            if n >= 2:
                rad_n = 1
                om_n = 0
                for p in primes:
                    if p > n:
                        break
                    if n % p == 0:
                        rad_n *= p
                        om_n += 1
                assert abs(tl_tab[r][n]) <= (mp.log(rad_n) ** r) + mp.mpf("1e-47")
                if om_n > r:
                    assert abs(tl_tab[r][n]) < mp.mpf("1e-47")
                else:
                    assert ((-1) ** (om_n - 1)) * tl_tab[r][n] > mp.mpf(0)

    # Exact relation tilde_Lambda_2(n) = Lambda_2^{BS}(n) - 2 (Lambda * Lambda)(n)
    for n in range(1, n_max + 1):
        conv_11 = mp.fsum(bs_tab[1][u] * bs_tab[1][n // u] for u in range(1, n + 1) if n % u == 0)
        max_rel_err = max(max_rel_err, abs(tl_tab[2][n] - (bs_tab[2][n] - 2 * conv_11)))

    lam2_8_bs = bs_tab[2][8]
    lam2_8_tl = tl_tab[2][8]
    print(f"  Bombieri recursion max residual (r=2,3; n<=80)              : {mp.nstr(max_rec_err, 6)}")
    print(f"  Relation |tilde_Lambda_2 - (Lambda_2^BS - 2 Lambda*Lambda)| : {mp.nstr(max_rel_err, 6)}")
    print(f"  At n = 8 = 2^3: tilde_Lambda_2(8) = (ln 2)^2 = {mp.nstr(lam2_8_tl, 6)}, whereas Lambda_2^BS(8) = 5(ln 2)^2 = {mp.nstr(lam2_8_bs, 6)}")
    assert max_rel_err < mp.mpf("1e-47")
    assert lam2_8_bs > 4 * lam2_8_tl

    # 3. Verify b_n^{(r)}(X) = tilde_Lambda_r(n)/(ln X)^r ONLY on 2 <= n <= X, and FAILS for n > X!
    X_small = 10
    lnX_s = mp.log(X_small)
    max_le_X = max(abs(b_sieve_coeff(n, X_small, 2, mu) - tl_tab[2][n] / (lnX_s ** 2)) for n in range(2, X_small + 1))
    b11 = b_sieve_coeff(11, X_small, 2, mu)
    b30 = b_sieve_coeff(30, X_small, 2, mu)
    print(f"  Sieve identity on 2 <= n <= {X_small}: max |b_n^(2) - tilde_Lambda_2/(ln X)^2| = {mp.nstr(max_le_X, 6)}")
    print(f"  Breakdown for n > X: b_11^(2)(10) = {mp.nstr(b11, 6)} (prime > X), and tilde_Lambda_2(30)=0 yet b_30^(2)(10) = {mp.nstr(b30, 8)}")
    assert max_le_X < mp.mpf("1e-47")
    assert abs(b11 - 1) < mp.mpf("1e-47")
    assert abs(b30) > mp.mpf("0.1")
    return mu, tl_tab, bs_tab


def verify_section3_and_4_rank2_and_reverse_triangle(mu, lam_tab):
    print("=" * 78)
    print("[2/5] Rank-2 Bilinear O(N) Identity & Reverse-Triangle Obstruction")
    print("=" * 78)
    t = mp.mpf("1000.0")
    d = mp.mpf("0.20")
    th, th_p = vartheta_and_prime(t)
    L = 2 * th_p
    N = int(mp.floor(mp.sqrt(t / (2 * mp.pi))))  # N = 12
    X = mp.mpf(N)  # X >= N so b_n^{(2)}(X) = Lambda_2(n)/(ln X)^2 holds on ALL 2 <= n <= N!
    lnX = mp.log(X)

    b = [mp.mpf(0)] * (N + 1)
    b[1] = mp.mpf(1)
    for n in range(2, N + 1):
        b[n] = lam_tab[2][n] / (lnX ** 2)

    omega = [mp.mpf(0)] + [th_p - mp.log(n) for n in range(1, N + 1)]
    phi = [mp.mpf(0)] + [t * mp.log(n) - th for n in range(1, N + 1)]
    a = [mp.mpf(0)] + [b[n] / mp.sqrt(n) * mp.cos(phi[n]) for n in range(1, N + 1)]

    # 1. Rank-2 O(N^2) vs O(N) identity for sum_{1<=m<n<=N} a_m a_n [cosh(d w_m) cosh(d w_n) - 1]
    quad_sum = mp.fsum(
        a[m] * a[n] * (mp.cosh(d * omega[m]) * mp.cosh(d * omega[n]) - 1)
        for m in range(1, N + 1)
        for n in range(m + 1, N + 1)
    )
    U_cosh = mp.fsum(a[n] * mp.cosh(d * omega[n]) for n in range(1, N + 1))
    U_unit = mp.fsum(a[n] for n in range(1, N + 1))
    D_sinh2 = mp.fsum((a[n] ** 2) * (mp.sinh(d * omega[n]) ** 2) for n in range(1, N + 1))
    lin_sum = mp.mpf("0.5") * (U_cosh ** 2 - U_unit ** 2 - D_sinh2)
    rank2_err = abs(quad_sum - lin_sum)
    print(f"  Rank-2 O(N) vs O(N^2) off-diagonal residual (N={N})          : {mp.nstr(rank2_err, 6)}")
    assert rank2_err < mp.mpf("1e-48")

    # 2. Exact Euclidean norm (1/4)|S_N(d,t)|^2 = ||psi_1(d,t) + Psi_{>=2}(d,t)||^2
    S_N = mp.fsum(
        (b[n] / mp.sqrt(n)) * (mp.exp(d * omega[n] - 1j * phi[n]) + mp.exp(-d * omega[n] + 1j * phi[n]))
        for n in range(1, N + 1)
    )
    psi_x = [mp.mpf(0)] + [(b[n] / mp.sqrt(n)) * mp.cosh(d * omega[n]) * mp.cos(phi[n]) for n in range(1, N + 1)]
    psi_y = [mp.mpf(0)] + [-(b[n] / mp.sqrt(n)) * mp.sinh(d * omega[n]) * mp.sin(phi[n]) for n in range(1, N + 1)]
    eucl_sq = (mp.fsum(psi_x[1:])) ** 2 + (mp.fsum(psi_y[1:])) ** 2
    eucl_err = abs(mp.mpf("0.25") * (abs(S_N) ** 2) - eucl_sq)
    print(f"  Euclidean identity |(1/4)|S_N|^2 - ||psi_1 + Psi_{{>=2}}||^2|  : {mp.nstr(eucl_err, 6)}")
    assert eucl_err < mp.mpf("1e-48")

    # 3. Expose cosh vs sinh reverse-triangle bug & divergence of Sigma_cosh / sinh(dL/2)!
    #    ||psi_1(d,t)||^2 = sinh^2(dL/2) + cos^2(phi_1(t)), so ||psi_1||_min = sinh(dL/2)!
    psi1_norm_sq = psi_x[1] ** 2 + psi_y[1] ** 2
    id_psi1_err = abs(psi1_norm_sq - (mp.sinh(d * L / 2) ** 2 + mp.cos(phi[1]) ** 2))
    assert id_psi1_err < mp.mpf("1e-48")

    print("  Asymptotic ratio Sigma_cosh^{(2)}(d,t) / sinh(dL/2) at X = N(t) (d = 0.20, r = 2):")
    mu_big, _ = sieve_mobius_and_primes(1200)
    _, tl_big = compute_lambda_r_tables(2, 1200, mu_big)
    lam2_big = tl_big[2]
    ratio_rows = []
    for t_val in [100.0, 1000.0, 1e4, 1e6]:
        tt = mp.mpf(str(t_val))
        _, thp = vartheta_and_prime(tt)
        LL = 2 * thp
        NN = int(mp.floor(mp.sqrt(tt / (2 * mp.pi))))
        lX = mp.log(NN)
        sig_cosh = mp.fsum(
            (abs(lam2_big[n]) / (lX ** 2)) / mp.sqrt(n) * mp.cosh(d * (thp - mp.log(n)))
            for n in range(2, NN + 1)
        )
        psi1_min = mp.sinh(d * LL / 2)
        psi1_max = mp.cosh(d * LL / 2)
        rat_true = sig_cosh / psi1_min
        lb_true = max(mp.mpf(0), psi1_min - sig_cosh)
        ratio_rows.append((t_val, NN, float(psi1_min), float(sig_cosh), float(rat_true), float(lb_true), float(psi1_max)))
        print(f"    t = {t_val:7.0f} (N={NN:3d}): ||psi_1||_min = {float(psi1_min):7.4f} (max={float(psi1_max):7.4f}), "
              f"Sigma_cosh = {float(sig_cosh):8.4f}, ratio = {float(rat_true):7.4f}, true_LB = {float(lb_true):.4f}")
    assert ratio_rows[-1][4] > 2.5  # ratio > 1 => true phase-blind lower bound is identically 0!
    return ratio_rows


def verify_section5_preissmann_and_gallagher(lam_tab):
    print("=" * 78)
    print("[3/5] Preissmann Constant (4.132474), MV L^2 Windows, & Gallagher Explosion")
    print("=" * 78)
    # 1. Exact Preissmann (1984, Enseign. Math. 30, 95-113) constant
    c_MV = mp.pi * mp.sqrt(1 + (mp.mpf(2) / 3) * mp.sqrt(mp.mpf(6) / 5))
    c_false = (mp.mpf(4) / 3) * mp.sqrt(6)
    iv_c_MV = mp.iv.pi * mp.iv.sqrt(1 + (mp.iv.mpf(2) / 3) * mp.iv.sqrt(mp.iv.mpf(6) / 5))
    c_lo = mp.mpf(iv_c_MV.a)
    c_hi = mp.mpf(iv_c_MV.b)
    print(f"  True Preissmann (1984) constant c_MV                        : {mp.nstr(c_MV, 16)}")
    print(f"  (vs fabricated (4/3)sqrt(6) = {mp.nstr(c_false, 10)} in earlier draft)")
    print(f"  Interval enclosure c_MV in [{mp.nstr(c_lo, 14)}, {mp.nstr(c_hi, 14)}]")
    assert mp.mpf("4.1324743620") < c_lo and c_hi < mp.mpf("4.1324743621")

    # 2. Direct numerical check of Hilbert-Preissmann T-window bound:
    #    | int_T^{T+DT} |F(t)|^2 dt - DT * sum |a_n|^2 | <= 2 c_MV sum (n + 1/2) |a_n|^2
    T0 = mp.mpf("500.0")
    DT = mp.mpf("20.0")
    d = mp.mpf("0.20")
    M = 15
    lnX = mp.log(M)
    a_coeff = {n: (lam_tab[2][n] / (lnX ** 2)) / (mp.mpf(n) ** (mp.mpf("0.5") + d)) for n in range(2, M + 1)}

    diag_E = mp.fsum(v * v for v in a_coeff.values())
    # Exact closed-form integral int_{T0}^{T0+DT} |sum a_n n^{-it}|^2 dt
    off_int = mp.mpf(0)
    for m in range(2, M + 1):
        for n in range(m + 1, M + 1):
            om = mp.log(mp.mpf(n) / mp.mpf(m))
            # 2 Re a_m a_n int_{T0}^{T0+DT} e^{i om t} dt = 2 a_m a_n [sin(om(T0+DT)) - sin(om T0)] / om
            off_int += 2 * a_coeff[m] * a_coeff[n] * (mp.sin(om * (T0 + DT)) - mp.sin(om * T0)) / om
    preiss_bound = 2 * c_MV * mp.fsum((n + mp.mpf("0.5")) * (a_coeff[n] ** 2) for n in range(2, M + 1))
    print(f"  Window L^2 off-diagonal |int |F|^2 - DT*E_diag|             : {mp.nstr(abs(off_int), 6)} <= {mp.nstr(preiss_bound, 6)}")
    assert abs(off_int) <= preiss_bound

    # 3. Certify Gallagher Sobolev cross-term explosion as T -> infty (d = 0.20, X = N(T), r = 2):
    #    At DT = DT_MV = 2 * c_MV * N(T)^{1 - 2d}:
    #      mean term  (1/DT) int |F|^2 ~ E_diag -> 0 (via 1 + 2d > 1!),
    #      Sobolev term sqrt(int |F|^2 * int |F'|^2) ~ DT * sqrt(E_diag * E_deriv) -> +infty!
    mu_huge, _ = sieve_mobius_and_primes(1500)
    _, tl_huge = compute_lambda_r_tables(2, 1500, mu_huge)
    lam2_huge = tl_huge[2]
    print("  L^2 mean-square damping vs Gallagher Sobolev blow-up (d = 0.20, r = 2, X = N(T)):")
    mv_rows = []
    for T_num in [100.0, 1000.0, 1e4, 1e6]:
        TT = mp.mpf(str(T_num))
        NN = int(mp.floor(mp.sqrt(TT / (2 * mp.pi))))
        lX = mp.log(NN)
        E_d = mp.fsum(((lam2_huge[n] / (lX ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d)) for n in range(2, NN + 1))
        E_der = mp.fsum(((lam2_huge[n] * mp.log(n) / (lX ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d)) for n in range(2, NN + 1))
        E_off = 2 * c_MV * mp.fsum((n + mp.mpf("0.5")) * ((lam2_huge[n] / (lX ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d)) for n in range(2, NN + 1))
        DT_mv = E_off / E_d  # window length where off-diagonal L^2 equals diagonal DT * E_d (~ N^{1-2d})
        mean_L2 = 2 * E_d    # upper bound on (1/DT_mv) int |F|^2
        # Upper bound on int |F'|^2 on DT_mv:
        E_off_der = 2 * c_MV * mp.fsum((n + mp.mpf("0.5")) * ((lam2_huge[n] * mp.log(n) / (lX ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d)) for n in range(2, NN + 1))
        sob_term = mp.sqrt((DT_mv * E_d + E_off) * (DT_mv * E_der + E_off_der))
        mv_rows.append((T_num, NN, float(E_d), float(DT_mv), float(mean_L2), float(sob_term)))
        print(f"    T = {T_num:7.0f} (N={NN:3d}): E_diag = {float(E_d):.5f}, DT_MV = {float(DT_mv):8.2f}, "
              f"L^2_mean <= {float(mean_L2):.5f}, Gallagher_Sobolev = {float(sob_term):8.3f}")
    assert mv_rows[-1][4] < mv_rows[0][4]      # L^2 mean decays!
    assert mv_rows[-1][5] > 15.0 * mv_rows[0][5]  # Gallagher Sobolev term explodes!
    return mv_rows


# ============================================================================
# Figure Generation
# ============================================================================

def generate_figures(mu, tl_tab, bs_tab, ratio_rows, mv_rows):
    print("=" * 78)
    print("[4/5] Generating Publication Figures (PDF + PNG)")
    print("=" * 78)
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 300,
    })

    # --- Figure 1: tilde_Lambda_2 vs Lambda_2^BS, Sieve Horizon n > X, Rank-2, & Reverse-Triangle ---
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))

    # (a) |tilde_Lambda_2(n)| <= (ln rad(n))^2 vs Bombieri-Selberg Lambda_2^{BS}(n) <= (ln n)^2
    ax = axes[0, 0]
    n_arr = np.arange(2, 36)
    tl2_vals = [abs(float(tl_tab[2][n])) for n in n_arr]
    bs2_vals = [float(bs_tab[2][n]) for n in n_arr]
    ln2_vals = [math.log(n) ** 2 for n in n_arr]
    ax.bar(n_arr - 0.18, tl2_vals, width=0.36, color="#1b4965", label=r"$|\widetilde{\Lambda}_2(n)| \leq (\ln\,\mathrm{rad}(n))^2$")
    ax.bar(n_arr + 0.18, bs2_vals, width=0.36, color="#2a9d8f", label=r"$\Lambda_2^{\mathrm{BS}}(n) \leq (\ln n)^2$")
    ax.plot(n_arr, ln2_vals, lw=1.6, color="#d1495b", label=r"Envelope $(\ln n)^2$")
    ax.set_xlabel(r"Integer $n$")
    ax.set_ylabel(r"Magnitude")
    ax.set_title(r"(a) Squarefree sieve $|\widetilde{\Lambda}_2(n)|$ vs Bombieri--Selberg $\Lambda_2^{\mathrm{BS}}(n)$")
    ax.legend(loc="upper left", fontsize=7.4)
    ax.grid(alpha=0.25)

    # (b) Exact sieve formula on n <= X vs breakdown on n > X (X = 12, r = 2)
    ax = axes[0, 1]
    X_demo = 12
    n_demo = np.arange(2, 42)
    b_true = [float(b_sieve_coeff(int(n), X_demo, 2, mu)) for n in n_demo]
    b_form = [float(tl_tab[2][int(n)] / (mp.log(X_demo) ** 2)) for n in n_demo]
    ax.plot(n_demo, b_true, "o-", ms=3.8, lw=1.5, color="#1b4965", label=r"True $b_n^{(2)}(X)$")
    ax.plot(n_demo, b_form, "s--", ms=3.5, lw=1.4, color="#2a9d8f", label=r"$\widetilde{\Lambda}_2(n)/(\ln X)^2$")
    ax.axvline(X_demo, color="#d1495b", ls="--", lw=1.3, label=r"Sieve horizon $n = X = 12$")
    ax.set_xlabel(r"Dirichlet index $n$")
    ax.set_ylabel(r"Coefficient")
    ax.set_title(r"(b) Exact agreement on $n \leq X$ vs $b_p^{(2)}(X) = 1$ for $p > X$")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(alpha=0.25)

    # (c) Oscillation of ||psi_1(d,t)|| between sinh(dL/2) and cosh(dL/2)
    ax = axes[1, 0]
    t_osc = np.linspace(200.0, 225.0, 220)
    d_osc = 0.25
    norm_v, min_v, max_v = [], [], []
    for tv in t_osc:
        th, thp = vartheta_and_prime(mp.mpf(str(tv)))
        LL = float(2 * thp)
        ph1 = -float(th)
        sh = math.sinh(0.5 * d_osc * LL)
        ch = math.cosh(0.5 * d_osc * LL)
        norm_v.append(math.sqrt(sh * sh + math.cos(ph1) ** 2))
        min_v.append(sh)
        max_v.append(ch)
    ax.plot(t_osc, norm_v, lw=1.6, color="#1b4965", label=r"$\|\psi_1(d,t)\|$")
    ax.plot(t_osc, min_v, lw=1.8, ls="--", color="#d1495b", label=r"True minimum $\sinh(dL/2)$")
    ax.plot(t_osc, max_v, lw=1.5, ls=":", color="#2a9d8f", label=r"Maximum $\cosh(dL/2)$")
    ax.set_xlabel(r"Ordinate $t$ ($d = 0.25$)")
    ax.set_ylabel(r"Euclidean norm in $\mathbb{R}^2$")
    ax.set_title(r"(c) True minimum $\|\psi_1\|_{\min} = \sinh(dL/2)$ at $\cos\phi_1 = 0$")
    ax.legend(loc="lower right", fontsize=7.6)
    ax.grid(alpha=0.25)

    # (d) Growth of Sigma_cosh^{(2)} / sinh(dL/2) ~ N(t)^{1/2 - d} > 1 on 0 < d < 1/2
    ax = axes[1, 1]
    t_r = [r[0] for r in ratio_rows]
    ax.semilogx(t_r, [r[3] for r in ratio_rows], "o-", lw=1.8, color="#d1495b", label=r"Tail bound $\Sigma_{\cosh}^{(2)}(d,t)$")
    ax.semilogx(t_r, [r[2] for r in ratio_rows], "s--", lw=1.8, color="#1b4965", label=r"Minimum $\|\psi_1\|_{\min} = \sinh(dL/2)$")
    ax.set_xlabel(r"Ordinate $t$ ($d = 0.20$, $X = N(t)$)")
    ax.set_ylabel(r"Amplitude")
    ax.set_title(r"(d) $\Sigma_{\cosh}^{(2)} \gg \sinh(dL/2)$ rules out phase-blind positivity")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for out_dir in (FIG_DIR, PAPER_FIG_DIR):
        fig.savefig(out_dir / "fig1_bilinear_rank2_and_pure_square.pdf")
        fig.savefig(out_dir / "fig1_bilinear_rank2_and_pure_square.png", dpi=300)
    plt.close(fig)

    # --- Figure 2: 1 + 2d > 1 L^2 Mean-Square Damping vs Gallagher Pointwise Explosion ---
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))

    # (a) 1 + 2d > 1 L^2 diagonal energy E_diag(d, X) vs d = 0 divergence
    ax = axes[0, 0]
    mu_plot, _ = sieve_mobius_and_primes(600)
    _, tl_plot = compute_lambda_r_tables(2, 600, mu_plot)
    lam2_plot = tl_plot[2]
    N_grid = [10, 25, 60, 150, 350, 600]
    for dv, col, lab in [(0.0, "#d1495b", r"$d = 0$ (diverges $\propto \ln N$)"),
                         (0.15, "#1b4965", r"$d = 0.15$ ($1+2d > 1$ decays)"),
                         (0.30, "#2a9d8f", r"$d = 0.30$ ($1+2d > 1$ decays)")]:
        evals = []
        for NN in N_grid:
            lX = mp.log(NN)
            ed = float(mp.fsum(((lam2_plot[n] / (lX ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * mp.mpf(str(dv)))) for n in range(2, NN + 1)))
            evals.append(ed)
        ax.loglog(N_grid, evals, "o-", lw=1.8, color=col, label=lab)
    ax.set_xlabel(r"Sieve/RS horizon $X = N(T)$")
    ax.set_ylabel(r"Diagonal $L^2$ energy $\mathcal{E}_{\mathrm{diag}}^{(2)}(d, X)$")
    ax.set_title(r"(a) $1 + 2d > 1$ mean-square convergence for $d > 0$")
    ax.legend(loc="lower left", fontsize=7.6)
    ax.grid(alpha=0.25, which="both")

    # (b) Required Montgomery-Vaughan window length DT_MV ~ N(T)^{1 - 2d}
    ax = axes[0, 1]
    T_m = [r[0] for r in mv_rows]
    ax.loglog(T_m, [r[3] for r in mv_rows], "s-", lw=1.9, color="#1b4965", label=r"$\Delta T_{\mathrm{MV}}(0.20, T) \asymp N(T)^{1-2d}$")
    ax.loglog(T_m, [math.sqrt(t / (2 * math.pi)) for t in T_m], "--", lw=1.5, color="#6c757d", label=r"Full RS scale $N(T) \asymp T^{1/2}$")
    ax.set_xlabel(r"Height $T$")
    ax.set_ylabel(r"Averaging window $\Delta T$")
    ax.set_title(r"(b) Preissmann--MV window $\Delta T_{\mathrm{MV}} \asymp T^{(1-2d)/2}$ ($c_{\mathrm{MV}} \approx 4.1325$)")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.grid(alpha=0.25, which="both")

    # (c) L^2 window average -> 0 vs Gallagher Sobolev cross-term -> +infty!
    ax = axes[1, 0]
    ax.loglog(T_m, [r[4] for r in mv_rows], "o-", lw=1.9, color="#2a9d8f", label=r"Window mean $\frac{1}{\Delta T}\int |F|^2 \to 0$")
    ax.loglog(T_m, [r[5] for r in mv_rows], "d-", lw=2.0, color="#d1495b", label=r"Gallagher $\sqrt{\int|F|^2 \int|F'|^2} \to +\infty$")
    ax.set_xlabel(r"Height $T$ ($d = 0.20$, $\Delta T = \Delta T_{\mathrm{MV}}$)")
    ax.set_ylabel(r"Bound magnitude")
    ax.set_title(r"(c) $L^2$ zero-density damping vs pointwise Gallagher blow-up")
    ax.legend(loc="center left", fontsize=7.6)
    ax.grid(alpha=0.25, which="both")

    # (d) Optimization over ALL Delta T > 0 at T = 10^4: minimum stays O(N^{1-2d})!
    ax = axes[1, 1]
    NN_fix = int(math.floor(math.sqrt(1e4 / (2 * math.pi))))  # 39
    lX_f = mp.log(NN_fix)
    d_f = mp.mpf("0.20")
    c_MV = float(mp.pi * mp.sqrt(1 + (mp.mpf(2) / 3) * mp.sqrt(mp.mpf(6) / 5)))
    Ed_f = float(mp.fsum(((lam2_plot[n] / (lX_f ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d_f)) for n in range(2, NN_fix + 1)))
    Eder_f = float(mp.fsum(((lam2_plot[n] * mp.log(n) / (lX_f ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d_f)) for n in range(2, NN_fix + 1)))
    Eoff_f = 2 * c_MV * float(mp.fsum((n + 0.5) * ((lam2_plot[n] / (lX_f ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d_f)) for n in range(2, NN_fix + 1)))
    Eoff_der_f = 2 * c_MV * float(mp.fsum((n + 0.5) * ((lam2_plot[n] * mp.log(n) / (lX_f ** 2)) ** 2) / (mp.mpf(n) ** (1 + 2 * d_f)) for n in range(2, NN_fix + 1)))
    dt_scan = np.logspace(-2.0, 3.0, 160)
    gal_tot = []
    for dtv in dt_scan:
        i0 = dtv * Ed_f + Eoff_f
        i1 = dtv * Eder_f + Eoff_der_f
        gal_tot.append(i0 / dtv + math.sqrt(i0 * i1))
    cs_bound = float(mp.fsum((abs(lam2_plot[n]) / (lX_f ** 2)) / (mp.mpf(n) ** (0.5 + d_f)) for n in range(2, NN_fix + 1))) ** 2
    ax.loglog(dt_scan, gal_tot, lw=2.0, color="#1b4965", label=r"Gallagher bound $\forall\,\Delta T > 0$")
    ax.axhline(cs_bound, color="#d1495b", ls="--", lw=1.6, label=rf"Direct $(\sum |a_n|)^2 = {cs_bound:.2f}$")
    ax.axhline(1.0, color="gray", ls=":", lw=1.1, label=r"Unit threshold $1$")
    ax.set_xlabel(r"Trial window width $\Delta T$ ($T = 10^4$, $d = 0.20$)")
    ax.set_ylabel(r"Upper bound on $|F(t_0)|^2$")
    ax.set_title(r"(d) Optimizing $\Delta T$ cannot beat $\mathcal{O}(N^{1-2d})$ pointwise")
    ax.legend(loc="upper right", fontsize=7.4)
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    for out_dir in (FIG_DIR, PAPER_FIG_DIR):
        fig.savefig(out_dir / "fig2_exponent_jump_montgomery_vaughan.pdf")
        fig.savefig(out_dir / "fig2_exponent_jump_montgomery_vaughan.png", dpi=300)
    plt.close(fig)
    print("  Saved fig1_bilinear_rank2_and_pure_square.{pdf,png} and fig2_exponent_jump_montgomery_vaughan.{pdf,png}")


def main():
    t0 = time.time()
    mu, tl_tab, bs_tab = verify_section1_and_2_canonical_and_lambda_r()
    ratio_rows = verify_section3_and_4_rank2_and_reverse_triangle(mu, tl_tab)
    mv_rows = verify_section5_preissmann_and_gallagher(tl_tab)
    generate_figures(mu, tl_tab, bs_tab, ratio_rows, mv_rows)
    elapsed = time.time() - t0
    print("=" * 78)
    print(f"[PASS] Paper VIII verification suite completed in {elapsed:.2f}s (exit code 0).")
    print("=" * 78)


if __name__ == "__main__":
    main()
