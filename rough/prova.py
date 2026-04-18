"""
Rough Heston pipeline: Markovian approximation, MC simulator, barrier analyzer,
density analyzer.

Formule di riferimento
----------------------
Modello Rough Heston (El Euch & Rosenbaum, 2019, "The characteristic function
of rough Heston models", Math. Finance 29(1):3-38):

    dX_t = (mu - V_t/2) dt + sqrt(V_t) dB_t,        X_t = log(S_t / S_0)

    V_t = V_0 + (1/Gamma(alpha)) * integral_0^t (t-s)^(alpha-1) *
          [ kappa*(theta - V_s) ds + xi*sqrt(V_s) dW_s ]

    B_t = rho * W_t + sqrt(1 - rho^2) * Z_t,        W _|_ Z Brownians

    alpha = H + 1/2,   H in (0, 1/2)  (rough regime)

Markovian approximation (Abi Jaber & El Euch, 2019, "Multi-factor approximation
of rough volatility models", SIAM J. Financial Math. 10(2):309-349):

Il kernel frazionario K_alpha(t) = t^(alpha-1)/Gamma(alpha) e' la trasformata
di Laplace della misura  mu(dx) = x^(-alpha) / (Gamma(alpha) * Gamma(1-alpha)) dx :

    K_alpha(t) = integral_0^infty exp(-x*t) mu(dx)

Discretizzata con partizione geometrica eta_i = r^(i - N/2), i = 0,...,N,
definiamo

    c_i     = integral_{eta_{i-1}}^{eta_i} mu(dx)
            = (eta_i^(1-a) - eta_{i-1}^(1-a)) / ((1-a) * Gamma(a) * Gamma(1-a))

    gamma_i = (1/c_i) * integral_{eta_{i-1}}^{eta_i} x * mu(dx)
            = ((1-a)/(2-a)) *
              (eta_i^(2-a) - eta_{i-1}^(2-a)) / (eta_i^(1-a) - eta_{i-1}^(1-a))

(con a = alpha). Allora K_alpha(t) ~= sum_i c_i * exp(-gamma_i t), e il processo
V si scrive come somma di N fattori di Ornstein-Uhlenbeck-like driven dal medesimo W:

    V_t = V_0 + sum_{i=1}^N c_i * U_i(t)

    dU_i(t) = [ kappa*(theta - V_t) - gamma_i*U_i(t) ] dt
              + xi*sqrt(V_t) dW_t,     U_i(0) = 0

Schema numerico
---------------
Euler-Maruyama con "full truncation" (Lord, Koekkoek & van Dijk, 2010): ovunque
comparga V come driver di diffusione o drift si usa max(V, 0). Questo garantisce
che sqrt(V) sia ben definito e previene l'esplosione numerica dovuta alle
incursioni negative tipiche dello schema Euler su CIR-like.

Sanity checks incorporati
-------------------------
- Verifica approssimazione del kernel su grid logaritmica.
- Riporta frazione di step dove V e' scesa sotto zero (full trunc la gestisce,
  ma se e' elevata aumenta n_steps o N_factors).
- Ties nei first-hitting time sono risolti pessimisticamente (SL).

AVVERTENZE
----------
- Sotto Q (risk-neutral), drift = r - q (per crypto perp ~ funding rate).
  Sotto P, drift = mu_atteso. Parametro drift esplicito, NON lasciare default 0
  senza pensarci.
- Fees/funding/slippage: incluso solo un fee per side, flat. Per backtest
  seri aggiungi il funding rate integrato sul tempo di holding.
- La simulazione discreta SOTTOSTIMA la probabilita' di hit di barriere strette
  (continuity correction non implementata). Se le barriere sono a meno di
  1-2 sigma della volatilita' sullo step, raddoppia n_steps e verifica
  che le probabilita' non si muovano piu'.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy import stats


# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------

@dataclass
class RoughHestonParams:
    """
    Parametri del modello Rough Heston.

    V0    : variance spot (annualizzata)
    kappa : velocita' di mean reversion
    theta : long-run variance (annualizzata)
    xi    : vol-of-vol
    rho   : correlazione W-Z (tipicamente negativa, leverage effect)
    H     : Hurst exponent, in (0, 0.5) per regime rough
    """
    V0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    H: float

    @property
    def alpha(self) -> float:
        return self.H + 0.5

    def validate(self) -> None:
        if self.V0 <= 0:
            raise ValueError(f"V0 must be > 0, got {self.V0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if self.xi <= 0:
            raise ValueError(f"xi must be > 0, got {self.xi}")
        if not (-1 < self.rho < 1):
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if not (0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")


# ---------------------------------------------------------------------------
# Markovian approximation
# ---------------------------------------------------------------------------

def abi_jaber_coefficients(N: int, alpha: float, r: float = 2.5):
    """
    Coefficienti (c_i, gamma_i)_{i=1..N} di Abi Jaber & El Euch (2019).

    Partizione geometrica: eta_i = r^(i - N/2) per i = 0,1,...,N.

    Returns
    -------
    c     : ndarray (N,)  scale weights
    gamma : ndarray (N,)  mean-reversion rates (positivi, ordinati crescente)
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha in (0,1) richiesto, got {alpha}")
    if N < 1:
        raise ValueError("N >= 1")
    if r <= 1:
        raise ValueError("r > 1")

    idx = np.arange(N + 1, dtype=float) - N / 2.0
    eta = r ** idx

    gamma_factor = gamma_fn(alpha) * gamma_fn(1.0 - alpha)

    eta1 = eta[:-1]
    eta2 = eta[1:]

    c = (eta2 ** (1.0 - alpha) - eta1 ** (1.0 - alpha)) / ((1.0 - alpha) * gamma_factor)

    num_g = eta2 ** (2.0 - alpha) - eta1 ** (2.0 - alpha)
    den_g = eta2 ** (1.0 - alpha) - eta1 ** (1.0 - alpha)
    gamma_vec = ((1.0 - alpha) / (2.0 - alpha)) * (num_g / den_g)

    return c, gamma_vec


def check_kernel_approximation(c, gamma_vec, alpha, t_grid=None):
    """
    Verifica |K_alpha(t) - sum c_i exp(-gamma_i t)| / K_alpha(t) su grid log.

    Returns dict con t, exact, approx, rel_err e sintesi.
    """
    if t_grid is None:
        t_grid = np.logspace(-3, 1, 200)
    exact = t_grid ** (alpha - 1.0) / gamma_fn(alpha)
    approx = (c[:, None] * np.exp(-gamma_vec[:, None] * t_grid[None, :])).sum(axis=0)
    rel_err = np.abs(approx - exact) / np.abs(exact)
    return {
        "t": t_grid,
        "exact": exact,
        "approx": approx,
        "rel_err": rel_err,
        "max_rel_err": float(rel_err.max()),
        "mean_rel_err": float(rel_err.mean()),
        "median_rel_err": float(np.median(rel_err)),
    }


# ---------------------------------------------------------------------------
# Simulatore forward
# ---------------------------------------------------------------------------

def simulate_rough_heston(
    params: RoughHestonParams,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    N_factors: int = 20,
    r: float = 2.5,
    drift: float = 0.0,
    scheme: Literal["full_truncation", "reflection"] = "full_truncation",
    seed: Optional[int] = None,
    return_variance: bool = True,
    antithetic: bool = False,
):
    """
    Simula path S_t e V_t sotto rough Heston via Markovian approximation.

    Parametri
    ---------
    drift : float
        Drift del log-prezzo per unita' di tempo (annualizzato).
        Sotto Q: drift = r - q (per crypto ~ funding).
        Sotto P: drift = mu stimato.
    antithetic : bool
        Se True, meta' dei path usano gli increments, l'altra meta' i loro
        negativi (variance reduction; riduce error su media, non sulla coda).

    Returns
    -------
    S_paths : ndarray (n_paths, n_steps+1)
    V_paths : ndarray (n_paths, n_steps+1)   [se return_variance]
    """
    params.validate()

    c, gamma_vec = abi_jaber_coefficients(N_factors, params.alpha, r=r)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    rng = np.random.default_rng(seed)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths pari per antithetic")
        half = n_paths // 2
        dW_h = rng.standard_normal(size=(half, n_steps)) * sqrt_dt
        dZ_h = rng.standard_normal(size=(half, n_steps)) * sqrt_dt
        dW = np.concatenate([dW_h, -dW_h], axis=0)
        dZ = np.concatenate([dZ_h, -dZ_h], axis=0)
    else:
        dW = rng.standard_normal(size=(n_paths, n_steps)) * sqrt_dt
        dZ = rng.standard_normal(size=(n_paths, n_steps)) * sqrt_dt

    dB = params.rho * dW + np.sqrt(1.0 - params.rho ** 2) * dZ

    log_S = np.full(n_paths, np.log(S0))
    U = np.zeros((n_paths, N_factors))
    V = np.full(n_paths, params.V0)

    S_paths = np.empty((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    V_paths = np.empty((n_paths, n_steps + 1)) if return_variance else None
    if V_paths is not None:
        V_paths[:, 0] = params.V0

    n_neg_V = 0  # diagnostic
    # pre-compute A-stable denominator per U_i implicit step: (1 + gamma_i * dt)
    impl_den = 1.0 + gamma_vec * dt  # shape (N_factors,)

    for n in range(n_steps):
        if scheme == "full_truncation":
            V_pos = np.maximum(V, 0.0)
        elif scheme == "reflection":
            V_pos = np.abs(V)
        else:
            raise ValueError(f"scheme unknown: {scheme}")

        n_neg_V += int((V < 0).sum())
        sqrt_V = np.sqrt(V_pos)

        drift_v = params.kappa * (params.theta - V_pos)  # (n_paths,)

        # Semi-implicit Euler su U_i (A-stable, necessario perche' gamma_i puo'
        # essere grande per i alti => explicit esplode con gamma_i*dt > 2):
        #
        #   U_i^{n+1} = (U_i^n + drift_v*dt + xi*sqrt(V)*dW_n) / (1 + gamma_i*dt)
        #
        # Derivazione:
        #   U_i^{n+1} = U_i^n + [drift_v - gamma_i U_i^{n+1}] dt + xi sqrt(V) dW_n
        #   => U_i^{n+1} (1 + gamma_i dt) = U_i^n + drift_v dt + xi sqrt(V) dW_n
        # (diffusion trattata esplicitamente; drift_v valutato al tempo n
        # per evitare non-linearita' implicita su V_t)
        numer = U + drift_v[:, None] * dt + \
            params.xi * sqrt_V[:, None] * dW[:, n:n + 1]
        U = numer / impl_den[None, :]

        V = params.V0 + U @ c

        log_S = log_S + (drift - 0.5 * V_pos) * dt + sqrt_V * dB[:, n]

        S_paths[:, n + 1] = np.exp(log_S)
        if V_paths is not None:
            V_paths[:, n + 1] = V

    total_V_obs = n_paths * n_steps
    frac_neg_V = n_neg_V / total_V_obs if total_V_obs > 0 else 0.0

    if return_variance:
        return S_paths, V_paths, {"frac_neg_V_pre_trunc": frac_neg_V}
    return S_paths, {"frac_neg_V_pre_trunc": frac_neg_V}


# ---------------------------------------------------------------------------
# Barrier (TP / SL) analyzer
# ---------------------------------------------------------------------------

@dataclass
class BarrierResult:
    outcomes: np.ndarray     # '<U8': 'TP' | 'SL' | 'TIMEOUT'
    hit_times: np.ndarray    # anni
    pnl_net: np.ndarray      # fractional, NET di fees + leverage applicata
    final_prices: np.ndarray
    # summary
    p_tp: float
    p_sl: float
    p_timeout: float
    expected_pnl: float
    pnl_std: float
    sharpe: float
    median_time_to_tp: float
    median_time_to_sl: float
    win_rate: float
    profit_factor: float
    expectancy_R: float      # E[PnL] / sl_pct, in "R"


class BarrierAnalyzer:
    """
    Calcola first-hitting-time di barriere TP / SL su path simulati.

    Bias noti:
    - Path discreti -> sottostima P(hit) per barriere strette. Controllo
      incrementando n_steps.
    - Non considera liquidation explicit (per leverage alto e SL largo la
      liquidation puo' colpire prima dello SL; aggiungilo come SL_eff).
    - Fees flat per lato; niente funding integrato.
    """

    @staticmethod
    def analyze(
        S_paths: np.ndarray,
        S0: float,
        tp_pct: float,
        sl_pct: float,
        dt: float,
        direction: Literal["long", "short"] = "long",
        leverage: float = 1.0,
        fees_per_side: float = 0.0,
    ) -> BarrierResult:
        if tp_pct <= 0 or sl_pct <= 0:
            raise ValueError("tp_pct, sl_pct > 0")

        n_paths, n_points = S_paths.shape
        T_total = (n_points - 1) * dt

        if direction == "long":
            up_barrier = S0 * (1.0 + tp_pct)
            down_barrier = S0 * (1.0 - sl_pct)
            hit_tp_mask = S_paths >= up_barrier
            hit_sl_mask = S_paths <= down_barrier
        elif direction == "short":
            up_barrier = S0 * (1.0 + sl_pct)       # avversa: prezzo sale -> SL
            down_barrier = S0 * (1.0 - tp_pct)     # favorevole: prezzo scende -> TP
            hit_tp_mask = S_paths <= down_barrier
            hit_sl_mask = S_paths >= up_barrier
        else:
            raise ValueError(f"direction: {direction}")

        tp_hit = hit_tp_mask.any(axis=1)
        sl_hit = hit_sl_mask.any(axis=1)

        first_tp = np.argmax(hit_tp_mask, axis=1).astype(float)
        first_sl = np.argmax(hit_sl_mask, axis=1).astype(float)
        first_tp[~tp_hit] = np.inf
        first_sl[~sl_hit] = np.inf

        outcomes = np.full(n_paths, "TIMEOUT", dtype="<U8")
        outcomes[first_tp < first_sl] = "TP"
        outcomes[first_sl < first_tp] = "SL"
        tie = np.isfinite(first_tp) & np.isfinite(first_sl) & (first_tp == first_sl)
        outcomes[tie] = "SL"  # tie -> pessimistico

        hit_times = np.full(n_paths, T_total)
        tp_mask = outcomes == "TP"
        sl_mask = outcomes == "SL"
        hit_times[tp_mask] = first_tp[tp_mask] * dt
        hit_times[sl_mask] = first_sl[sl_mask] * dt

        final_prices = S_paths[:, -1]
        to_mask = outcomes == "TIMEOUT"

        # PnL gross (fractional rispetto a notional)
        pnl_gross = np.empty(n_paths)
        if direction == "long":
            pnl_gross[tp_mask] = tp_pct
            pnl_gross[sl_mask] = -sl_pct
            pnl_gross[to_mask] = (final_prices[to_mask] - S0) / S0
        else:
            pnl_gross[tp_mask] = tp_pct
            pnl_gross[sl_mask] = -sl_pct
            pnl_gross[to_mask] = (S0 - final_prices[to_mask]) / S0

        pnl_net = pnl_gross * leverage - 2.0 * fees_per_side  # entry + exit

        p_tp = float(tp_mask.mean())
        p_sl = float(sl_mask.mean())
        p_timeout = float(to_mask.mean())
        expected_pnl = float(pnl_net.mean())
        pnl_std = float(pnl_net.std(ddof=1)) if n_paths > 1 else 0.0
        sharpe_like = expected_pnl / pnl_std if pnl_std > 0 else 0.0

        tt_tp = hit_times[tp_mask]
        tt_sl = hit_times[sl_mask]
        med_tp = float(np.median(tt_tp)) if len(tt_tp) > 0 else np.nan
        med_sl = float(np.median(tt_sl)) if len(tt_sl) > 0 else np.nan

        wins = pnl_net > 0
        losses = pnl_net < 0
        win_rate = float(wins.mean())
        gp = pnl_net[wins].sum()
        gl = -pnl_net[losses].sum()
        pf = float(gp / gl) if gl > 0 else float("inf")

        # Expectancy in R (risk units): 1R = sl_pct * leverage (perdita massima nominale)
        risk_per_trade = sl_pct * leverage
        expectancy_R = expected_pnl / risk_per_trade if risk_per_trade > 0 else 0.0

        return BarrierResult(
            outcomes=outcomes,
            hit_times=hit_times,
            pnl_net=pnl_net,
            final_prices=final_prices,
            p_tp=p_tp,
            p_sl=p_sl,
            p_timeout=p_timeout,
            expected_pnl=expected_pnl,
            pnl_std=pnl_std,
            sharpe=sharpe_like,
            median_time_to_tp=med_tp,
            median_time_to_sl=med_sl,
            win_rate=win_rate,
            profit_factor=pf,
            expectancy_R=expectancy_R,
        )

    @staticmethod
    def sweep(
        S_paths: np.ndarray,
        S0: float,
        tp_range: np.ndarray,
        sl_range: np.ndarray,
        dt: float,
        direction: Literal["long", "short"] = "long",
        leverage: float = 1.0,
        fees_per_side: float = 0.0,
    ) -> List[Dict[str, float]]:
        """Cartesiano su (tp, sl). Ritorna lista di dict (compatibile con Polars)."""
        out = []
        for tp in tp_range:
            for sl in sl_range:
                r = BarrierAnalyzer.analyze(
                    S_paths, S0, float(tp), float(sl), dt,
                    direction, leverage, fees_per_side
                )
                out.append({
                    "tp_pct": float(tp),
                    "sl_pct": float(sl),
                    "p_tp": r.p_tp,
                    "p_sl": r.p_sl,
                    "p_timeout": r.p_timeout,
                    "expected_pnl": r.expected_pnl,
                    "pnl_std": r.pnl_std,
                    "sharpe": r.sharpe,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "expectancy_R": r.expectancy_R,
                    "median_time_to_tp": r.median_time_to_tp,
                    "median_time_to_sl": r.median_time_to_sl,
                })
        return out


# ---------------------------------------------------------------------------
# Density analyzer
# ---------------------------------------------------------------------------

@dataclass
class DensityResult:
    n: int
    mean: float
    std: float
    skew: float
    kurt_excess: float     # excess (Fisher): kurt - 3
    jb_stat: float
    jb_pvalue: float
    var_95: float          # 5% quantile
    var_99: float
    es_95: float           # Expected Shortfall condizionata a <= VaR_95
    es_99: float
    hill_right: float      # tail index coda destra
    hill_left: float       # tail index coda sinistra
    student_t_df: float
    student_t_loglik: float
    normal_loglik: float
    bic_normal: float
    bic_student_t: float
    density_grid: np.ndarray
    density_values: np.ndarray


class DensityAnalyzer:
    """
    Diagnostica della densita' dei rendimenti (raccomandato LOG-rendimenti).

    Formule:
    - kurt excess (Fisher): K_e = E[(X-mu)^4]/sigma^4 - 3
    - Jarque-Bera: JB = (n/6) * (skew^2 + kurt_excess^2/4), ~ chi^2(2) sotto H0 N.
    - Hill estimator (Hill 1975):
        x_(1) >= x_(2) >= ... >= x_(n)  ordinati decrescenti
        xi_hat(k) = (1/k) sum_{i=1..k} log(x_(i)) - log(x_(k+1))
        tail_index alpha = 1 / xi_hat
      Convenzione: P(X > u) ~ L(u) * u^(-alpha) per u grande.
      alpha < 4 -> kurtosi infinita.  alpha < 2 -> varianza infinita.
      alpha < 1 -> media infinita.
    - Student-t fit via MLE (scipy.stats.t.fit).
    - BIC = k log(n) - 2 log L    (Normal: k=2; Student-t con loc/scale: k=3).
    """

    @staticmethod
    def analyze(
        returns: np.ndarray,
        k_hill: Optional[int] = None,
        grid_points: int = 400,
    ) -> DensityResult:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 20:
            raise ValueError(f"Too few returns: {n}")

        mean = float(np.mean(r))
        std = float(np.std(r, ddof=1))
        skew = float(stats.skew(r))
        kurt_excess = float(stats.kurtosis(r, fisher=True))

        jb_stat, jb_pvalue = stats.jarque_bera(r)

        var_95 = float(np.quantile(r, 0.05))
        var_99 = float(np.quantile(r, 0.01))
        left_95 = r[r <= var_95]
        left_99 = r[r <= var_99]
        es_95 = float(left_95.mean()) if len(left_95) > 0 else var_95
        es_99 = float(left_99.mean()) if len(left_99) > 0 else var_99

        hill_r = DensityAnalyzer._hill(r, side="right", k=k_hill)
        hill_l = DensityAnalyzer._hill(r, side="left", k=k_hill)

        # MLE Student-t (df, loc, scale)
        t_df, t_loc, t_scale = stats.t.fit(r)
        t_loglik = float(stats.t.logpdf(r, t_df, t_loc, t_scale).sum())
        n_loglik = float(stats.norm.logpdf(r, mean, std).sum())

        # BIC: k log n - 2 log L  (convention: piu' basso = meglio)
        bic_normal = 2.0 * np.log(n) - 2.0 * n_loglik
        bic_t = 3.0 * np.log(n) - 2.0 * t_loglik

        # KDE (Silverman di default)
        kde = stats.gaussian_kde(r)
        grid = np.linspace(r.min(), r.max(), grid_points)
        dens = kde(grid)

        return DensityResult(
            n=n,
            mean=mean, std=std, skew=skew, kurt_excess=kurt_excess,
            jb_stat=float(jb_stat), jb_pvalue=float(jb_pvalue),
            var_95=var_95, var_99=var_99, es_95=es_95, es_99=es_99,
            hill_right=hill_r, hill_left=hill_l,
            student_t_df=float(t_df),
            student_t_loglik=t_loglik,
            normal_loglik=n_loglik,
            bic_normal=float(bic_normal),
            bic_student_t=float(bic_t),
            density_grid=grid,
            density_values=dens,
        )

    @staticmethod
    def _hill(x: np.ndarray, side: Literal["right", "left"], k: Optional[int] = None) -> float:
        if side == "right":
            tail = x[x > 0]
        else:
            tail = -x[x < 0]
        n_tail = len(tail)
        if n_tail < 20:
            return float("nan")
        s = np.sort(tail)[::-1]
        if k is None:
            k = max(20, int(0.1 * n_tail))
        k = min(k, n_tail - 1)
        log_top = np.log(s[:k])
        log_kp1 = np.log(s[k])
        xi_inv = log_top.mean() - log_kp1
        if xi_inv <= 0:
            return float("nan")
        return 1.0 / xi_inv


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    S0: float,
    horizon_days: float,
    params: RoughHestonParams,
    drift_annual: float = 0.0,
    n_steps: int = 500,
    n_paths: int = 20000,
    N_factors: int = 20,
    tp_range: Optional[np.ndarray] = None,
    sl_range: Optional[np.ndarray] = None,
    direction: Literal["long", "short"] = "long",
    leverage: float = 1.0,
    fees_per_side: float = 0.0006,
    seed: Optional[int] = 42,
    antithetic: bool = True,
) -> Dict[str, Any]:
    """
    1) Abi Jaber coefficients + kernel sanity check
    2) Simulazione MC di S_t e V_t
    3) Density analysis sui log-return all'orizzonte
    4) Sweep TP/SL
    """
    T = horizon_days / 365.0

    c, gamma_vec = abi_jaber_coefficients(N_factors, params.alpha)
    kernel_check = check_kernel_approximation(c, gamma_vec, params.alpha)

    S_paths, V_paths, sim_diag = simulate_rough_heston(
        params, S0, T, n_steps, n_paths,
        N_factors=N_factors, drift=drift_annual,
        seed=seed, return_variance=True, antithetic=antithetic,
    )
    dt = T / n_steps

    log_returns = np.log(S_paths[:, -1] / S0)
    density = DensityAnalyzer.analyze(log_returns)

    if tp_range is None:
        tp_range = np.array([0.005, 0.01, 0.015, 0.02, 0.03])
    if sl_range is None:
        sl_range = np.array([0.005, 0.01, 0.015, 0.02, 0.03])

    sweep = BarrierAnalyzer.sweep(
        S_paths, S0, tp_range, sl_range, dt, direction,
        leverage, fees_per_side,
    )

    variance_stats = {
        "V_mean_path": float(V_paths.mean()),
        "V_median_terminal": float(np.median(V_paths[:, -1])),
        "V_q05_terminal": float(np.quantile(V_paths[:, -1], 0.05)),
        "V_q95_terminal": float(np.quantile(V_paths[:, -1], 0.95)),
        "V_min_path": float(V_paths.min()),
        "frac_neg_V_pre_trunc": sim_diag["frac_neg_V_pre_trunc"],
    }

    return {
        "params": params,
        "S0": S0,
        "T_years": T,
        "dt": dt,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "kernel_approximation_check": kernel_check,
        "S_paths": S_paths,
        "V_paths": V_paths,
        "log_returns_horizon": log_returns,
        "density": density,
        "barrier_sweep": sweep,
        "variance_stats": variance_stats,
    }


# =============================================================================
# MODULO DI CALIBRAZIONE
# =============================================================================
#
# Calibrazione di Rough Heston a una IV surface via:
#   - characteristic function tramite fractional Riccati (El Euch-Rosenbaum 2019)
#   - pricing di call europee via COS method (Fang-Oosterlee 2008)
#   - inversione BS per ricavare IV modello
#   - minimizzazione RMSE(IV_model - IV_market) via differential evolution
#
# Riferimento tesi: Bertolo, "Numerical methods for Rough Heston models",
# Padova 2024.
#
# Formule core:
#
# Char fn di X_T = log(S_T / S_0) - r*T  (contributo r aggiunto esternamente):
#
#     phi(u; T) = exp( V_0 * psi_1(u, T) + kappa*theta * psi_2(u, T) )
#
# con:
#     psi_1(u, T) = I^(1-alpha) h(u, .)(T)     (integrale frazionario RL)
#     psi_2(u, T) = integral_0^T h(u, s) ds
#
# h soddisfa la fractional Riccati (derivata di Caputo di ordine alpha = H+1/2):
#
#     D^alpha h(u, t) = 1/2 (-u^2 - iu) + (iu*rho*xi - kappa) h + (xi^2/2) h^2
#     h(u, 0) = 0
#
# COS method per call europea:
#
#     C(S_0, K, T) = exp(-rT) sum_{k=0}^{N-1}'
#                    Re[phi_x(u_k) exp(-i u_k a)] * V_k(K)
#
# con u_k = k pi / (b-a), phi_x(u) = phi_Y(u) * exp(i u log(F/K)),
# F = S_0 exp(rT), V_k(K) = (2K/(b-a))(chi_k(0,b) - psi_k(0,b)).
# =============================================================================

from scipy import stats as _stats
from scipy import optimize as _optimize


# ------------------ Black-Scholes utilities ------------------

def bs_call_price(S, K, T, r, sigma):
    """Prezzo call europea Black-Scholes."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _stats.norm.cdf(d1) - K * np.exp(-r * T) * _stats.norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Prezzo put via parita' put-call."""
    return bs_call_price(S, K, T, r, sigma) + K * np.exp(-r * T) - S


def bs_implied_vol(price, S, K, T, r, option_type="call"):
    """
    Inversione Newton-Raphson con fallback a Brent su sigma in [1e-6, 5.0].
    Restituisce NaN se il prezzo e' fuori bounds di no-arbitrage.
    """
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
        upper = S
        f = lambda sig: bs_call_price(S, K, T, r, sig) - price
    elif option_type == "put":
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)
        upper = K * np.exp(-r * T)
        f = lambda sig: bs_put_price(S, K, T, r, sig) - price
    else:
        raise ValueError(f"option_type in {{call, put}}, got {option_type}")

    if price < intrinsic - 1e-10 or price > upper + 1e-10:
        return np.nan
    try:
        return _optimize.brentq(f, 1e-6, 5.0, xtol=1e-8, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


# ------------------ Fractional Adams PECE vettorizzato ------------------

def fractional_adams_pece_vec(alpha, A, B, C, T, n_steps):
    """
    Risolve in parallelo N_u FDE scalari:
        D^alpha h_j(t) = A_j + B_j h_j(t) + C h_j(t)^2,    h_j(0) = 0

    Schema Diethelm-Ford-Freed (2002):
      Predictor (Adams-Bashforth frazionale, f costante a sinistra):
          y^P_{k+1} = (h^a / (a Gamma(a))) *
                      sum_{j=0..k} [(k+1-j)^a - (k-j)^a] f(t_j, y_j)
      Corrector (Adams-Moulton, f lineare):
          y_{k+1} = (h^a / (a(a+1)Gamma(a))) *
                    [sum_{j=0..k} A_{j,k+1} f(t_j, y_j) +
                     A_{k+1,k+1} f(t_{k+1}, y^P_{k+1})]
          A_{0,k+1} = k^(a+1) - (k-a)(k+1)^a
          A_{j,k+1} = (k-j+2)^(a+1) + (k-j)^(a+1) - 2(k-j+1)^(a+1)  (1 <= j <= k)
          A_{k+1,k+1} = 1

    Condizione iniziale h(0)=0, quindi f(0, 0) = A.

    Parametri
    ---------
    alpha : float in (0, 1]
    A, B  : array complex, shape (N_u,)
    C     : scalare complex (indipendente da u)
    T     : float, orizzonte
    n_steps : int

    Restituisce
    -----------
    y : array complex, shape (n_steps+1, N_u), con y[0] = 0.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha deve essere in (0, 1], got {alpha}")

    N_u = A.shape[0]
    h = T / n_steps
    h_a = h ** alpha
    inv_g = 1.0 / gamma_fn(alpha)
    factor_b = h_a / alpha * inv_g
    factor_a = h_a / (alpha * (alpha + 1.0)) * inv_g

    y = np.zeros((n_steps + 1, N_u), dtype=np.complex128)
    f_hist = np.zeros((n_steps + 1, N_u), dtype=np.complex128)
    f_hist[0] = A  # f(0, h=0) = A + B*0 + C*0 = A
    blown = np.zeros(N_u, dtype=bool)  # traccia u con overflow

    for k in range(n_steps):
        j_arr = np.arange(k + 1)
        b_w = ((k + 1 - j_arr).astype(float)) ** alpha \
            - ((k - j_arr).astype(float)) ** alpha
        y_pred = factor_b * (b_w @ f_hist[:k + 1])

        a_w = np.empty(k + 2)
        a_w[0] = k ** (alpha + 1.0) - (k - alpha) * (k + 1.0) ** alpha
        if k >= 1:
            j_mid = np.arange(1, k + 1)
            m = (k - j_mid + 1).astype(float)
            a_w[1:k + 1] = (m + 1.0) ** (alpha + 1.0) \
                + (m - 1.0) ** (alpha + 1.0) \
                - 2.0 * m ** (alpha + 1.0)
        a_w[k + 1] = 1.0

        f_pred = A + B * y_pred + C * y_pred * y_pred
        y_new = factor_a * (a_w[:k + 1] @ f_hist[:k + 1]
                            + a_w[k + 1] * f_pred)
        # Protezione overflow: se |h| esplode, marchia blown e freeze a 0.
        # char function per questi u risultera' posta a 0 nel post-processing.
        mag = np.abs(y_new)
        newly_blown = mag > 1e3
        blown |= newly_blown
        y_new = np.where(blown, 0.0, y_new)
        y[k + 1] = y_new
        f_hist[k + 1] = np.where(blown, 0.0, A + B * y_new + C * y_new * y_new)

    return y, blown


# ------------------ Characteristic function di rough Heston ------------------

def rheston_char_function(u_arr, params, T, n_steps=150):
    """
    phi(u; T) = E[exp(iu (X_T - rT))] per X_T = log(S_T/S_0) sotto Q.

    Il drift risk-free r va aggiunto esternamente:
        phi_with_r(u) = exp(iu r T) * phi(u)

    Parametri
    ---------
    u_arr : array di reali, shape (N_u,) -- punti di valutazione in Fourier
    params : RoughHestonParams
    T : float
    n_steps : int -- risoluzione temporale FDE

    Restituisce
    -----------
    phi : array complex, shape (N_u,)
    """
    alpha = params.alpha  # = H + 0.5
    u_arr = np.asarray(u_arr, dtype=float)
    iu = 1j * u_arr
    A_vec = 0.5 * (-u_arr * u_arr - iu)
    B_vec = iu * params.rho * params.xi - params.kappa
    C_val = 0.5 * params.xi * params.xi + 0j

    h = fractional_adams_pece_vec(alpha, A_vec, B_vec, C_val, T, n_steps)
    if isinstance(h, tuple):
        h, blown = h
    else:
        blown = np.zeros(A_vec.shape[0], dtype=bool)
    t_grid = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps

    # psi_2: integral trapezoidale standard
    psi_2 = np.trapezoid(h, t_grid, axis=0)

    # psi_1 = I^(1-alpha) h(T): product-trapezoidal (h lineare su ogni intervallo)
    # Su [t_k, t_{k+1}] con B = T-t_k, A = T-t_{k+1}:
    #     J_1 = integral (T-s)^(b-1)(t_{k+1}-s) ds  [coef di h_k]
    #     J_2 = integral (T-s)^(b-1)(s-t_k) ds      [coef di h_{k+1}]
    beta = 1.0 - alpha
    Bv = T - t_grid[:-1]
    Av = T - t_grid[1:]
    I_a = Bv ** beta - Av ** beta
    I_b = Bv ** (beta + 1.0) - Av ** (beta + 1.0)
    J_1 = I_b / (beta + 1.0) - Av * I_a / beta
    J_2 = Bv * I_a / beta - I_b / (beta + 1.0)
    w = np.zeros(n_steps + 1)
    w[:-1] += J_1 / dt
    w[1:] += J_2 / dt
    psi_1 = (w @ h) / gamma_fn(beta)

    log_phi = params.V0 * psi_1 + params.kappa * params.theta * psi_2
    # Proprieta' teorica: |phi(u)| <= 1, quindi Re(log phi) <= 0.
    # Se numericamente viola => overflow della Riccati, forza phi=0 per quel u.
    re_clipped = np.minimum(log_phi.real, 0.0)
    phi = np.exp(re_clipped + 1j * log_phi.imag)
    phi = np.where(np.isnan(phi) | np.isinf(phi), 0.0, phi)
    phi = np.where(blown, 0.0, phi)
    return phi


# ------------------ COS method multi-strike ------------------

def _cos_call_V_k(a, b, K, N_cos):
    """Coefficienti COS per payoff call K(e^x - 1)^+, x = log(S_T/K)."""
    BA = b - a
    k = np.arange(N_cos, dtype=float)
    kpi_BA = k * np.pi / BA
    c, d = 0.0, b
    if d <= c:
        return np.zeros(N_cos)
    cos_d = np.cos(kpi_BA * (d - a)); cos_c = np.cos(kpi_BA * (c - a))
    sin_d = np.sin(kpi_BA * (d - a)); sin_c = np.sin(kpi_BA * (c - a))
    exp_d = np.exp(d);                exp_c = np.exp(c)

    chi = (cos_d * exp_d - cos_c * exp_c
           + kpi_BA * (sin_d * exp_d - sin_c * exp_c)) \
        / (1.0 + kpi_BA ** 2)

    psi = np.empty(N_cos)
    psi[0] = d - c
    if N_cos > 1:
        psi[1:] = (BA / (k[1:] * np.pi)) * (
            np.sin(kpi_BA[1:] * (d - a)) - np.sin(kpi_BA[1:] * (c - a))
        )
    return (2.0 * K / BA) * (chi - psi)


def cos_call_prices(S0, strikes, T, r, params, N_cos=160, L=12.0,
                    n_steps_fde=150):
    """
    Prezza call europee su vari strike (fissato T) via COS method.

    Range di troncamento:
        c1 = log(S0/K) + (r - V0/2) T   (media BS)
        c2 = V0 T                        (var BS)
        [a, b] = [c1 - L sqrt(c2), c1 + L sqrt(c2)]

    Il char function phi_Y e' calcolato UNA volta per tutti gli u_k
    (indipendente da K), quindi per ogni K si applica solo lo shift di
    forward e i coefficienti V_k(K).

    Restituisce
    -----------
    prices : array shape (n_strikes,)
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    F = S0 * np.exp(r * T)

    # Range comune: usa log(S_0/K_ref) e il massimo K per coprire tutti gli strike
    # Semplifichiamo calcolando per-strike (l'overhead di phi_Y e' pagato solo 1x)
    # MA poiche' u_k = k*pi/(b-a) varia con K via (b-a), il phi_Y va riusato
    # solo se (b-a) e' costante. Usiamo (b-a) = 2*L*sqrt(V0*T) costante per
    # ogni strike (cambia solo il centro c1). Cosi' u_k e' lo stesso per tutti.
    BA = 2.0 * L * np.sqrt(params.V0 * T)
    k = np.arange(N_cos, dtype=float)
    u_k = k * np.pi / BA

    # Calcola phi_Y(u_k; T) UNA sola volta
    phi_Y = rheston_char_function(u_k, params, T, n_steps=n_steps_fde)

    weights = np.ones(N_cos)
    weights[0] = 0.5

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        c1 = np.log(S0 / K) + (r - 0.5 * params.V0) * T
        a = c1 - L * np.sqrt(params.V0 * T)
        b = c1 + L * np.sqrt(params.V0 * T)
        phi_x = phi_Y * np.exp(1j * u_k * np.log(F / K))
        V_k = _cos_call_V_k(a, b, K, N_cos)
        integrand = np.real(phi_x * np.exp(-1j * u_k * a)) * V_k
        prices[i] = np.exp(-r * T) * np.sum(weights * integrand)

    return prices


def rheston_iv_surface(params, S0, r, strikes, maturities, N_cos=160, L=12.0,
                       n_steps_fde=150):
    """
    Restituisce dict[T] -> dict(K -> IV) usando COS + inversione BS.
    Per ogni (K, T), prezza la call, poi inverte BS.
    """
    surface = {}
    F_fn = lambda T: S0 * np.exp(r * T)
    for T in maturities:
        prices = cos_call_prices(S0, strikes, T, r, params,
                                 N_cos=N_cos, L=L, n_steps_fde=n_steps_fde)
        surface[T] = {}
        for K, px in zip(strikes, prices):
            iv = bs_implied_vol(px, S0, K, T, r, "call")
            surface[T][K] = iv
    return surface


# ------------------ Calibrazione ------------------

def rheston_calibrate(
    market_data,
    S0,
    r=0.0,
    param_bounds=None,
    N_cos=128,
    L=12.0,
    n_steps_fde=100,
    de_maxiter=40,
    de_popsize=15,
    de_tol=1e-4,
    de_seed=42,
    verbose=True,
):
    """
    Calibra parametri rough Heston a una IV surface di mercato.

    Parametri
    ---------
    market_data : list of tuples (K, T, IV) oppure (K, T, IV, weight)
        IV sono le implied vol BS OTM di mercato.
    S0 : float, spot
    r : float, tasso risk-free (0 per ETH perp; funding rate per cripto)
    param_bounds : list di (lo, hi) per [V0, kappa, theta, xi, rho, H]
        Default conservativi.
    N_cos, L, n_steps_fde : risoluzione pricer
    de_* : parametri differential evolution

    Restituisce
    -----------
    dict con chiavi: params, rmse_iv, n_obs, optimizer_result
    """
    # Parsing market data
    data = []
    for row in market_data:
        if len(row) == 3:
            K, T, iv = row; w = 1.0
        elif len(row) == 4:
            K, T, iv, w = row
        else:
            raise ValueError(f"market_data row: {row}")
        if np.isfinite(iv) and iv > 0:
            data.append((float(K), float(T), float(iv), float(w)))
    data = sorted(data, key=lambda x: (x[1], x[0]))
    if not data:
        raise ValueError("market_data vuoto / invalido")

    maturities = sorted(set(row[1] for row in data))
    by_T = {T: [r for r in data if r[1] == T] for T in maturities}

    # Pre-compute prezzi di mercato OTM per ogni (K, T)
    market_otm = {}  # T -> list of (K, iv, px_otm, option_type, weight)
    for T in maturities:
        F = S0 * np.exp(r * T)
        rows = []
        for K, T_, iv, w in by_T[T]:
            if K >= F:
                px = bs_call_price(S0, K, T, r, iv)
                otype = "call"
            else:
                px = bs_put_price(S0, K, T, r, iv)
                otype = "put"
            rows.append((K, iv, px, otype, w))
        market_otm[T] = rows

    if param_bounds is None:
        param_bounds = [
            (0.001, 4.0),     # V0
            (0.01, 15.0),     # kappa
            (0.001, 4.0),     # theta
            (0.05, 3.5),      # xi
            (-0.99, 0.95),    # rho
            (0.02, 0.49),     # H  (rough: strictly < 0.5)
        ]

    eval_count = [0]

    def loss(theta_vec):
        V0, kappa, theta_v, xi, rho, H = theta_vec
        try:
            params = RoughHestonParams(V0=V0, kappa=kappa, theta=theta_v,
                                       xi=xi, rho=rho, H=H)
            params.validate()
        except (ValueError, AssertionError):
            return 1e6

        total_sq = 0.0
        total_w = 0.0
        try:
            for T in maturities:
                rows = market_otm[T]
                F = S0 * np.exp(r * T)
                strikes = np.array([r_[0] for r_ in rows])
                # Prezzi call via COS
                model_call = cos_call_prices(S0, strikes, T, r, params,
                                             N_cos=N_cos, L=L,
                                             n_steps_fde=n_steps_fde)
                for i, (K, iv_mkt, _px_mkt, otype, w) in enumerate(rows):
                    px_call = model_call[i]
                    if otype == "put":
                        # parita'
                        px_model = px_call + K * np.exp(-r * T) - S0
                    else:
                        px_model = px_call
                    iv_model = bs_implied_vol(px_model, S0, K, T, r, otype)
                    if not np.isfinite(iv_model):
                        total_sq += w * 1.0  # penalty
                    else:
                        total_sq += w * (iv_model - iv_mkt) ** 2
                    total_w += w
        except Exception:
            return 1e6

        if total_w == 0.0:
            return 1e6
        rmse = np.sqrt(total_sq / total_w)
        eval_count[0] += 1
        if verbose and eval_count[0] % 25 == 0:
            print(f"    [eval {eval_count[0]:4d}]  RMSE_IV = {rmse:.5f}  "
                  f"V0={V0:.3f} k={kappa:.2f} th={theta_v:.3f} "
                  f"xi={xi:.2f} rho={rho:+.2f} H={H:.3f}")
        return rmse

    if verbose:
        print(f"Calibrazione Rough Heston: {len(data)} punti IV, "
              f"{len(maturities)} maturities")
        print(f"Strategia: differential evolution, popsize={de_popsize}, "
              f"maxiter={de_maxiter}")

    result = _optimize.differential_evolution(
        loss, param_bounds,
        maxiter=de_maxiter, popsize=de_popsize, tol=de_tol,
        polish=True, seed=de_seed, workers=1, updating="deferred",
        disp=False,
    )

    calibrated = RoughHestonParams(
        V0=result.x[0], kappa=result.x[1], theta=result.x[2],
        xi=result.x[3], rho=result.x[4], H=result.x[5],
    )
    return {
        "params": calibrated,
        "rmse_iv": float(result.fun),
        "n_obs": len(data),
        "n_evals": eval_count[0],
        "optimizer_result": result,
    }


if __name__ == "__main__":
    print("Import module; vedi run_example.py e calibration_demo.py per l'uso.")