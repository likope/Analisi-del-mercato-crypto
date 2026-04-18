"""
Main unificato: Deribit IV surface -> calibrazione Rough Heston -> simulazione
forward -> analisi barriere TP/SL e distribuzione.

Lancio:
    python main.py
(modifica i parametri in __main__ per cambiare horizon, leverage, direzione.)
"""

import numpy as np
from rough.cheneso import fetch_iv_surface, summarize_surface
from rough.prova import rheston_calibrate, run_pipeline


def run(
    currency: str = "ETH",
    # calibrazione
    moneyness_range: tuple = (0.80, 1.20),
    T_calib_days: tuple = (7, 90),
    # simulazione forward
    horizon_days: float = 1.0,
    direction: str = "long",      # "long" o "short"
    leverage: float = 10.0,
    fees_per_side: float = 0.0005,
    n_paths: int = 20_000,
    n_steps_per_day: int = 288,   # bars 5-min
    # barrier sweep
    tp_pcts: tuple = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05),
    sl_pcts: tuple = (0.005, 0.01, 0.015, 0.02, 0.03),
    # overrides performance/accuratezza calibrazione
    N_cos: int = 96,
    L_cos: float = 10.0,
    n_steps_fde: int = 80,
    de_maxiter: int = 40,
    de_popsize: int = 15,
):
    # ================================================================
    # STEP 1 - FETCH IV SURFACE
    # ================================================================
    print("=" * 74)
    print(f"[1/3] FETCH IV SURFACE {currency} da Deribit")
    print("=" * 74)
    S0, market_data = fetch_iv_surface(
        currency=currency,
        moneyness_range=moneyness_range,
        T_range_days=T_calib_days,
    )
    summarize_surface(S0, market_data)
    if len(market_data) < 20:
        raise RuntimeError(f"Solo {len(market_data)} punti IV: allenta i filtri.")

    # ================================================================
    # STEP 2 - CALIBRA Rough Heston
    # ================================================================
    print("\n" + "=" * 74)
    print("[2/3] CALIBRAZIONE Rough Heston")
    print("=" * 74)
    calib = rheston_calibrate(
        market_data, S0, r=0.0,
        N_cos=N_cos, L=L_cos, n_steps_fde=n_steps_fde,
        de_maxiter=de_maxiter, de_popsize=de_popsize, de_tol=1e-4,
        verbose=True,
    )
    p = calib["params"]
    print(f"\n  RMSE IV = {calib['rmse_iv']*100:.2f} vol points   "
          f"(n_obs={calib['n_obs']}, evals={calib['n_evals']})")
    print(f"  V0    = {p.V0:.4f}   (vol spot = {np.sqrt(p.V0):.3f})")
    print(f"  kappa = {p.kappa:.3f}")
    print(f"  theta = {p.theta:.4f}   (vol LT = {np.sqrt(p.theta):.3f})")
    print(f"  xi    = {p.xi:.3f}")
    print(f"  rho   = {p.rho:+.3f}")
    print(f"  H     = {p.H:.3f}   (roughness)")

    # ================================================================
    # STEP 3 - FORWARD MC + BARRIER + DENSITY
    # ================================================================
    print("\n" + "=" * 74)
    print(f"[3/3] SIMULAZIONE FORWARD  horizon={horizon_days}d  "
          f"direction={direction}  leverage={leverage}x")
    print("=" * 74)

    n_steps_sim = max(50, int(round(horizon_days * n_steps_per_day)))
    fwd = run_pipeline(
        S0=S0,
        horizon_days=horizon_days,
        params=p,
        drift_annual=0.0,
        n_steps=n_steps_sim,
        n_paths=n_paths,
        N_factors=30,
        tp_range=np.array(tp_pcts),
        sl_range=np.array(sl_pcts),
        direction=direction,
        leverage=leverage,
        fees_per_side=fees_per_side,
        seed=42,
        antithetic=True,
    )

    # --- kernel approximation sanity ---
    ka = fwd["kernel_approximation_check"]
    print(f"\nKernel check (Abi Jaber): max rel err = "
          f"{ka['max_rel_err']:.2e}, mean = {ka['mean_rel_err']:.2e}")

    # --- variance path summary ---
    vs = fwd["variance_stats"]
    print(f"\nVariance process:")
    print(f"  V mean path    = {vs['V_mean_path']:.4f}")
    print(f"  V terminal q05 = {vs['V_q05_terminal']:.4f}  "
          f"median = {vs['V_median_terminal']:.4f}  "
          f"q95 = {vs['V_q95_terminal']:.4f}")
    print(f"  frac(V<0) pre-trunc = {vs['frac_neg_V_pre_trunc']:.3%}")

    # --- forward return distribution ---
    d = fwd["density"]
    print(f"\nDistribuzione forward log-returns ({horizon_days}d):")
    print(f"  mean      = {d.mean:+.5f}")
    print(f"  std       = {d.std:.5f}   ({d.std*100:.2f}%)")
    print(f"  skewness  = {d.skew:+.3f}")
    print(f"  ex.kurt   = {d.kurt_excess:+.3f}")
    print(f"  VaR 95%   = {d.var_95:+.4f}   ES 95% = {d.es_95:+.4f}")
    print(f"  VaR 99%   = {d.var_99:+.4f}   ES 99% = {d.es_99:+.4f}")
    print(f"  Hill left = {d.hill_left:.2f}   Hill right = {d.hill_right:.2f}")
    print(f"  Student-t df = {d.student_t_df:.2f}   "
          f"BIC_t - BIC_N = {d.bic_student_t - d.bic_normal:+.1f} "
          f"(negativo => Student-t meglio)")

    # --- barrier sweep ---
    print(f"\nBarrier sweep (direction={direction}, lev={leverage}x, "
          f"fees={fees_per_side*1e4:.0f}bps per side):")
    sweep = fwd["barrier_sweep"]
    # sweep e' Polars DataFrame
    top = sorted(sweep, key=lambda r: r["expectancy_R"], reverse=True)[:8]
    print(f"{'TP%':>6} {'SL%':>6} {'p_tp':>6} {'p_sl':>6} {'E[PnL]':>8} "
        f"{'Sharpe':>7} {'PF':>6} {'E[R]':>7}")
    for r in top:
        print(f"{r['tp_pct']*100:>5.1f}% {r['sl_pct']*100:>5.1f}% "
            f"{r['p_tp']:>6.3f} {r['p_sl']:>6.3f} {r['expected_pnl']:>+8.4f} "
            f"{r['sharpe']:>+7.3f} {r['profit_factor']:>6.2f} {r['expectancy_R']:>+7.3f}")
    print(top)

    return {
        "S0": S0,
        "market_data": market_data,
        "calibration": calib,
        "forward": fwd,
    }


if __name__ == "__main__":
    # Configurazione trade: 24h long ETH @ 10x con barrier sweep
    results = run(
        currency="ETH",
        horizon_days=1.0,
        direction="long",
        leverage=10.0,
        fees_per_side=0.0005,
        n_paths=20_000,
    )

    # Qui hai `results["calibration"]["params"]` = params coerenti col mercato
    # adesso. Salvarli con timestamp in CSV per trackare H, rho nel tempo.
    #
    # Esempio (decommenta se vuoi loggare):
    #
    #import time, csv, os
    # p = results["calibration"]["params"]
    # row = {
    #     "ts": int(time.time()),
    #     "S0": results["S0"],
    #     "V0": p.V0, "kappa": p.kappa, "theta": p.theta,
    #     "xi": p.xi, "rho": p.rho, "H": p.H,
    #     "rmse_iv": results["calibration"]["rmse_iv"],
    # }
    # path = "rheston_history.csv"
    # write_header = not os.path.exists(path)
    # with open(path, "a", newline="") as f:
    #     w = csv.DictWriter(f, fieldnames=row.keys())
    #     if write_header:
    #         w.writeheader()
    #     w.writerow(row)