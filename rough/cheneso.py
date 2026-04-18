"""
Fetch IV surface ETH da Deribit (REST pubblica, no auth) e calibra rough Heston.

Output dei dati di mercato nel formato atteso da rheston_calibrate:
    market_data = [(K, T_years, IV_decimal), ...]

Note sulla convenzione:
- Deribit calcola mark_iv con r=0 e usa l'index come spot, quindi si calibra
  coerentemente con r=0.
- Tengo solo OTM (call per K>S0, put per K<S0): piu' liquide, meno rumore.
"""

import time
import requests
import numpy as np
from rough.prova import rheston_calibrate

BASE = "https://www.deribit.com/api/v2"


def fetch_index(currency="ETH"):
    r = requests.get(f"{BASE}/public/get_index_price",
                     params={"index_name": f"{currency.lower()}_usd"}, timeout=10)
    r.raise_for_status()
    return float(r.json()["result"]["index_price"])


def fetch_iv_surface(currency="ETH",
                     moneyness_range=(0.80, 1.20),
                     T_range_days=(7, 120)):
    """
    Restituisce (S0, market_data).

    Filtri:
    - moneyness K/S0 in moneyness_range
    - time to maturity in T_range_days
    - solo OTM
    - scarta punti senza mark_iv valido
    """
    S0 = fetch_index(currency)

    # 1) metadati strumenti (strike, expiry, tipo)
    r1 = requests.get(f"{BASE}/public/get_instruments",
                      params={"currency": currency, "kind": "option",
                              "expired": "false"}, timeout=15)
    r1.raise_for_status()
    instruments = {i["instrument_name"]: i for i in r1.json()["result"]}

    # 2) mark_iv in un colpo solo per tutta la currency
    r2 = requests.get(f"{BASE}/public/get_book_summary_by_currency",
                      params={"currency": currency, "kind": "option"}, timeout=15)
    r2.raise_for_status()
    summaries = r2.json()["result"]

    now_ms = int(time.time() * 1000)
    T_min = T_range_days[0] / 365.25
    T_max = T_range_days[1] / 365.25

    market_data = []
    for s in summaries:
        name = s["instrument_name"]
        inst = instruments.get(name)
        if inst is None:
            continue

        K = float(inst["strike"])
        T = (inst["expiration_timestamp"] - now_ms) / (365.25 * 86400 * 1000)
        if not (T_min <= T <= T_max):
            continue
        mny = K / S0
        if not (moneyness_range[0] <= mny <= moneyness_range[1]):
            continue

        is_call = inst["option_type"] == "call"
        # OTM only
        if is_call and K < S0:
            continue
        if (not is_call) and K > S0:
            continue

        mark_iv = s.get("mark_iv")
        if mark_iv is None or mark_iv <= 0:
            continue

        market_data.append((K, T, mark_iv / 100.0))

    return S0, market_data


def summarize_surface(S0, market_data):
    from collections import defaultdict
    by_T = defaultdict(list)
    for K, T, iv in market_data:
        by_T[int(round(T * 365))].append((K, iv))
    print(f"Spot: ${S0:,.2f}")
    print(f"Punti totali: {len(market_data)}")
    print(f"{'T(d)':>5} {'n':>4}  {'K_min':>8} {'K_max':>8}  "
          f"{'IV_min':>7} {'IV_max':>7}")
    for T_d in sorted(by_T):
        rows = by_T[T_d]
        Ks = [r[0] for r in rows]
        ivs = [r[1] for r in rows]
        print(f"{T_d:>5d} {len(rows):>4d}  {min(Ks):>8.0f} {max(Ks):>8.0f}  "
              f"{min(ivs):>7.3f} {max(ivs):>7.3f}")


if __name__ == "__main__":
    print("=" * 70)
    print("FETCH IV SURFACE ETH da Deribit")
    print("=" * 70)
    S0, market_data = fetch_iv_surface(
        currency="ETH",
        moneyness_range=(0.80, 1.20),
        T_range_days=(7, 90),
    )
    summarize_surface(S0, market_data)

    if len(market_data) < 20:
        print("\nTroppo pochi punti, verifica filtri / connessione.")
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("CALIBRAZIONE Rough Heston")
    print("=" * 70)
    result = rheston_calibrate(
        market_data, S0, r=0.0,
        N_cos=96, L=10.0, n_steps_fde=80,
        de_maxiter=40, de_popsize=15, de_tol=1e-4,
        verbose=True,
    )
    p = result["params"]
    print(f"\nRMSE IV in-sample: {result['rmse_iv']:.5f} "
          f"({result['rmse_iv']*100:.2f} vol points)")
    print(f"n_obs = {result['n_obs']}, n_evals = {result['n_evals']}")
    print(f"\nParametri calibrati:")
    print(f"  V0    = {p.V0:.4f}    (sqrt = {np.sqrt(p.V0):.3f} = vol spot)")
    print(f"  kappa = {p.kappa:.3f}")
    print(f"  theta = {p.theta:.4f}   (sqrt = {np.sqrt(p.theta):.3f} = vol LT)")
    print(f"  xi    = {p.xi:.3f}")
    print(f"  rho   = {p.rho:+.3f}    (skew: negativo => put expensive)")
    print(f"  H     = {p.H:.3f}     (roughness: <0.5 = rough)")