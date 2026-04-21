def netgex(df, spot: float):
    """
    Calcola il netgex e il oi_totale
    """
    df = df.with_columns(
        pl.when(pl.col("type") == "put")
        .then(pl.lit(-1))
        .otherwise(pl.lit(1))
        .alias("segno")
    ).with_columns(
        (pl.col("oi") * pl.col("gamma") * pl.col("segno") * 0.01 * spot**2).alias("Net gex")
    )
    Net_gex = df["Net gex"].sum()
    oi_totale = df["oi"].sum()
    return df, Net_gex, oi_totale


def walls(df, spot):
    """
    Individua i walls e analizza gli oi
    """
    df2 = df.clone()
    df = (
        df.group_by("strike")
        .agg([
            pl.col("Net gex").filter(pl.col("type") == "call").sum().alias("gex calls"),
            pl.col("Net gex").filter(pl.col("type") == "put").sum().alias("gex puts"),
            pl.col("oi").filter(pl.col("type") == "call").sum().alias("oi_calls"),
            pl.col("oi").filter(pl.col("type") == "put").sum().alias("oi_puts"),
            (pl.col("oi") * pl.col("gamma")).filter(pl.col("type") == "call").sum().alias("weight call"),
            (pl.col("oi") * pl.col("gamma")).filter(pl.col("type") == "put").sum().alias("weight put"),
        ])
    )
    oi_calls = df["oi_calls"].sum()
    oi_puts = df["oi_puts"].sum()

    df_oi = (
        df2.group_by("strike")
        .agg([
            pl.col("oi").filter(pl.col("type") == "call").sum().alias("oi_calls"),
            pl.col("oi").filter(pl.col("type") == "put").sum().alias("oi_puts"),
        ])
        .with_columns((pl.col("oi_calls") + pl.col("oi_puts")).alias("oi_total"))
        .sort("oi_total", descending=True)
    )

    call_walls = df.filter(pl.col("strike") > spot).sort("weight call", descending=True).head(3)
    put_walls = df.filter(pl.col("strike") < spot).sort("weight put", descending=True).head(3)
    return df, call_walls, put_walls, oi_calls, oi_puts, df_oi


def iv_skew(df):
    """25-delta skew: differenza IV tra OTM puts e OTM calls come misura del sentiment di mercato."""
    puts_25d = df.filter(
        (pl.col("type") == "put") &
        (pl.col("delta") >= -0.30) &
        (pl.col("delta") <= -0.20)
    )
    calls_25d = df.filter(
        (pl.col("type") == "call") &
        (pl.col("delta") >= 0.20) &
        (pl.col("delta") <= 0.30)
    )
    if len(puts_25d) == 0 or len(calls_25d) == 0:
        return None, None, None, "dati insufficienti per il 25-delta"

    iv_p = puts_25d["iv"].mean()
    iv_c = calls_25d["iv"].mean()
    skew = iv_p - iv_c

    if skew > 3:
        signal = "put premium (mercato difensivo)"
    elif skew < -3:
        signal = "call premium (mercato esuberante)"
    else:
        signal = "neutro"

    return round(skew, 2), round(iv_p, 2), round(iv_c, 2), signal


def _get_atm_iv(df, spot):
    """IV media call+put allo strike più vicino allo spot."""
    atm_strike = (
        df.with_columns((pl.col("strike") - spot).abs().alias("dist"))
        .sort("dist")["strike"][0]
    )
    atm_iv = df.filter(pl.col("strike") == atm_strike)["iv"].mean()
    return round(atm_iv, 2)


def cvd_spot(symbol: str = "ETHUSDT", interval: str = "1m", limit: int = 100):
    df = fetch_cvd_spot(symbol, interval, limit)
    cvd_current = df["cvd"][-1]
    signal = "BUY pressure" if cvd_current > 0 else "SELL pressure"
    return df, cvd_current, signal


def ciclo(currency, expiry, spot_accumulo, oi_totale, currency_binance,
          oi_calls_totale, oi_puts_totale, oi_history, cvd_history, timestamps,
          iv_skew_history, atm_iv_history, cvd_interval="1m", cvd_limit=100):
    cycle_num = len(spot_accumulo) + 1
    print(f"\n--- Ciclo {cycle_num} [{datetime.now().strftime('%H:%M:%S')}] ---")

    spot = get_binance_spot(currency_binance)
    spot_accumulo.append(spot)
    timestamps.append(datetime.now())
    print(f"Spot: {spot:.2f}")

    df = fetch_option(expiry, currency)
    df, Netgex, oi_totale_current = netgex(df, spot)

    if len(spot_accumulo) > 1:
        delta_spot = spot_accumulo[-1] - spot_accumulo[-2]
        if abs(delta_spot) > 2:
            print(f"  *** Movimento forte: delta_spot = {delta_spot:+.2f}")

    oi_totale.append(oi_totale_current)
    if len(oi_totale) > 1 and oi_totale[-2] > 0:
        delta_oi = oi_totale[-1] - oi_totale[-2]
        pct = abs(delta_oi) / oi_totale[-2]
        if pct > _OI_CHANGE_THRESHOLD_PCT:
            print(f"  Cambio OI: {delta_oi:+.0f} ({pct * 100:.2f}%)")

    gex_label = "short gamma" if Netgex > 0 else "long gamma"
    print(f"OI totale: {oi_totale_current:.0f}")
    print(f"Netgex: {Netgex:.2f} ({gex_label})")

    skew, iv_put, iv_call, skew_signal = iv_skew(df)
    iv_skew_history.append(skew)
    if skew is not None:
        print(f"IV Skew 25d: {skew:+.2f} (put {iv_put:.1f}% / call {iv_call:.1f}%) → {skew_signal}")
    else:
        print(f"IV Skew: {skew_signal}")

    atm_iv = _get_atm_iv(df, spot)
    atm_iv_history.append(atm_iv)
    print(f"ATM IV: {atm_iv:.1f}%")

    df, call_walls, put_walls, oi_calls, oi_puts, df_oi = walls(df, spot)
    oi_history.append(df_oi)
    oi_calls_totale.append(oi_calls)
    oi_puts_totale.append(oi_puts)

    put_call_ratio = oi_puts / oi_calls if oi_calls > 0 else None
    ratio_str = f"{put_call_ratio:.2f}" if put_call_ratio is not None else "N/A"
    print(f"Put/Call ratio: {ratio_str}")

    df_cvd, cvd_current, cvd_signal = cvd_spot(currency_binance, cvd_interval, cvd_limit)
    cvd_history.append(df_cvd)
    print(f"CVD spot: {cvd_current:.2f} ({cvd_signal})")

    print(f"Call walls: {call_walls['strike'].to_list()}")
    print(f"Put walls:  {put_walls['strike'].to_list()}")

    return (spot_accumulo, call_walls, put_walls, spot, Netgex, oi_totale_current,
            oi_totale, oi_calls_totale, oi_puts_totale, oi_history, cvd_history,
            timestamps, iv_skew_history, atm_iv_history)
