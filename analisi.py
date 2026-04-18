import polars as pl
import numpy as np

def netgex(df, spot: float):
    df = df.with_columns(
        pl.when(pl.col("type") == "put")
        .then(pl.lit(-1))
        .otherwise(pl.lit(1))
        .alias("segno")
    ).with_columns(
        (pl.col("oi")*pl.col("gamma")*pl.col("segno")*0.01*spot**2).alias("Net gex")
    )
    Net_gex = df["Net gex"].sum()
    return df, Net_gex


def walls(df, spot):
    """
    Analisi degli oi, net gex e degli wall.
    """
    df = (
        df.group_by("strike")
        .agg([
            pl.col("Net gex").filter(pl.col("type") == "call").sum().alias("gex calls"),
            pl.col("Net gex").filter(pl.col("type") == "put").sum().alias("gex puts"),
            pl.col("oi").filter(pl.col("type") == "call").sum().alias("oi_calls"),
            pl.col("oi").filter(pl.col("type") == "put").sum().alias("oi_puts"),
            (pl.col("oi") * pl.col("gamma"))
              .filter(pl.col("type") == "call").sum()
              .alias("weight call"),
            (pl.col("oi") * pl.col("gamma"))
              .filter(pl.col("type") == "put").sum()
              .alias("weight put"),
        ])
    )
    call_walls = (
        df.filter(pl.col("strike") > spot)
        .sort("weight call", descending = True)
        .head(3)
    )

    put_walls = (
        df.filter(pl.col("strike") < spot)
        .sort("weight put", descending = True)
        .head(3)
    )
    return df, call_walls, put_walls


import numpy as np

def griglia_carr_madan(N: int, A: float) -> tuple:
    """
    Costruisce griglia di nodi in v (dominio Fourier) e k (log-strike).

    Args:
        N: numero di nodi (potenza di 2)
        A: upper bound dominio v

    Returns:
        v: array nodi in dominio Fourier
        k: array nodi log-strike (centrati su 0)
        eta: passo in v
        lambda_: passo in k
    """
    eta = A / N                          # passo in v
    lambda_ = 2 * np.pi / (N * eta)      # passo in k, da condizione Nyquist

    v = eta * np.arange(N)               # v[j] = eta*j, j=0,...,N-1
    k = -lambda_ * N / 2 + lambda_ * np.arange(N)   # k[l] centrato su 0

    return v, k, eta, lambda_


def cf_bs(y, sigma, T):
    """
    Characteristic function BS (detrendizzata).
    φ(y) = exp(-σ²/2 · iy T - σ²y²T/2)
    """
    phi = np.exp(-sigma**2 / 2 * 1j * y * T - sigma**2 * y**2 * T / 2)
    return phi


def zeta_T(v, r, T, sigma):
    # Fix v=0
    v = v.copy()
    v[0] = 1e-22
    
    # Pezzo 1: fase
    fase = np.exp(1j * v * r * T)
    
    # Pezzo 2: numeratore
    num = cf_bs(v - 1j, sigma, T) - 1
    
    # Pezzo 3: denominatore
    den = 1j * v * (1 + 1j * v)
    
    # Assemblaggio
    zeta = fase * (num / den)
    return zeta