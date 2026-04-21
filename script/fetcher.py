import httpx

deribit_url = "https://www.deribit.com/api/v2/public"
_DERIBIT_CONCURRENCY = 10


def _with_retry(func, attempts: int = 3, delay: float = 2.0):
    """
    Esegue il retry
    """
    for i in range(attempts):
        try:
            return func()
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"  Tentativo {i + 1}/{attempts} fallito ({e}), riprovo tra {delay:.0f}s...")
            time.sleep(delay)


def deribit_get(endpoint: str, params: dict) -> dict:
    """
    Esegue la chiamata api a deribit
    """
    def call():
        """
        Definisce la chiamata
        """
        response = httpx.get(f"{deribit_url}/{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Deribit api: {data['error']}")
        return data["result"]
    return _with_retry(call)


def get_binance_spot(symbol: str = "ETHUSDT") -> float:
    """
    Esegue la chiamata api a binance per lo spot
    """
    def call():
        """
        Definisce la chiamata
        """
        r = httpx.get("https://api.binance.com/api/v3/ticker/price",
                      params={"symbol": symbol}, timeout=15)
        r.raise_for_status()
        return float(r.json()["price"])
    return _with_retry(call)


def fetch_cvd_spot(symbol: str = "ETHUSDT", interval: str = "1m", limit: int = 100) -> pl.DataFrame:
    """
    Esegue la chiamata api a binance per il cvd spot
    """
    def call():
        """
        Definisce la chiamata
        """
        r = httpx.get("https://api.binance.com/api/v3/klines",
                      params={"symbol": symbol, "interval": interval, "limit": limit},
                      timeout=15)
        r.raise_for_status()
        return r.json()

    klines = _with_retry(call)
    rows = []
    for k in klines:
        vol = float(k[5])
        buy_vol = float(k[9])
        rows.append({
            "open_time": int(k[0]),
            "volume": vol,
            "buy_volume": buy_vol,
            "sell_volume": vol - buy_vol,
            "delta": buy_vol - (vol - buy_vol),
        })
    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("delta").cum_sum().alias("cvd"))


    async def _fetch_ticker_async(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                               name: str, strike: float, option_type: str) -> dict | None:
    async with sem:
        for attempt in range(3):
            try:
                r = await client.get(f"{deribit_url}/ticker",
                                     params={"instrument_name": name}, timeout=10)
                r.raise_for_status()
                data = r.json()
                if "error" in data:
                    raise RuntimeError(f"Deribit: {data['error']}")
                ticker = data["result"]
                return {
                    "instrument": name,
                    "strike": strike,
                    "type": option_type,
                    "gamma": ticker["greeks"]["gamma"],
                    "delta": ticker["greeks"]["delta"],
                    "vega": ticker["greeks"]["vega"],
                    "iv": ticker["mark_iv"],
                    "oi": ticker["open_interest"],
                    "volume": ticker["stats"]["volume"],
                    "bid": ticker["best_bid_price"],
                    "ask": ticker["best_ask_price"],
                    "mark": ticker["mark_price"],
                }
            except Exception as e:
                if attempt == 2:
                    print(f"Skip {name}: {e}")
                    return None
                await asyncio.sleep(2.0)
    return None


async def _fetch_option_async(expiry: str, currency: str) -> pl.DataFrame:
    """
    Scarica 10 opzioni alla volta in modo asincrono
    """
    instruments = deribit_get("get_instruments",
                              {"currency": currency.upper(), "kind": "option", "expired": "false"})
    expiry = expiry.upper()
    filtered = [i for i in instruments if i["instrument_name"].split("-")[1] == expiry]

    sem = asyncio.Semaphore(_DERIBIT_CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_ticker_async(client, sem,
                                inst["instrument_name"],
                                inst["strike"],
                                inst["option_type"])
            for inst in filtered
        ]
        results = await asyncio.gather(*tasks)

    options = [r for r in results if r is not None]
    if not options:
        raise RuntimeError("Nessuno strumento fetchato con successo")
    return pl.DataFrame(options)


def fetch_option(expiry: str, currency: str) -> pl.DataFrame:
    """
    Aspetta che il download di opzioni sia finito
    """
    return asyncio.run(_fetch_option_async(expiry, currency))
