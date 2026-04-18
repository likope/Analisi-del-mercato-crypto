import requests
import polars as pl
import time

deribit_url = "https://www.deribit.com/api/v2/public"

def deribit_get(endpoint: str, params: dict) ->dir:
    """
    Funzione per fare richieste al url di deribit.
    """
    response = requests.get(f"{deribit_url}/{endpoint}", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"Deribit api: {data["error"]}")
    return data["result"]


def get_spot(currency: str) ->float:
    spot = deribit_get("get_index_price", {"index_name": f"{currency.lower()}_usd"})["index_price"]
    return spot
     

def fetch_option(expiry: str, currency: str) ->pl.DataFrame:

    params = {"currency": currency.upper(), "kind": "option", "expired": "false"}
    instruments = deribit_get("get_instruments", params)

    # 2. Filtra per la scadenza che ti interessa
    expiry = expiry.upper()
    filtered = [i for i in instruments
                if i["instrument_name"].split("-")[1] == expiry]

    # 3. Per ogni strumento, prendi il ticker
    options = []
    for inst in filtered:
        name = inst["instrument_name"]
        try:
            params={"instrument_name": name}
            ticker = deribit_get("ticker", params)
    
            options.append({
                "instrument": name,
                "strike": inst["strike"],
                "type": inst["option_type"],        # "call" o "put"
                "gamma": ticker["greeks"]["gamma"],
                "delta": ticker["greeks"]["delta"],
                "vega": ticker["greeks"]["vega"],
                "iv": ticker["mark_iv"],
                "oi": ticker["open_interest"],
                "volume": ticker["stats"]["volume"],
                "bid": ticker["best_bid_price"],
                "ask": ticker["best_ask_price"],
                "mark": ticker["mark_price"],
            })
        except Exception as e:
            print(f"Skip {name}: {e}")
            continue
        time.sleep(0.05)
        if not options:
            raise RuntimeError("Nessuno strumento fetchato con successo")

    df = pl.DataFrame(options)
    return df