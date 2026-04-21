import re
from analisi import ciclo
import time
from plot import plot, plot_oi_profile, plot_cvd, plot_iv_analysis

print("""Software per l'analisi di mercato.
Supporta ETH e BTC su Deribit + Binance spot.
Segnali: Netgex, IV Skew 25d, ATM IV, Gamma Walls, OI, Put/Call ratio, CVD spot.
Premere Ctrl+C per interrompere.""")

# --- Input con validazione ---
while True:
    currency = input("\nScegli il simbolo da analizzare: ETH, BTC\n").upper().strip()
    if currency in ("ETH", "BTC"):
        break
    print("Simbolo non valido. Inserisci ETH o BTC.")

while True:
    expiry = input("\nInserisci la data dell'opzione (es. 27DEC24)\n").upper().strip()
    if re.match(r"^\d{1,2}[A-Z]{3}\d{2,4}$", expiry):
        break
    print("Formato non valido. Esempio: 27DEC24 o 27DEC2024")

# --- Parametri CVD (opzionali) ---
_CVD_INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h")
print(f"\nParametri CVD — invio per usare i default (interval=1m, candles=100)")
raw_interval = input(f"Intervallo candele {_CVD_INTERVALS} [1m]: ").strip() or "1m"
cvd_interval = raw_interval if raw_interval in _CVD_INTERVALS else "1m"
try:
    cvd_limit = int(input("Numero candele (10-500) [100]: ").strip() or "100")
    cvd_limit = max(10, min(500, cvd_limit))
except ValueError:
    cvd_limit = 100
print(f"CVD: interval={cvd_interval}, candles={cvd_limit}")

currency_binance = f"{currency}USDT"

spot_accumulo = []
oi_totale = []
oi_calls_totale = []
oi_puts_totale = []
oi_history = []
cvd_history = []
timestamps = []
iv_skew_history = []
atm_iv_history = []

try:
    while True:
        (spot_accumulo, call_walls, put_walls, spot, Netgex, oi_totale_current,
         oi_totale, oi_calls_totale, oi_puts_totale, oi_history, cvd_history,
         timestamps, iv_skew_history, atm_iv_history) = ciclo(
            currency, expiry, spot_accumulo, oi_totale, currency_binance,
            oi_calls_totale, oi_puts_totale, oi_history, cvd_history, timestamps,
            iv_skew_history, atm_iv_history, cvd_interval, cvd_limit
        )
        plot(spot_accumulo, call_walls, put_walls, oi_totale, oi_calls_totale, oi_puts_totale, timestamps)
        plot_oi_profile(oi_history)
        plot_cvd(cvd_history)
        plot_iv_analysis(iv_skew_history, atm_iv_history, timestamps)
        time.sleep(60)
except KeyboardInterrupt:
    print("\nSessione terminata.")
except Exception as e:
    print(f"\nErrore imprevisto: {e}")
    raise