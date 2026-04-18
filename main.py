from fetcher import get_spot, fetch_option
from analisi import netgex, walls
import numpy as np
import matplotlib.pyplot as plt

t = 1
n = 0

print("""Software per l'analisi di mercato,
al momento supporta solo cripto,
è in grado di rilevare supporti e resistenze dinamiche""")
currency = str(input("Scegli il simbolo da analizzare: ETH, BTC\n"))
expiry = str(input("\nInserisci la data dell'opzione da analizzare\n"))
spot_accumulo = []

def ciclo(n, t):
      while n<t:
            print(f"Analisi numero: {n}")
            spot = get_spot(currency)
            spot_accumulo.append(spot)
            print(f"L'attuale prezzo spot è: {spot}")
            df = fetch_option(expiry, currency)
            df, Netgex = netgex(df, spot)
            print(f"Netgex = {Netgex}")
            df, call_walls, put_walls = walls(df, spot)
            print(df)
            print(f"""Call walls= {call_walls}""")
            print(f"""Put walls= {put_walls}""")
            print("Analisi completa")
            n = n+1
      return spot_accumulo
i = 0
media = []
while i<3:
      spot_accumulo = ciclo(n, t)
      mean = np.mean(spot_accumulo)
      media.append(mean)
      i=i+1

print(media)
if media[0]<media[i-1]:
      print("Trend bullish")
else:
      print("trend bearish")
print(spot_accumulo)

plt.plot(media, y, color=blue)
plt.show()