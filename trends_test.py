from datetime import datetime, timezone
from typing import Any, Dict, List
from decimal import Decimal
from Metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import pytz
import requests
import json

# 1. Graficar precios en grafico de lineas

# Funcion para convertir el timestamp a UTC
def timestamp_to_utc(timestamp_ms: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)

# Convertir a diccionario
def toDicto(kline: List[Any]) -> Dict[str, Any]:
    return {
        "open_time": timestamp_to_utc(kline[0]),
        "open_price": Decimal(kline[1]),
        "high_price": Decimal(kline[2]),
        "low_price": Decimal(kline[3]),
        "close_price": Decimal(kline[4]),
        "volume": Decimal(kline[5]),
        "close_time": timestamp_to_utc(kline[6]),
        "quote_asset_volume": Decimal(kline[7]),
        "number_of_trades": int(kline[8]),
        "taker_buy_base_asset_volume": Decimal(kline[9]),
        "taker_buy_quote_asset_volume": Decimal(kline[10]),
    }

# URL de la API
url = "https://api.binance.com/api/v3/"

# Convertir fecha a milisegundos (Fecha UTC)                # YYYY/MM/DD - HH:MM:SS
start_date = datetime(2026, 2, 4, 4, tzinfo=timezone.utc)   # 2O26/02/04 - 04:00:00
start_time = int(start_date.timestamp() * 1000)
end_date = datetime(2026, 2, 12, tzinfo=timezone.utc)       # 2026/02/12 - 00:00:00
end_time = int(end_date.timestamp() * 1000)

# Request
params = {
    "symbol": "BTCUSDT",
    "interval": "4h",
    "startTime": start_time,
    "endTime": end_time,
}
response = requests.get(url+"klines", params=params)
data = response.json()
# print(json.dumps(data[-1], indent=3)) # Ver la ultima  vela

# Verificar la fecha de la primera vela
timestamp_ms = data[0][0]
date_utc = timestamp_to_utc(timestamp_ms)
print("Fecha de apertura de la primera vela:", date_utc)

# Limpiar los datos
cleaned_data = [toDicto(kline) for kline in data]
print(json.dumps(cleaned_data[-1], indent=3, default=str)) # Ver la ultima  vela
X = [k["close_time"] for k in cleaned_data]
prices = [k["close_price"] for k in cleaned_data]

# Representar precios con grafico de lineas
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(X, prices)
# ax.set_xlabel("Fecha de cierre")
# ax.set_ylabel("Precio de cierre")
# ax.set_title("Bitcoin / TetherUS • 4h • Binance")
# ax.grid(True, alpha=0.2)
# plt.show()

# -------------------------------------------------------------------------------------- #
# 2. Encontrar tendencias en el precio

# Parámetros ajustables 
WINDOW = 10 # velas por ventana (ej. 12 * 4h = 48h)
MIN_WIN = 5
MIN_R2 = 0.3 # mínimo R^2 para considerar pendiente significativa 
PCT_SLOPE_MIN = 1.0 

i = WINDOW - 1

# x (variable independiente)
x = np.arange(WINDOW)
print("\nNuevo X")
print(x)

# y (variable independiente)
float_prices = np.array([float(p) for p in prices])
y = float_prices[i - WINDOW + 1 : i + 1]
print("\nNuevo Y")
print(y)

# Modelo de Regresion Lineal
a, b = np.polyfit(x, y, 1)
print("\nModelo:", np.poly1d(np.polyfit(x, y, 1)))
print("Pendiente:", a, "Intercepto:", b)
y_pred = a*x + b
print("\nPrediccion de Y:", y_pred)


# Media de los precios
y_mean = y.mean()
print("\nMedia de los precios:", y_mean)

# pct_slope (pendiente porcentual)
pct_slope = (a/y_mean) * 100
print("\nPendiente porcentual:", pct_slope)

res_sum = np.sum((y - y_pred) ** 2)
totals_sum = np.sum((y - y_mean) ** 2)
r2 = 1.0 - res_sum / totals_sum

print("\nCoef. de determinacion (r2):", r2)

def fit_model_metrics(y_win: np.ndarray):
    # Ajusta y = a x + b sobre el array y_win (contiguo). Devuelve (a, b, r2, pct_slope, mean_y).
    # x es 0..n-1.
    n = len(y_win)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, None

    x_local = np.arange(n, dtype=float)
    a_, b_ = np.polyfit(x, y_win, 1)
    y_hat_ = a*x_local + b

    mean_y = float(y_win.mean())
    ss_res = float(np.sum((y_win - y_hat_) ** 2))
    ss_total = float(np.sum((y_win - mean_y) ** 2))
    r2_ = 1.0 - ss_res / ss_total
    pct_slope_ = (a_ / mean_y) * 100.0 
    return float(a_), float(b_), float(r2_), float(pct_slope_), mean_y, y_hat_

def classify_trend(a: float, r2: float, r2_min: float, pct_slope: float, pct_slope_min: float) -> str:
    if np.isnan(r2) or r2 < r2_min or np.isnan(a):
        return "SIDE"
    if not np.isnan(pct_slope) and abs(pct_slope) < pct_slope_min:
        return "SIDE"
    return "UP" if a > 0 else "DOWN"

def best_subsegment_in_window(
        prices: np.ndarray,
        end_i: int,
        window: int,
        min_win: int,
        r2_min: float,
        pct_slope_min: float
        ) -> Metrics | None:
    
    assert min_win <= window, "min_len no puede ser mayor que window"

    start_window = max(0, end_i - window + 1)
    y_block = prices[start_window : end_i + 1]