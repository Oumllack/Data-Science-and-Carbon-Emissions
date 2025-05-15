import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Charger les données
DATA_PATH = 'data/MER_T12_06.csv'
df = pd.read_csv(DATA_PATH)
df['YYYYMM'] = df['YYYYMM'].astype(str).str.zfill(6)
df = df[df['YYYYMM'].str.match(r'^\d{6}$')]
df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m', errors='coerce')
df = df.dropna(subset=['Date'])
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])
seuil = df['Value'].quantile(0.99)
df = df[df['Value'] <= seuil]
ts = df.groupby('Date')['Value'].mean().resample('MS').mean()

# Test de stationnarité
adf = adfuller(ts.dropna())
d = 0
if adf[1] >= 0.05:
    d = 1
    ts = ts.diff().dropna()

# Optimisation des hyperparamètres ARIMA (grid search simple)
def arima_grid_search(ts, p_values, d, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for q in q_values:
            try:
                model = ARIMA(ts, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_score:
                    best_score, best_cfg = aic, (p, d, q)
            except:
                continue
    return best_cfg

p_values = range(0, 4)
q_values = range(0, 4)
best_order = arima_grid_search(ts, p_values, d, q_values)
print(f"Meilleur ordre ARIMA trouvé (AIC): {best_order}")

# Backtesting (rolling forecast sur les 10 dernières années)
test_size = 120  # 10 ans
train, test = ts[:-test_size], ts[-test_size:]
history = list(train)
predictions = []
for t in range(len(test)):
    model = ARIMA(history, order=best_order)
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(test.iloc[t])

mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
print(f"MSE (10 dernières années) : {mse:.2f}")
print(f"MAE (10 dernières années) : {mae:.2f}")

# Prédictions futures (50 ans)
model = ARIMA(ts, order=best_order)
model_fit = model.fit()
forecast_steps = 50 * 12
forecast = model_fit.forecast(steps=forecast_steps)
forecast_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()
future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
future_df = pd.DataFrame({
    'Date': future_dates,
    'Prédiction': forecast,
    'Intervalle_Bas': forecast_ci.iloc[:, 0],
    'Intervalle_Haut': forecast_ci.iloc[:, 1]
})
Path('outputs').mkdir(exist_ok=True)
future_df.to_csv('outputs/arima_optim_predictions.csv', index=False)

# Visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Historique', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test réel', line=dict(color='black', dash='dot')))
fig.add_trace(go.Scatter(x=test.index, y=predictions, mode='lines', name='Prédiction backtest', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Prédiction'], mode='lines', name='Prédiction future', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Intervalle_Haut'], mode='lines', name='Intervalle haut', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Intervalle_Bas'], mode='lines', name='Intervalle bas', fill='tonexty', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
fig.update_layout(
    title='Optimisation ARIMA, backtesting et prédictions futures (50 ans)',
    xaxis_title='Date',
    yaxis_title='Émissions (Million Metric Tons of CO2)',
    hovermode='x unified'
)
Path('figures').mkdir(exist_ok=True)
fig.write_html('figures/arima_optim_predictions.html')

print("\nPrédictions futures sauvegardées dans outputs/arima_optim_predictions.csv")
print("Visualisation interactive : figures/arima_optim_predictions.html")
print(f"\nMétriques sur les 10 dernières années :\nMSE = {mse:.2f}\nMAE = {mae:.2f}") 