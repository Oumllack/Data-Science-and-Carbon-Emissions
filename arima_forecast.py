import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")

# Charger les données prétraitées
DATA_PATH = 'data/MER_T12_06.csv'
df = pd.read_csv(DATA_PATH)

# Prétraitement minimal (comme dans main.py)
df['YYYYMM'] = df['YYYYMM'].astype(str).str.zfill(6)
df['Year'] = df['YYYYMM'].str[:4].astype(int)
df['Month'] = df['YYYYMM'].str[4:6].astype(int)
df = df[df['Month'] <= 12]
df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])
seuil = df['Value'].quantile(0.99)
df = df[df['Value'] <= seuil]
df = df.dropna()

# Préparation de la série temporelle (moyenne mensuelle)
ts = df.groupby('Date')['Value'].mean().resample('MS').mean()

# Vérifier la stationnarité (test de Dickey Fuller)
adf = adfuller(ts.dropna())
print("Test de Dickey Fuller (H0 : non stationnaire) :")
print("p-valeur :", adf[1])
if adf[1] < 0.05:
    print("La série est stationnaire (p < 0.05).")
else:
    print("La série n'est pas stationnaire (p >= 0.05).")

# Entraîner auto-ARIMA (pour trouver les meilleurs ordres p, d, q et P, D, Q, m)
print("\nEntraînement du modèle ARIMA (cela peut prendre quelques minutes)...")
model_auto = auto_arima(ts, seasonal=True, m=12, trace=True, suppress_warnings=True, error_action="ignore", stepwise=True, maxiter=50, n_jobs=-1, random_state=42, information_criterion='aicc')
print("Meilleur modèle ARIMA trouvé :", model_auto.summary())

# Entraîner SARIMAX (statsmodels) avec les ordres trouvés par auto-ARIMA
order = model_auto.order
seasonal_order = model_auto.seasonal_order
print("\nOrdres choisis par auto-ARIMA :")
print("Ordre (p,d,q) :", order)
print("Ordre saisonnier (P,D,Q,m) :", seasonal_order)

model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False, maxiter=100, method='lbfgs')
print("\nRésumé du modèle SARIMAX :")
print(model_fit.summary())

# Prédire les 50 prochaines années (mois par mois)
forecast_steps = 50 * 12
forecast = model_fit.forecast(steps=forecast_steps, alpha=0.05)
forecast_ci = model_fit.get_forecast(steps=forecast_steps, alpha=0.05).conf_int()

# Créer un DataFrame pour les prédictions futures
future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
future_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast, 'yhat_lower': forecast_ci.iloc[:, 0], 'yhat_upper': forecast_ci.iloc[:, 1]})

# Sauvegarder les résultats
Path('outputs').mkdir(exist_ok=True)
future_df.to_csv('outputs/arima_predictions_futures.csv', index=False)

# Visualisation interactive
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Historique', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat'], mode='lines', name='Prédiction ARIMA', line=dict(color='green')))
fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat_upper'], mode='lines', name='Intervalle haut', line=dict(color='rgba(0,200,0,0.2)'), showlegend=False))
fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat_lower'], mode='lines', name='Intervalle bas', fill='tonexty', line=dict(color='rgba(0,200,0,0.2)'), showlegend=False))
fig.update_layout(title='Prédictions ARIMA pour les 50 prochaines années', xaxis_title='Date', yaxis_title='Émissions (Million Metric Tons of CO2)')
Path('figures').mkdir(exist_ok=True)
fig.write_html('figures/arima_predictions_futures.html')

print("\nPrédictions ARIMA sauvegardées dans outputs/arima_predictions_futures.csv")
print("Visualisation interactive : figures/arima_predictions_futures.html") 