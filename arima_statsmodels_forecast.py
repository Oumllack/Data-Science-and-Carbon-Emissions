import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# Charger les données
DATA_PATH = 'data/MER_T12_06.csv'
df = pd.read_csv(DATA_PATH)

# Prétraitement
df['YYYYMM'] = df['YYYYMM'].astype(str).str.zfill(6)
# Filtrer les lignes où YYYYMM est un nombre valide de 6 chiffres
df = df[df['YYYYMM'].str.match(r'^\d{6}$')]
df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m', errors='coerce')
df = df.dropna(subset=['Date'])  # Supprimer les lignes avec des dates invalides
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])
seuil = df['Value'].quantile(0.99)
df = df[df['Value'] <= seuil]

# Préparation de la série temporelle (moyenne mensuelle)
ts = df.groupby('Date')['Value'].mean().resample('MS').mean()

# Test de stationnarité
adf = adfuller(ts.dropna())
print("Test de Dickey Fuller (H0 : non stationnaire) :")
print(f"p-valeur : {adf[1]:.4f}")
print("La série est stationnaire" if adf[1] < 0.05 else "La série n'est pas stationnaire")

# Différenciation pour rendre la série stationnaire si nécessaire
d = 0
if adf[1] >= 0.05:
    d = 1
    ts_diff = ts.diff().dropna()
    print("\nDifférenciation appliquée (d=1)")

# Entraînement du modèle ARIMA
print("\nEntraînement du modèle ARIMA...")
# Paramètres initiaux (p,d,q) = (2,1,2) - à ajuster selon les résultats
model = ARIMA(ts, order=(2, d, 2))
model_fit = model.fit()
print("\nRésumé du modèle ARIMA :")
print(model_fit.summary())

# Prédictions pour 50 ans
forecast_steps = 50 * 12
forecast = model_fit.forecast(steps=forecast_steps)
forecast_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()

# Création du DataFrame des prédictions
future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
future_df = pd.DataFrame({
    'Date': future_dates,
    'Prédiction': forecast,
    'Intervalle_Bas': forecast_ci.iloc[:, 0],
    'Intervalle_Haut': forecast_ci.iloc[:, 1]
})

# Sauvegarde des résultats
Path('outputs').mkdir(exist_ok=True)
future_df.to_csv('outputs/arima_statsmodels_predictions.csv', index=False)

# Visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Historique', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Prédiction'], mode='lines', name='Prédiction ARIMA', line=dict(color='red')))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Intervalle_Haut'], mode='lines', name='Intervalle haut', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Intervalle_Bas'], mode='lines', name='Intervalle bas', fill='tonexty', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))

fig.update_layout(
    title='Prédictions ARIMA des émissions de CO2 (50 ans)',
    xaxis_title='Date',
    yaxis_title='Émissions (Million Metric Tons of CO2)',
    hovermode='x unified'
)

Path('figures').mkdir(exist_ok=True)
fig.write_html('figures/arima_statsmodels_predictions.html')

print("\nPrédictions sauvegardées dans outputs/arima_statsmodels_predictions.csv")
print("Visualisation interactive : figures/arima_statsmodels_predictions.html") 