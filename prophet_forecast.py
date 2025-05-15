import pandas as pd
try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet
import plotly.graph_objects as go
from pathlib import Path

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

# Adapter au format Prophet
df_prophet = df[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})

# Instancier et entraîner Prophet
model = Prophet(yearly_seasonality=True, seasonality_mode='multiplicative')
model.fit(df_prophet)

# Créer le DataFrame pour les 50 prochaines années (mois par mois)
future = model.make_future_dataframe(periods=50*12, freq='MS')
forecast = model.predict(future)

# Sauvegarder les résultats
Path('outputs').mkdir(exist_ok=True)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('outputs/prophet_predictions_futures.csv', index=False)

# Visualisation interactive
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Historique', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Prédiction Prophet', line=dict(color='green')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Intervalle haut', line=dict(color='rgba(0,200,0,0.2)'), showlegend=False))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Intervalle bas', fill='tonexty', line=dict(color='rgba(0,200,0,0.2)'), showlegend=False))
fig.update_layout(title='Prédictions Prophet pour les 50 prochaines années', xaxis_title='Date', yaxis_title='Émissions (Million Metric Tons of CO2)')
Path('figures').mkdir(exist_ok=True)
fig.write_html('figures/prophet_predictions_futures.html')

print("Prédictions Prophet sauvegardées dans outputs/prophet_predictions_futures.csv")
print("Visualisation interactive : figures/prophet_predictions_futures.html") 