import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

def prepare_features(df, is_future=False, last_values=None):
    """Préparation des features avancées"""
    df = df.copy()
    
    # Features temporelles
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['DayOfYear'] = df.index.dayofyear
    
    # Features cycliques
    df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
    df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter']/4)
    df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter']/4)
    
    # Features de tendance
    df['Trend'] = np.arange(len(df))
    df['Trend_squared'] = df['Trend'] ** 2
    
    if not is_future:
        # Features de lag et moyennes mobiles uniquement pour les données historiques
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df['Value'].shift(lag)
        
        for window in [3, 6, 12]:
            df[f'MA_{window}'] = df['Value'].rolling(window=window).mean()
    else:
        # Pour les prédictions futures, utiliser les dernières valeurs connues
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = np.nan
            if last_values is not None and len(last_values) >= lag:
                df.loc[df.index[0], f'lag_{lag}'] = last_values[-lag]
        
        for window in [3, 6, 12]:
            df[f'MA_{window}'] = np.nan
            if last_values is not None and len(last_values) >= window:
                df.loc[df.index[0], f'MA_{window}'] = np.mean(last_values[-window:])
    
    # Features saisonnières
    df['Is_Summer'] = df['Month'].isin([6, 7, 8]).astype(int)
    df['Is_Winter'] = df['Month'].isin([12, 1, 2]).astype(int)
    
    return df

def train_arima(ts, order):
    """Entraînement du modèle ARIMA"""
    model = ARIMA(ts, order=order)
    return model.fit()

def train_xgboost(X_train, y_train):
    """Entraînement du modèle XGBoost avec hyperparamètres optimisés"""
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    return model.fit(X_train, y_train)

def ensemble_predict(arima_model, xgb_model, X_future, last_values, steps):
    """Prédictions combinées ARIMA + XGBoost"""
    # Prédictions ARIMA
    arima_pred = arima_model.forecast(steps=steps)
    arima_ci = arima_model.get_forecast(steps=steps).conf_int()
    
    # Prédictions XGBoost
    xgb_pred = xgb_model.predict(X_future)
    
    # Combinaison pondérée (70% ARIMA, 30% XGBoost)
    ensemble_pred = 0.7 * arima_pred + 0.3 * xgb_pred
    
    # Ajustement des intervalles de confiance
    ensemble_ci_lower = arima_ci.iloc[:, 0] * 0.7 + xgb_pred * 0.3
    ensemble_ci_upper = arima_ci.iloc[:, 1] * 0.7 + xgb_pred * 0.3
    
    return ensemble_pred, ensemble_ci_lower, ensemble_ci_upper

# Chargement et prétraitement des données
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

# Préparation de la série temporelle
ts = df.groupby('Date')['Value'].mean().resample('MS').mean()
ts_df = pd.DataFrame({'Value': ts})
ts_df = prepare_features(ts_df)
ts_df = ts_df.dropna()

# Test de stationnarité et différenciation si nécessaire
adf = adfuller(ts.dropna())
d = 0
if adf[1] >= 0.05:
    d = 1
    ts = ts.diff().dropna()
    ts_df['Value'] = ts_df['Value'].diff().dropna()

# Séparation train/test
test_size = 120  # 10 ans
train_df = ts_df[:-test_size]
test_df = ts_df[-test_size:]

# Préparation des features pour XGBoost
feature_cols = [col for col in train_df.columns if col != 'Value']
X_train = train_df[feature_cols]
y_train = train_df['Value']
X_test = test_df[feature_cols]
y_test = test_df['Value']

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement des modèles
print("Entraînement des modèles...")
arima_order = (2, d, 3)  # Meilleur ordre trouvé précédemment
arima_model = train_arima(ts[:-test_size], order=arima_order)
xgb_model = train_xgboost(X_train_scaled, y_train)

# Évaluation sur la période de test
arima_pred = arima_model.forecast(steps=test_size)
xgb_pred = xgb_model.predict(X_test_scaled)
ensemble_pred = 0.7 * arima_pred + 0.3 * xgb_pred

# Métriques d'évaluation
print("\nMétriques sur la période de test (10 dernières années):")
print(f"ARIMA - MSE: {mean_squared_error(y_test, arima_pred):.2f}")
print(f"XGBoost - MSE: {mean_squared_error(y_test, xgb_pred):.2f}")
print(f"Ensemble - MSE: {mean_squared_error(y_test, ensemble_pred):.2f}")
print(f"\nEnsemble - MAE: {mean_absolute_error(y_test, ensemble_pred):.2f}")
print(f"Ensemble - R²: {r2_score(y_test, ensemble_pred):.2f}")

# Prédictions futures
print("\nGénération des prédictions futures...")
future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=50*12, freq='MS')
future_df = pd.DataFrame(index=future_dates)
# Utiliser les 12 dernières valeurs pour les lags et moyennes mobiles
future_df = prepare_features(future_df, is_future=True, last_values=ts.values[-12:])

# Préparation des features pour les prédictions futures
X_future = future_df[feature_cols]
X_future_scaled = scaler.transform(X_future)

# Prédictions ensemble
ensemble_pred, ci_lower, ci_upper = ensemble_predict(
    arima_model, xgb_model, X_future_scaled, ts.values[-12:], len(future_dates)
)

# Création du DataFrame des prédictions
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Prédiction': ensemble_pred,
    'Intervalle_Bas': ci_lower,
    'Intervalle_Haut': ci_upper
})

# Sauvegarde des résultats
Path('outputs').mkdir(exist_ok=True)
predictions_df.to_csv('outputs/ensemble_predictions.csv', index=False)

# Visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Historique', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_df.index, y=y_test, mode='lines', name='Test réel', line=dict(color='black', dash='dot')))
fig.add_trace(go.Scatter(x=test_df.index, y=ensemble_pred[:test_size], mode='lines', name='Prédiction test', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Prédiction'], mode='lines', name='Prédiction future', line=dict(color='red')))
fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Intervalle_Haut'], mode='lines', name='Intervalle haut', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))
fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Intervalle_Bas'], mode='lines', name='Intervalle bas', fill='tonexty', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False))

fig.update_layout(
    title='Prédictions Ensemble (ARIMA + XGBoost) des émissions de CO2',
    xaxis_title='Date',
    yaxis_title='Émissions (Million Metric Tons of CO2)',
    hovermode='x unified'
)

Path('figures').mkdir(exist_ok=True)
fig.write_html('figures/ensemble_predictions.html')

print("\nPrédictions sauvegardées dans outputs/ensemble_predictions.csv")
print("Visualisation interactive : figures/ensemble_predictions.html") 