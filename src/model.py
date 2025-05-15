import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go

class EmissionPredictor:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()

    def prepare_features(self, target_column='Value', test_size=0.2):
        X = pd.DataFrame({
            'Year': self.data['Year'],
            'Month': self.data['Month'],
            'Month_sin': np.sin(2 * np.pi * self.data['Month']/12),
            'Month_cos': np.cos(2 * np.pi * self.data['Month']/12)
        })
        y = self.data[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Préparation des features terminée")

    def train_models(self):
        self.models['linear'] = LinearRegression()
        self.models['linear'].fit(self.X_train, self.y_train)
        self.models['gb'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models['gb'].fit(self.X_train, self.y_train)
        self.models['xgb'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.models['xgb'].fit(self.X_train, self.y_train)
        print("Entraînement des modèles terminé")

    def evaluate_models(self):
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results[name] = {
                'MSE': mse,
                'RMSE': np.sqrt(mse),
                'R2': r2
            }
        return pd.DataFrame(results).T

    def plot_predictions(self, model_name='xgb'):
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        y_pred = self.models[model_name].predict(self.X_test)
        dates = self.data['Date'].iloc[-len(self.y_test):]
        results_df = pd.DataFrame({
            'Date': dates,
            'Réel': self.y_test.values,
            'Prédit': y_pred
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Réel'],
            mode='lines',
            name='Valeurs Réelles'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Prédit'],
            mode='lines',
            name='Prédictions',
            line=dict(dash='dash')
        ))
        fig.update_layout(
            title='Prédictions vs Valeurs Réelles des Émissions de CO2',
            xaxis_title='Date',
            yaxis_title='Émissions (Million Metric Tons of CO2)',
            showlegend=True
        )
        fig.show()

    def save_predictions(self, model_name='xgb', output_file='outputs/predictions.csv'):
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        y_pred = self.models[model_name].predict(self.X_test)
        dates = self.data['Date'].iloc[-len(self.y_test):]
        results_df = pd.DataFrame({
            'Date': dates,
            'Réel': self.y_test.values,
            'Prédit': y_pred
        })
        results_df.to_csv(output_file, index=False)
        print(f"\nPrédictions sauvegardées dans {output_file}")
        print("\nAperçu des 10 premières prédictions :")
        print(results_df.head(10)) 