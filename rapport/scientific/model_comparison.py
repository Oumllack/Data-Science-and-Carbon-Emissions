#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de comparaison des différents modèles de prédiction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging
import json
from pathlib import Path

class ModelComparison:
    def __init__(self, data_path='data/MER_T12_06.csv'):
        """Initialise la comparaison des modèles"""
        self.data = pd.read_csv(data_path)
        self.data['YYYYMM'] = pd.to_datetime(self.data['YYYYMM'], format='%Y%m')
        self.data.set_index('YYYYMM', inplace=True)
        
        # Paramètres de test
        self.test_size = 120  # 10 ans de données
        self.train_data = self.data[:-self.test_size]
        self.test_data = self.data[-self.test_size:]
        
        # Stockage des résultats
        self.results = {}
        
    def prepare_data(self):
        """Prépare les données pour les différents modèles"""
        # Données pour ARIMA et Prophet
        self.arima_data = self.data['Value'].copy()
        self.prophet_data = self.data.reset_index().rename(
            columns={'YYYYMM': 'ds', 'Value': 'y'}
        )
        
        # Données pour XGBoost et LSTM
        self.features = self._create_features()
        self.X_train = self.features[:-self.test_size]
        self.X_test = self.features[-self.test_size:]
        self.y_train = self.data['Value'][:-self.test_size]
        self.y_test = self.data['Value'][-self.test_size:]
        
    def _create_features(self):
        """Crée les features pour XGBoost et LSTM"""
        df = self.data.copy()
        
        # Features temporelles
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Features cycliques
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Features de tendance
        df['trend'] = np.arange(len(df))
        
        # Features de lag
        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = df['Value'].shift(lag)
        
        # Moyennes mobiles
        for window in [3, 6, 12]:
            df[f'ma_{window}'] = df['Value'].rolling(window=window).mean()
        
        # Features saisonnières
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        return df.dropna()
    
    def train_arima(self):
        """Entraîne le modèle ARIMA"""
        model = ARIMA(self.arima_data[:-self.test_size], order=(2,1,2))
        self.arima_model = model.fit()
        predictions = self.arima_model.forecast(steps=self.test_size)
        self.results['ARIMA'] = self._evaluate_model(
            self.test_data['Value'], predictions
        )
        
    def train_xgboost(self):
        """Entraîne le modèle XGBoost"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.results['XGBoost'] = self._evaluate_model(
            self.y_test, predictions
        )
        
    def train_prophet(self):
        """Entraîne le modèle Prophet"""
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(self.prophet_data[:-self.test_size])
        future = model.make_future_dataframe(periods=self.test_size, freq='M')
        forecast = model.predict(future)
        predictions = forecast['yhat'][-self.test_size:]
        self.results['Prophet'] = self._evaluate_model(
            self.test_data['Value'], predictions
        )
        
    def train_lstm(self):
        """Entraîne le modèle LSTM"""
        # Préparation des données pour LSTM
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:(i + seq_length)])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 12
        X_train_seq, y_train_seq = create_sequences(
            self.X_train.values, seq_length
        )
        X_test_seq, y_test_seq = create_sequences(
            self.X_test.values, seq_length
        )
        
        # Architecture du modèle
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, X_train.shape[1])),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        predictions = model.predict(X_test_seq)
        self.results['LSTM'] = self._evaluate_model(
            y_test_seq, predictions.flatten()
        )
    
    def _evaluate_model(self, y_true, y_pred):
        """Évalue les performances d'un modèle"""
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    def compare_models(self):
        """Exécute la comparaison complète des modèles"""
        logging.info("Démarrage de la comparaison des modèles...")
        
        self.prepare_data()
        self.train_arima()
        self.train_xgboost()
        self.train_prophet()
        self.train_lstm()
        
        # Sauvegarde des résultats
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv('rapport/data/model_comparison_results.csv')
        
        # Génération du graphique de comparaison
        self._plot_comparison()
        
        logging.info("Comparaison des modèles terminée")
        return results_df
    
    def _plot_comparison(self):
        """Génère un graphique de comparaison des modèles"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Ajout des métriques pour chaque modèle
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        for metric in metrics:
            values = [self.results[model][metric] for model in self.results.keys()]
            fig.add_trace(go.Bar(
                name=metric,
                x=list(self.results.keys()),
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Comparaison des performances des modèles',
            xaxis_title='Modèle',
            yaxis_title='Valeur',
            barmode='group',
            template='plotly_white'
        )
        
        fig.write_html('rapport/figures/model_comparison.html')

def compare_models():
    """Fonction principale pour la comparaison des modèles"""
    comparison = ModelComparison()
    return comparison.compare_models()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compare_models() 