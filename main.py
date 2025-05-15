import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from datetime import datetime

class DataAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.analysis_results = {}
        
    def preprocess_data(self):
        """Prétraite les données pour l'analyse"""
        # Extraction du mois et de l'année à partir de la colonne YYYYMM
        self.data['YYYYMM'] = self.data['YYYYMM'].astype(str).str.zfill(6)
        self.data['Year'] = self.data['YYYYMM'].str[:4].astype(int)
        self.data['Month'] = self.data['YYYYMM'].str[4:6].astype(int)
        
        # Suppression des lignes avec des mois invalides (>12)
        self.data = self.data[self.data['Month'] <= 12]
        
        # Conversion de la colonne YYYYMM en datetime
        self.data['Date'] = pd.to_datetime(self.data['YYYYMM'], format='%Y%m')
        
        # Conversion de Value en float, suppression des valeurs non numériques
        self.data['Value'] = pd.to_numeric(self.data['Value'], errors='coerce')
        self.data = self.data.dropna(subset=['Value'])
        
        # Nettoyage des outliers (on garde <= 99e percentile)
        seuil = self.data['Value'].quantile(0.99)
        self.data = self.data[self.data['Value'] <= seuil]
        
        # Suppression des lignes avec des valeurs manquantes
        self.data = self.data.dropna()
        
        print("Prétraitement terminé avec succès (outliers supprimés)")
        return self.data
    
    def analyze_basic_stats(self) -> pd.DataFrame:
        """Analyse statistique descriptive des données"""
        stats = self.data.describe()
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """Analyse des valeurs manquantes"""
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Valeurs Manquantes': missing_values,
            'Pourcentage': missing_percentage
        })
        self.analysis_results['missing_values'] = missing_df
        return missing_df
    
    def plot_emissions_trend(self):
        """Visualise l'évolution des émissions dans le temps"""
        fig = px.line(self.data, 
                     x='Date', 
                     y='Value',
                     title='Évolution des Émissions de CO2 du Secteur Électrique au Charbon',
                     labels={'Value': 'Émissions (Million Metric Tons of CO2)',
                            'Date': 'Date'})
        fig.show()
    
    def plot_monthly_patterns(self):
        """Visualise les patterns mensuels des émissions"""
        monthly_avg = self.data.groupby('Month')['Value'].mean().reset_index()
        
        fig = px.bar(monthly_avg,
                    x='Month',
                    y='Value',
                    title='Moyenne Mensuelle des Émissions de CO2',
                    labels={'Value': 'Émissions Moyennes (Million Metric Tons of CO2)',
                           'Month': 'Mois'})
        fig.show()
    
    def plot_yearly_trend(self):
        """Visualise la tendance annuelle des émissions"""
        yearly_avg = self.data.groupby('Year')['Value'].mean().reset_index()
        
        fig = px.line(yearly_avg,
                     x='Year',
                     y='Value',
                     title='Tendance Annuelle des Émissions de CO2',
                     labels={'Value': 'Émissions Moyennes (Million Metric Tons of CO2)',
                            'Year': 'Année'})
        fig.show()
    
    def analyze_seasonality(self):
        """Analyse la saisonnalité des émissions"""
        # Calcul des moyennes mensuelles
        monthly_stats = self.data.groupby('Month')['Value'].agg(['mean', 'std']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_stats['Month'],
            y=monthly_stats['mean'],
            name='Moyenne',
            error_y=dict(type='data', array=monthly_stats['std'], visible=True)
        ))
        
        fig.update_layout(
            title='Saisonnalité des Émissions de CO2',
            xaxis_title='Mois',
            yaxis_title='Émissions (Million Metric Tons of CO2)',
            showlegend=True
        )
        fig.show()
        
        return monthly_stats

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
        """Prépare les features pour la prédiction avec des features plus sophistiquées et transformation log"""
        # Création d'un DataFrame pour les features
        X = pd.DataFrame(index=self.data.index)
        
        # Features temporelles de base
        X['Year'] = self.data['Year']
        X['Month'] = self.data['Month']
        X['Month_sin'] = np.sin(2 * np.pi * self.data['Month']/12)
        X['Month_cos'] = np.cos(2 * np.pi * self.data['Month']/12)
        
        # Features de tendance
        X['Trend'] = np.arange(len(self.data))
        
        # Moyennes mobiles
        for window in [3, 6, 12]:
            X[f'MA_{window}'] = self.data[target_column].rolling(window=window).mean()
            X[f'MA_{window}_std'] = self.data[target_column].rolling(window=window).std()
        
        # Features de lag
        for lag in [1, 2, 3, 6, 12]:
            X[f'Lag_{lag}'] = self.data[target_column].shift(lag)
        
        # Features de saisonnalité
        X['Season'] = self.data['Month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else 'Fall')
        X = pd.get_dummies(X, columns=['Season'], prefix='Season')
        
        # Features polynomiales pour capturer des relations non-linéaires
        X['Year_squared'] = X['Year'] ** 2
        X['Month_squared'] = X['Month'] ** 2
        
        # Suppression des lignes avec des valeurs manquantes (créées par les lags)
        X = X.dropna()
        y = self.data.loc[X.index, target_column]
        
        # Transformation log(1+x) de la variable cible
        y = np.log1p(y)
        self.y_transformer = StandardScaler()
        y = self.y_transformer.fit_transform(y.values.reshape(-1, 1)).ravel()
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Normalisation des features
        self.X_scaler = StandardScaler()
        self.X_train = self.X_scaler.fit_transform(self.X_train)
        self.X_test = self.X_scaler.transform(self.X_test)
        
        print("Préparation des features terminée")
        print(f"Nombre de features : {X.shape[1]}")
        return X.columns.tolist()
    
    def train_models(self):
        """Entraîne différents modèles de régression avec des hyperparamètres optimisés"""
        # Régression Linéaire avec régularisation
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['ridge'].fit(self.X_train, self.y_train)
        
        # Gradient Boosting avec plus d'arbres et de profondeur
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        )
        self.models['gb'].fit(self.X_train, self.y_train)
        
        # XGBoost avec des hyperparamètres optimisés
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.models['xgb'].fit(self.X_train, self.y_train)
        
        print("Entraînement des modèles terminé")
    
    def evaluate_models(self):
        """Évalue les performances des modèles"""
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
        """Visualise les prédictions vs valeurs réelles"""
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        y_pred = self.models[model_name].predict(self.X_test)
        
        # Création d'un DataFrame pour la visualisation
        dates = self.data['Date'].iloc[-len(self.y_test):]
        results_df = pd.DataFrame({
            'Date': dates,
            'Réel': self.y_test,
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

    def save_predictions(self, model_name='xgb', output_file='predictions.csv'):
        """Sauvegarde les prédictions et les valeurs réelles dans un fichier CSV"""
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        y_pred = self.models[model_name].predict(self.X_test)
        dates = self.data['Date'].iloc[-len(self.y_test):]
        
        # Transformation inverse des prédictions et des valeurs réelles
        y_pred = self.y_transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_pred = np.expm1(y_pred)
        y_test = self.y_transformer.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        y_test = np.expm1(y_test)
        
        results_df = pd.DataFrame({
            'Date': dates,
            'Réel': y_test,
            'Prédit': y_pred
        })
        results_df.to_csv(output_file, index=False)
        print(f"\nPrédictions sauvegardées dans {output_file}")
        print("\nAperçu des 10 premières prédictions :")
        print(results_df.head(10))

    def predict_future(self, years_ahead=50, model_name='xgb'):
        """Prédit les émissions pour les années futures avec les nouvelles features"""
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        # Création des dates futures
        last_date = self.data['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                   periods=years_ahead*12,
                                   freq='ME')
        
        # Création du DataFrame pour les features futures
        future_features = pd.DataFrame(index=range(len(future_dates)))
        future_features['Year'] = future_dates.year
        future_features['Month'] = future_dates.month
        future_features['Month_sin'] = np.sin(2 * np.pi * future_dates.month/12)
        future_features['Month_cos'] = np.cos(2 * np.pi * future_dates.month/12)
        future_features['Trend'] = np.arange(len(self.data), len(self.data) + len(future_dates))
        
        # Ajout des features de saisonnalité
        future_features['Season'] = [
            'Winter' if m in [12, 1, 2] else
            'Spring' if m in [3, 4, 5] else
            'Summer' if m in [6, 7, 8] else 'Fall'
            for m in future_dates.month
        ]
        future_features = pd.get_dummies(future_features, columns=['Season'], prefix='Season')
        
        # Features polynomiales
        future_features['Year_squared'] = future_features['Year'] ** 2
        future_features['Month_squared'] = future_features['Month'] ** 2
        
        # Compléter les colonnes manquantes (lags, moyennes mobiles) par 0
        for col in self.X_scaler.feature_names_in_:
            if col not in future_features.columns:
                future_features[col] = 0
        future_features = future_features[self.X_scaler.feature_names_in_]
        
        # Normalisation des features futures
        future_features_scaled = self.X_scaler.transform(future_features)
        
        # Prédictions
        future_predictions = self.models[model_name].predict(future_features_scaled)
        
        # Transformation inverse des prédictions
        future_predictions = self.y_transformer.inverse_transform(
            future_predictions.reshape(-1, 1)
        ).ravel()
        future_predictions = np.expm1(future_predictions)
        
        # Création du DataFrame des prédictions futures
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Prédiction': future_predictions
        })
        
        # Visualisation des prédictions futures
        fig = go.Figure()
        
        # Ajout des données historiques
        fig.add_trace(go.Scatter(
            x=self.data['Date'],
            y=self.data['Value'],
            mode='lines',
            name='Données Historiques',
            line=dict(color='blue')
        ))
        
        # Ajout des prédictions futures
        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Prédiction'],
            mode='lines',
            name='Prédictions Futures',
            line=dict(color='red', dash='dash')
        ))
        
        # Ajout d'une bande de confiance
        std_dev = np.std(self.data['Value'])
        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Prédiction'] + 2*std_dev,
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Intervalle de Confiance'
        ))
        fig.add_trace(go.Scatter(
            x=future_df['Date'],
            y=future_df['Prédiction'] - 2*std_dev,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Intervalle de Confiance'
        ))
        
        fig.update_layout(
            title=f'Prédictions des Émissions de CO2 pour les {years_ahead} Prochaines Années',
            xaxis_title='Date',
            yaxis_title='Émissions (Million Metric Tons of CO2)',
            showlegend=True
        )
        
        # Sauvegarde de la figure
        fig.write_html('figures/predictions_futures.html')
        fig.show()
        
        # Sauvegarde des prédictions futures
        future_df.to_csv('outputs/predictions_futures.csv', index=False)
        print(f"\nPrédictions futures sauvegardées dans 'outputs/predictions_futures.csv'")
        
        return future_df

def main():
    # Chargement des données
    data_path = 'data/MER_T12_06.csv'
    df = pd.read_csv(data_path)
    
    # Analyse exploratoire
    print("\n=== Analyse Exploratoire des Données ===")
    analyzer = DataAnalyzer(df)
    df = analyzer.preprocess_data()
    
    # Statistiques de base
    print("\nStatistiques descriptives :")
    print(analyzer.analyze_basic_stats())
    
    # Analyse des valeurs manquantes
    print("\nAnalyse des valeurs manquantes :")
    print(analyzer.analyze_missing_values())
    
    # Visualisations
    print("\nGénération des visualisations...")
    analyzer.plot_emissions_trend()
    analyzer.plot_monthly_patterns()
    analyzer.plot_yearly_trend()
    analyzer.analyze_seasonality()
    
    # Modélisation
    print("\n=== Début de la Modélisation ===")
    predictor = EmissionPredictor(df)
    predictor.prepare_features()
    predictor.train_models()
    
    # Évaluation
    results = predictor.evaluate_models()
    print("\nRésultats d'évaluation des modèles :")
    print(results)
    
    # Visualisation des prédictions
    predictor.plot_predictions()
    predictor.save_predictions()
    
    # Prédictions futures
    print("\n=== Prédictions pour les 50 Prochaines Années ===")
    future_predictions = predictor.predict_future(years_ahead=50, model_name='xgb')
    
    # Affichage des statistiques des prédictions futures
    print("\nStatistiques des prédictions futures :")
    print(future_predictions['Prédiction'].describe())

if __name__ == "__main__":
    main() 