import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class DataAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.analysis_results = {}

    def preprocess_data(self):
        self.data['YYYYMM'] = self.data['YYYYMM'].astype(str).str.zfill(6)
        self.data['Year'] = self.data['YYYYMM'].str[:4].astype(int)
        self.data['Month'] = self.data['YYYYMM'].str[4:6].astype(int)
        self.data = self.data[self.data['Month'] <= 12]
        self.data['Date'] = pd.to_datetime(self.data['YYYYMM'], format='%Y%m')
        self.data['Value'] = pd.to_numeric(self.data['Value'], errors='coerce')
        self.data = self.data.dropna(subset=['Value'])
        self.data = self.data.dropna()
        print("Prétraitement terminé avec succès")
        return self.data

    def analyze_basic_stats(self) -> pd.DataFrame:
        stats = self.data.describe()
        self.analysis_results['basic_stats'] = stats
        return stats

    def analyze_missing_values(self) -> pd.DataFrame:
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Valeurs Manquantes': missing_values,
            'Pourcentage': missing_percentage
        })
        self.analysis_results['missing_values'] = missing_df
        return missing_df

    def plot_emissions_trend(self, save_path=None):
        fig = px.line(self.data, x='Date', y='Value',
                     title='Évolution des Émissions de CO2 du Secteur Électrique au Charbon',
                     labels={'Value': 'Émissions (Million Metric Tons of CO2)', 'Date': 'Date'})
        if save_path:
            fig.write_image(save_path)
        fig.show()

    def plot_monthly_patterns(self, save_path=None):
        monthly_avg = self.data.groupby('Month')['Value'].mean().reset_index()
        fig = px.bar(monthly_avg, x='Month', y='Value',
                    title='Moyenne Mensuelle des Émissions de CO2',
                    labels={'Value': 'Émissions Moyennes (Million Metric Tons of CO2)', 'Month': 'Mois'})
        if save_path:
            fig.write_image(save_path)
        fig.show()

    def plot_yearly_trend(self, save_path=None):
        yearly_avg = self.data.groupby('Year')['Value'].mean().reset_index()
        fig = px.line(yearly_avg, x='Year', y='Value',
                     title='Tendance Annuelle des Émissions de CO2',
                     labels={'Value': 'Émissions Moyennes (Million Metric Tons of CO2)', 'Year': 'Année'})
        if save_path:
            fig.write_image(save_path)
        fig.show()

    def analyze_seasonality(self, save_path=None):
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
        if save_path:
            fig.write_image(save_path)
        fig.show()
        return monthly_stats 