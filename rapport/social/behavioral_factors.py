#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse des facteurs comportementaux et sociaux influençant les émissions de CO2
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import json

class BehavioralAnalysis:
    def __init__(self, data_path='data/MER_T12_06.csv'):
        """Initialise l'analyse comportementale"""
        self.data = pd.read_csv(data_path)
        self.data['YYYYMM'] = pd.to_datetime(self.data['YYYYMM'], format='%Y%m')
        self.data.set_index('YYYYMM', inplace=True)
        
        # Sources de données supplémentaires
        self.social_indicators = {
            'consommation_energie': self._get_energy_consumption_data(),
            'transport': self._get_transport_data(),
            'habitat': self._get_housing_data(),
            'consommation': self._get_consumption_data()
        }
        
    def _get_energy_consumption_data(self):
        """Récupère les données de consommation énergétique"""
        # Simulation de données (à remplacer par des données réelles)
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
        data = pd.DataFrame(index=dates)
        
        # Tendances de consommation énergétique
        data['consommation_totale'] = np.random.normal(100, 10, len(dates))
        data['energie_renouvelable'] = np.random.normal(20, 2, len(dates)) * (1 + np.linspace(0, 1, len(dates)))
        data['energie_fossile'] = data['consommation_totale'] - data['energie_renouvelable']
        
        return data
    
    def _get_transport_data(self):
        """Récupère les données de transport"""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
        data = pd.DataFrame(index=dates)
        
        # Indicateurs de transport
        data['voitures_electriques'] = np.random.normal(5, 1, len(dates)) * (1 + np.linspace(0, 2, len(dates)))
        data['transport_public'] = np.random.normal(30, 5, len(dates))
        data['trafic_routier'] = np.random.normal(80, 10, len(dates))
        
        return data
    
    def _get_housing_data(self):
        """Récupère les données d'habitat"""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
        data = pd.DataFrame(index=dates)
        
        # Indicateurs d'habitat
        data['isolation_thermique'] = np.random.normal(60, 5, len(dates)) * (1 + np.linspace(0, 0.5, len(dates)))
        data['chauffage_electrique'] = np.random.normal(40, 5, len(dates))
        data['chauffage_fossile'] = np.random.normal(60, 5, len(dates)) * (1 - np.linspace(0, 0.3, len(dates)))
        
        return data
    
    def _get_consumption_data(self):
        """Récupère les données de consommation"""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='M')
        data = pd.DataFrame(index=dates)
        
        # Indicateurs de consommation
        data['produits_locaux'] = np.random.normal(30, 5, len(dates)) * (1 + np.linspace(0, 0.8, len(dates)))
        data['produits_bio'] = np.random.normal(15, 3, len(dates)) * (1 + np.linspace(0, 1.5, len(dates)))
        data['emballages_recyclables'] = np.random.normal(50, 8, len(dates)) * (1 + np.linspace(0, 0.6, len(dates)))
        
        return data
    
    def analyze_behavioral_factors(self):
        """Analyse les facteurs comportementaux"""
        logging.info("Démarrage de l'analyse des facteurs comportementaux...")
        
        results = {}
        
        # Analyse des corrélations
        correlations = self._analyze_correlations()
        results['correlations'] = correlations
        
        # Analyse des tendances comportementales
        trends = self._analyze_behavioral_trends()
        results['trends'] = trends
        
        # Analyse des changements de comportement
        changes = self._analyze_behavioral_changes()
        results['changes'] = changes
        
        # Sauvegarde des résultats
        self._save_results(results)
        
        # Génération des visualisations
        self._create_visualizations(results)
        
        logging.info("Analyse des facteurs comportementaux terminée")
        return results
    
    def _analyze_correlations(self):
        """Analyse les corrélations entre les indicateurs sociaux et les émissions"""
        correlations = {}
        
        # Création d'un DataFrame combiné
        combined_data = pd.DataFrame(index=self.data.index)
        combined_data['emissions'] = self.data['Value']
        
        for category, indicators in self.social_indicators.items():
            for column in indicators.columns:
                combined_data[f'{category}_{column}'] = indicators[column]
        
        # Calcul des corrélations
        corr_matrix = combined_data.corr()['emissions'].sort_values(ascending=False)
        correlations['all'] = corr_matrix
        
        # Corrélations par catégorie
        for category in self.social_indicators.keys():
            category_cols = [col for col in combined_data.columns if col.startswith(category)]
            corr_matrix = combined_data[category_cols + ['emissions']].corr()['emissions']
            correlations[category] = corr_matrix
        
        return correlations
    
    def _analyze_behavioral_trends(self):
        """Analyse les tendances comportementales"""
        trends = {}
        
        for category, indicators in self.social_indicators.items():
            category_trends = {}
            for column in indicators.columns:
                # Calcul de la tendance linéaire
                x = np.arange(len(indicators))
                y = indicators[column].values
                slope, _ = np.polyfit(x, y, 1)
                
                # Calcul du changement moyen annuel
                yearly_change = slope * 12
                
                category_trends[column] = {
                    'slope': slope,
                    'yearly_change': yearly_change,
                    'change_percent': (yearly_change / y.mean()) * 100
                }
            
            trends[category] = category_trends
        
        return trends
    
    def _analyze_behavioral_changes(self):
        """Analyse les changements de comportement significatifs"""
        changes = {}
        
        for category, indicators in self.social_indicators.items():
            category_changes = {}
            for column in indicators.columns:
                # Calcul des changements par période
                data = indicators[column]
                periods = {
                    '2010-2015': data['2010':'2015'],
                    '2015-2020': data['2015':'2020'],
                    '2020-2023': data['2020':'2023']
                }
                
                period_changes = {}
                for period, period_data in periods.items():
                    change = (
                        period_data.mean() - data['2010':'2015'].mean()
                    ) / data['2010':'2015'].mean() * 100
                    period_changes[period] = change
                
                category_changes[column] = period_changes
            
            changes[category] = category_changes
        
        return changes
    
    def _save_results(self, results):
        """Sauvegarde les résultats de l'analyse"""
        # Sauvegarde des corrélations
        for category, corr in results['correlations'].items():
            corr.to_csv(f'rapport/data/behavioral_correlations_{category}.csv')
        
        # Sauvegarde des tendances
        trends_df = pd.DataFrame()
        for category, trends in results['trends'].items():
            for indicator, trend_data in trends.items():
                trends_df.loc[f'{category}_{indicator}'] = trend_data
        trends_df.to_csv('rapport/data/behavioral_trends.csv')
        
        # Sauvegarde des changements
        changes_df = pd.DataFrame()
        for category, changes in results['changes'].items():
            for indicator, period_changes in changes.items():
                for period, change in period_changes.items():
                    changes_df.loc[f'{category}_{indicator}', period] = change
        changes_df.to_csv('rapport/data/behavioral_changes.csv')
    
    def _create_visualizations(self, results):
        """Crée les visualisations de l'analyse"""
        # Graphique des corrélations
        fig_corr = go.Figure()
        
        for category, corr in results['correlations'].items():
            if category != 'all':
                fig_corr.add_trace(go.Bar(
                    name=category,
                    x=corr.index,
                    y=corr.values,
                    text=[f'{v:.2f}' for v in corr.values],
                    textposition='auto',
                ))
        
        fig_corr.update_layout(
            title='Corrélations entre indicateurs sociaux et émissions de CO2',
            xaxis_title='Indicateur',
            yaxis_title='Corrélation',
            template='plotly_white',
            showlegend=True
        )
        
        fig_corr.write_html('rapport/figures/behavioral_correlations.html')
        
        # Graphique des tendances
        fig_trends = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Tendances de consommation énergétique',
                'Tendances de transport',
                'Tendances d\'habitat',
                'Tendances de consommation'
            )
        )
        
        for i, (category, indicators) in enumerate(self.social_indicators.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            for column in indicators.columns:
                fig_trends.add_trace(
                    go.Scatter(
                        x=indicators.index,
                        y=indicators[column],
                        name=f'{category}: {column}',
                        mode='lines'
                    ),
                    row=row,
                    col=col
                )
        
        fig_trends.update_layout(
            title='Évolution des indicateurs comportementaux',
            template='plotly_white',
            showlegend=True,
            height=800
        )
        
        fig_trends.write_html('rapport/figures/behavioral_trends.html')
        
        # Graphique des changements
        fig_changes = go.Figure()
        
        changes_df = pd.DataFrame()
        for category, changes in results['changes'].items():
            for indicator, period_changes in changes.items():
                for period, change in period_changes.items():
                    changes_df.loc[f'{category}_{indicator}', period] = change
        
        for period in changes_df.columns:
            fig_changes.add_trace(go.Bar(
                name=period,
                x=changes_df.index,
                y=changes_df[period],
                text=[f'{v:.1f}%' for v in changes_df[period]],
                textposition='auto',
            ))
        
        fig_changes.update_layout(
            title='Changements comportementaux par période',
            xaxis_title='Indicateur',
            yaxis_title='Changement (%)',
            template='plotly_white',
            showlegend=True
        )
        
        fig_changes.write_html('rapport/figures/behavioral_changes.html')

def analyze_behavioral_factors():
    """Fonction principale pour l'analyse des facteurs comportementaux"""
    analysis = BehavioralAnalysis()
    return analysis.analyze_behavioral_factors()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_behavioral_factors() 