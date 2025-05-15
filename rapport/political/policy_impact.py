#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse de l'impact des politiques climatiques sur les émissions de CO2
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PolicyImpactAnalysis:
    def __init__(self, data_path='data/MER_T12_06.csv'):
        """Initialise l'analyse d'impact des politiques"""
        self.data = pd.read_csv(data_path)
        self.data['YYYYMM'] = pd.to_datetime(self.data['YYYYMM'], format='%Y%m')
        self.data.set_index('YYYYMM', inplace=True)
        
        # Définition des périodes de politiques importantes
        self.policy_periods = {
            'Accord de Paris': {
                'start': '2015-12',
                'end': '2016-12',
                'description': 'Adoption de l\'accord de Paris sur le climat'
            },
            'Pacte Vert Européen': {
                'start': '2019-12',
                'end': '2020-12',
                'description': 'Lancement du Pacte Vert Européen'
            },
            'COVID-19': {
                'start': '2020-03',
                'end': '2021-03',
                'description': 'Période de la pandémie COVID-19'
            }
        }
        
    def analyze_policy_impact(self):
        """Analyse l'impact des politiques sur les émissions"""
        logging.info("Démarrage de l'analyse d'impact des politiques...")
        
        results = {}
        
        # Analyse pour chaque période de politique
        for policy, period in self.policy_periods.items():
            impact = self._analyze_period(
                period['start'],
                period['end'],
                period['description']
            )
            results[policy] = impact
        
        # Analyse des tendances avant/après chaque politique
        trend_analysis = self._analyze_trends()
        results['trend_analysis'] = trend_analysis
        
        # Sauvegarde des résultats
        self._save_results(results)
        
        # Génération des visualisations
        self._create_visualizations(results)
        
        logging.info("Analyse d'impact des politiques terminée")
        return results
    
    def _analyze_period(self, start_date, end_date, description):
        """Analyse l'impact d'une période spécifique"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Données de la période
        period_data = self.data.loc[start:end]
        
        # Données de la période précédente (même durée)
        pre_period = self.data.loc[
            start - (end - start):start
        ]
        
        # Calcul des statistiques
        period_stats = {
            'mean': period_data['Value'].mean(),
            'std': period_data['Value'].std(),
            'min': period_data['Value'].min(),
            'max': period_data['Value'].max(),
            'change_from_pre': (
                period_data['Value'].mean() - pre_period['Value'].mean()
            ) / pre_period['Value'].mean() * 100
        }
        
        return {
            'description': description,
            'period_stats': period_stats,
            'data': period_data
        }
    
    def _analyze_trends(self):
        """Analyse les tendances avant/après chaque politique"""
        trends = {}
        
        for policy, period in self.policy_periods.items():
            start = pd.to_datetime(period['start'])
            end = pd.to_datetime(period['end'])
            
            # Période avant
            pre_data = self.data.loc[:start]
            pre_trend = np.polyfit(
                range(len(pre_data)),
                pre_data['Value'],
                deg=1
            )[0]
            
            # Période après
            post_data = self.data.loc[end:]
            post_trend = np.polyfit(
                range(len(post_data)),
                post_data['Value'],
                deg=1
            )[0]
            
            trends[policy] = {
                'pre_trend': pre_trend,
                'post_trend': post_trend,
                'trend_change': post_trend - pre_trend
            }
        
        return trends
    
    def _save_results(self, results):
        """Sauvegarde les résultats de l'analyse"""
        # Conversion des résultats en DataFrame
        policy_stats = pd.DataFrame({
            policy: results[policy]['period_stats']
            for policy in self.policy_periods.keys()
        }).T
        
        trend_stats = pd.DataFrame({
            policy: results['trend_analysis'][policy]
            for policy in self.policy_periods.keys()
        }).T
        
        # Sauvegarde
        policy_stats.to_csv('rapport/data/policy_impact_stats.csv')
        trend_stats.to_csv('rapport/data/policy_trend_stats.csv')
        
        # Sauvegarde des données détaillées
        for policy in self.policy_periods.keys():
            results[policy]['data'].to_csv(
                f'rapport/data/policy_{policy.lower().replace(" ", "_")}_data.csv'
            )
    
    def _create_visualizations(self, results):
        """Crée les visualisations de l'analyse"""
        # Graphique principal des émissions avec périodes de politiques
        fig = go.Figure()
        
        # Ajout des données d'émissions
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Value'],
            name='Émissions de CO2',
            line=dict(color='blue')
        ))
        
        # Ajout des périodes de politiques
        colors = ['red', 'green', 'orange']
        for (policy, period), color in zip(self.policy_periods.items(), colors):
            start = pd.to_datetime(period['start'])
            end = pd.to_datetime(period['end'])
            
            # Zone de la politique
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                name=policy
            )
            
            # Annotation
            fig.add_annotation(
                x=start,
                y=self.data['Value'].max(),
                text=policy,
                showarrow=True,
                arrowhead=1,
                yshift=10
            )
        
        fig.update_layout(
            title='Impact des politiques climatiques sur les émissions de CO2',
            xaxis_title='Date',
            yaxis_title='Émissions de CO2 (millions de tonnes)',
            template='plotly_white',
            showlegend=True
        )
        
        fig.write_html('rapport/figures/policy_impact.html')
        
        # Graphique des changements de tendance
        fig_trends = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Tendances avant/après les politiques', 'Changements de tendance')
        )
        
        # Tendances
        for policy, trend_data in results['trend_analysis'].items():
            fig_trends.add_trace(
                go.Bar(
                    name=policy,
                    x=[policy],
                    y=[trend_data['pre_trend']],
                    name='Avant',
                    marker_color='blue'
                ),
                row=1, col=1
            )
            fig_trends.add_trace(
                go.Bar(
                    name=policy,
                    x=[policy],
                    y=[trend_data['post_trend']],
                    name='Après',
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # Changements
        fig_trends.add_trace(
            go.Bar(
                x=list(results['trend_analysis'].keys()),
                y=[data['trend_change'] for data in results['trend_analysis'].values()],
                name='Changement de tendance',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        fig_trends.update_layout(
            title='Analyse des tendances des émissions',
            template='plotly_white',
            showlegend=True
        )
        
        fig_trends.write_html('rapport/figures/policy_trends.html')

def analyze_policy_impact():
    """Fonction principale pour l'analyse d'impact des politiques"""
    analysis = PolicyImpactAnalysis()
    return analysis.analyze_policy_impact()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyze_policy_impact() 