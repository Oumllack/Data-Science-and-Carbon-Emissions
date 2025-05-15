#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de création du tableau de bord interactif
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import logging
from pathlib import Path
import json

class DashboardCreator:
    def __init__(self):
        """Initialise le créateur de tableau de bord"""
        self.app = dash.Dash(__name__)
        self.data = self._load_data()
        self.setup_layout()
        self.setup_callbacks()
        
    def _load_data(self):
        """Charge toutes les données nécessaires"""
        data = {}
        
        # Données principales
        data['emissions'] = pd.read_csv('data/MER_T12_06.csv')
        data['emissions']['YYYYMM'] = pd.to_datetime(data['emissions']['YYYYMM'], format='%Y%m')
        
        # Données des modèles
        data['model_comparison'] = pd.read_csv('rapport/data/model_comparison_results.csv')
        
        # Données politiques
        data['policy_impact'] = pd.read_csv('rapport/data/policy_impact_stats.csv')
        data['policy_trends'] = pd.read_csv('rapport/data/policy_trend_stats.csv')
        
        # Données comportementales
        data['behavioral_correlations'] = pd.read_csv('rapport/data/behavioral_correlations_all.csv')
        data['behavioral_trends'] = pd.read_csv('rapport/data/behavioral_trends.csv')
        data['behavioral_changes'] = pd.read_csv('rapport/data/behavioral_changes.csv')
        
        return data
    
    def setup_layout(self):
        """Configure la mise en page du tableau de bord"""
        self.app.layout = html.Div([
            # En-tête
            html.H1('Tableau de Bord : Analyse des Émissions de CO2',
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Filtres
            html.Div([
                html.Div([
                    html.Label('Période'),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=self.data['emissions']['YYYYMM'].min(),
                        end_date=self.data['emissions']['YYYYMM'].max(),
                        display_format='YYYY-MM'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label('Modèle'),
                    dcc.Dropdown(
                        id='model-selector',
                        options=[
                            {'label': 'ARIMA', 'value': 'ARIMA'},
                            {'label': 'XGBoost', 'value': 'XGBoost'},
                            {'label': 'Prophet', 'value': 'Prophet'},
                            {'label': 'LSTM', 'value': 'LSTM'}
                        ],
                        value='XGBoost'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
            ], style={'margin': '20px'}),
            
            # Graphiques principaux
            html.Div([
                # Émissions et prédictions
                dcc.Graph(id='emissions-graph'),
                
                # Comparaison des modèles
                dcc.Graph(id='model-comparison-graph'),
                
                # Impact des politiques
                dcc.Graph(id='policy-impact-graph'),
                
                # Facteurs comportementaux
                dcc.Graph(id='behavioral-factors-graph')
            ]),
            
            # Métriques clés
            html.Div([
                html.Div([
                    html.H3('Métriques Clés', style={'textAlign': 'center'}),
                    html.Div(id='key-metrics')
                ], style={'width': '100%', 'margin': '20px'})
            ]),
            
            # Onglets pour les analyses détaillées
            dcc.Tabs([
                dcc.Tab(label='Analyse Scientifique', children=[
                    html.Div([
                        dcc.Graph(id='scientific-analysis-graph')
                    ])
                ]),
                dcc.Tab(label='Analyse Politique', children=[
                    html.Div([
                        dcc.Graph(id='political-analysis-graph')
                    ])
                ]),
                dcc.Tab(label='Analyse Sociale', children=[
                    html.Div([
                        dcc.Graph(id='social-analysis-graph')
                    ])
                ])
            ])
        ])
    
    def setup_callbacks(self):
        """Configure les callbacks pour l'interactivité"""
        @self.app.callback(
            [Output('emissions-graph', 'figure'),
             Output('model-comparison-graph', 'figure'),
             Output('policy-impact-graph', 'figure'),
             Output('behavioral-factors-graph', 'figure'),
             Output('key-metrics', 'children')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('model-selector', 'value')]
        )
        def update_graphs(start_date, end_date, selected_model):
            # Filtrage des données
            mask = (self.data['emissions']['YYYYMM'] >= start_date) & \
                   (self.data['emissions']['YYYYMM'] <= end_date)
            filtered_data = self.data['emissions'][mask]
            
            # Graphique des émissions
            emissions_fig = go.Figure()
            emissions_fig.add_trace(go.Scatter(
                x=filtered_data['YYYYMM'],
                y=filtered_data['Value'],
                name='Émissions réelles',
                line=dict(color='blue')
            ))
            
            # Ajout des prédictions du modèle sélectionné
            # (À implémenter avec les données de prédiction)
            
            emissions_fig.update_layout(
                title='Émissions de CO2 et Prédictions',
                xaxis_title='Date',
                yaxis_title='Émissions (millions de tonnes)',
                template='plotly_white'
            )
            
            # Graphique de comparaison des modèles
            model_fig = go.Figure()
            for metric in ['MSE', 'RMSE', 'MAE', 'R2']:
                model_fig.add_trace(go.Bar(
                    name=metric,
                    x=self.data['model_comparison'].index,
                    y=self.data['model_comparison'][metric],
                    text=[f'{v:.2f}' for v in self.data['model_comparison'][metric]],
                    textposition='auto',
                ))
            
            model_fig.update_layout(
                title='Comparaison des Performances des Modèles',
                xaxis_title='Modèle',
                yaxis_title='Valeur',
                template='plotly_white',
                barmode='group'
            )
            
            # Graphique d'impact des politiques
            policy_fig = go.Figure()
            for policy in self.data['policy_impact'].index:
                policy_fig.add_trace(go.Bar(
                    name=policy,
                    x=['Moyenne', 'Écart-type', 'Min', 'Max', 'Changement'],
                    y=self.data['policy_impact'].loc[policy],
                    text=[f'{v:.2f}' for v in self.data['policy_impact'].loc[policy]],
                    textposition='auto',
                ))
            
            policy_fig.update_layout(
                title='Impact des Politiques Climatiques',
                xaxis_title='Métrique',
                yaxis_title='Valeur',
                template='plotly_white',
                barmode='group'
            )
            
            # Graphique des facteurs comportementaux
            behavioral_fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Corrélations',
                    'Tendances',
                    'Changements par Période',
                    'Impact sur les Émissions'
                )
            )
            
            # Corrélations
            behavioral_fig.add_trace(
                go.Bar(
                    x=self.data['behavioral_correlations'].index,
                    y=self.data['behavioral_correlations']['correlation'],
                    name='Corrélations'
                ),
                row=1, col=1
            )
            
            # Tendances
            for column in self.data['behavioral_trends'].columns:
                behavioral_fig.add_trace(
                    go.Scatter(
                        x=self.data['behavioral_trends'].index,
                        y=self.data['behavioral_trends'][column],
                        name=column,
                        mode='lines'
                    ),
                    row=1, col=2
                )
            
            # Changements
            for period in self.data['behavioral_changes'].columns:
                behavioral_fig.add_trace(
                    go.Bar(
                        x=self.data['behavioral_changes'].index,
                        y=self.data['behavioral_changes'][period],
                        name=period
                    ),
                    row=2, col=1
                )
            
            behavioral_fig.update_layout(
                title='Analyse des Facteurs Comportementaux',
                template='plotly_white',
                height=800,
                showlegend=True
            )
            
            # Métriques clés
            key_metrics = html.Div([
                html.Div([
                    html.H4('Performance du Modèle'),
                    html.P(f'MSE: {self.data["model_comparison"].loc[selected_model, "MSE"]:.2f}'),
                    html.P(f'RMSE: {self.data["model_comparison"].loc[selected_model, "RMSE"]:.2f}'),
                    html.P(f'MAE: {self.data["model_comparison"].loc[selected_model, "MAE"]:.2f}'),
                    html.P(f'R²: {self.data["model_comparison"].loc[selected_model, "R2"]:.2f}')
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4('Impact des Politiques'),
                    html.P(f'Réduction moyenne: {self.data["policy_impact"]["change_from_pre"].mean():.1f}%'),
                    html.P(f'Politique la plus efficace: {self.data["policy_impact"]["change_from_pre"].idxmin()}')
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),
                
                html.Div([
                    html.H4('Facteurs Comportementaux'),
                    html.P(f'Facteur le plus corrélé: {self.data["behavioral_correlations"].iloc[0].name}'),
                    html.P(f'Corrélation: {self.data["behavioral_correlations"].iloc[0]["correlation"]:.2f}')
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
            ])
            
            return emissions_fig, model_fig, policy_fig, behavioral_fig, key_metrics
    
    def run_server(self, debug=True, port=8050):
        """Lance le serveur du tableau de bord"""
        self.app.run_server(debug=debug, port=port)

def create_dashboard():
    """Fonction principale pour créer le tableau de bord"""
    dashboard = DashboardCreator()
    return dashboard

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dashboard = create_dashboard()
    dashboard.run_server() 