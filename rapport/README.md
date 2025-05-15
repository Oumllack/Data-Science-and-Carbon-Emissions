# Étude complète : Prédiction des émissions de CO2

## Structure de l'étude

### 1. Analyse Scientifique (`scientific/`)
- `model_comparison.py` : Comparaison approfondie des modèles (ARIMA, XGBoost, Prophet, LSTM)
- `feature_importance.py` : Analyse de l'importance des variables
- `cross_validation.py` : Validation croisée et robustesse des modèles
- `sensitivity_analysis.py` : Analyse de sensibilité des paramètres

### 2. Analyse Politique (`political/`)
- `policy_impact.py` : Analyse de l'impact des politiques climatiques
- `scenario_analysis.py` : Scénarios basés sur différentes politiques
- `economic_indicators.py` : Intégration des indicateurs économiques

### 3. Analyse Sociale (`social/`)
- `public_perception.py` : Analyse des tendances de consommation
- `demographic_impact.py` : Impact des changements démographiques
- `behavioral_factors.py` : Facteurs comportementaux

### 4. Visualisation et Reporting (`visualization/`)
- `interactive_dashboard.py` : Tableau de bord interactif
- `report_generator.py` : Génération automatique du rapport
- `presentation.py` : Création de slides de présentation

### 5. Documentation (`docs/`)
- `methodology.md` : Documentation méthodologique
- `results.md` : Résultats détaillés
- `recommendations.md` : Recommandations politiques et sociales

## Objectifs
1. **Scientifique** : Développer le modèle de prédiction le plus précis possible
2. **Politique** : Évaluer l'impact des politiques climatiques
3. **Social** : Comprendre les facteurs comportementaux et démographiques
4. **Pratique** : Fournir des recommandations actionnables

## Utilisation
1. Installer les dépendances : `pip install -r requirements.txt`
2. Exécuter les analyses : `python run_all_analyses.py`
3. Générer le rapport : `python generate_report.py`
4. Visualiser les résultats : `python launch_dashboard.py` 