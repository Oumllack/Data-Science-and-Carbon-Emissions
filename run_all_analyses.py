#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script principal pour exécuter l'ensemble des analyses
"""

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rapport/analysis.log'),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """Crée la structure de répertoires nécessaire"""
    directories = [
        'rapport/scientific',
        'rapport/political',
        'rapport/social',
        'rapport/visualization',
        'rapport/docs',
        'rapport/figures',
        'rapport/data'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Répertoire créé : {directory}")

def run_scientific_analysis():
    """Exécute les analyses scientifiques"""
    logging.info("Démarrage des analyses scientifiques...")
    
    # Import des modules d'analyse scientifique
    from scientific.model_comparison import compare_models
    from scientific.feature_importance import analyze_features
    from scientific.cross_validation import perform_cross_validation
    from scientific.sensitivity_analysis import analyze_sensitivity
    
    # Exécution des analyses
    models_comparison = compare_models()
    feature_importance = analyze_features()
    cv_results = perform_cross_validation()
    sensitivity_results = analyze_sensitivity()
    
    # Sauvegarde des résultats
    results = {
        'models_comparison': models_comparison,
        'feature_importance': feature_importance,
        'cv_results': cv_results,
        'sensitivity_results': sensitivity_results
    }
    pd.to_pickle(results, 'rapport/data/scientific_results.pkl')
    
    logging.info("Analyses scientifiques terminées")

def run_political_analysis():
    """Exécute les analyses politiques"""
    logging.info("Démarrage des analyses politiques...")
    
    # Import des modules d'analyse politique
    from political.policy_impact import analyze_policy_impact
    from political.scenario_analysis import analyze_scenarios
    from political.economic_indicators import analyze_economic_impact
    
    # Exécution des analyses
    policy_impact = analyze_policy_impact()
    scenarios = analyze_scenarios()
    economic_impact = analyze_economic_impact()
    
    # Sauvegarde des résultats
    results = {
        'policy_impact': policy_impact,
        'scenarios': scenarios,
        'economic_impact': economic_impact
    }
    pd.to_pickle(results, 'rapport/data/political_results.pkl')
    
    logging.info("Analyses politiques terminées")

def run_social_analysis():
    """Exécute les analyses sociales"""
    logging.info("Démarrage des analyses sociales...")
    
    # Import des modules d'analyse sociale
    from social.public_perception import analyze_public_perception
    from social.demographic_impact import analyze_demographic_impact
    from social.behavioral_factors import analyze_behavioral_factors
    
    # Exécution des analyses
    perception = analyze_public_perception()
    demographic = analyze_demographic_impact()
    behavioral = analyze_behavioral_factors()
    
    # Sauvegarde des résultats
    results = {
        'perception': perception,
        'demographic': demographic,
        'behavioral': behavioral
    }
    pd.to_pickle(results, 'rapport/data/social_results.pkl')
    
    logging.info("Analyses sociales terminées")

def generate_visualizations():
    """Génère les visualisations et le tableau de bord"""
    logging.info("Génération des visualisations...")
    
    from visualization.interactive_dashboard import create_dashboard
    from visualization.report_generator import generate_report
    from visualization.presentation import create_presentation
    
    # Création des visualisations
    dashboard = create_dashboard()
    report = generate_report()
    presentation = create_presentation()
    
    logging.info("Visualisations générées")

def main():
    """Fonction principale"""
    start_time = datetime.now()
    logging.info(f"Démarrage de l'analyse complète à {start_time}")
    
    try:
        # Configuration
        setup_directories()
        
        # Exécution des analyses
        run_scientific_analysis()
        run_political_analysis()
        run_social_analysis()
        
        # Génération des visualisations
        generate_visualizations()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Analyse complète terminée en {duration}")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution : {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 