import os
import requests
import pandas as pd
from pathlib import Path

def download_dataset():
    """Télécharge le dataset Gas Turbine CO and NOx Emission depuis UCI"""
    
    # Création du dossier data s'il n'existe pas
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # URL du dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00551/gt_2011.csv"
    
    # Chemin de destination
    output_file = data_dir / 'gt_2011.csv'
    
    print("Téléchargement du dataset en cours...")
    try:
        # Téléchargement du fichier
        response = requests.get(url)
        response.raise_for_status()  # Vérifie si le téléchargement a réussi
        
        # Sauvegarde du fichier
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        # Vérification du fichier téléchargé
        df = pd.read_csv(output_file)
        print(f"\nDataset téléchargé avec succès !")
        print(f"Emplacement : {output_file}")
        print(f"Nombre de lignes : {len(df)}")
        print(f"Nombre de colonnes : {len(df.columns)}")
        print("\nColonnes disponibles :")
        for col in df.columns:
            print(f"- {col}")
            
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement : {e}")
    except Exception as e:
        print(f"Erreur lors de la vérification du fichier : {e}")

if __name__ == "__main__":
    download_dataset() 