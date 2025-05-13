"""
Script principal pour collecter et traiter les données de différents indices.

Usage:
    python main.py --index SP500 --limit 10 --api_key YOUR_ALPHA_VANTAGE_API_KEY

Arguments:
    --index: Indice à traiter (SP500, NASDAQ, MSCIWORLD, EUROSTOXX)
    --limit: Nombre maximum de titres à traiter
    --api_key: Clé API Alpha Vantage (si non définie dans les variables d'environnement)
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import concurrent.futures

from src.data.constants import INDICES
from src.data.collectors.index_collector import IndexCollector
from src.data.collectors.price_collector import PriceCollector
from src.data.collectors.fundamental_collector import FundamentalCollector
from src.data.processors.data_processor import DataProcessor

# Configuration du logging
def setup_logging():
    """
    Configure le logging pour l'application.
    """
    # Créer le répertoire des logs s'il n'existe pas
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Nom du fichier de log avec date et heure
    log_file = os.path.join(log_dir, f"stock_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('stock_screener')

def parse_arguments():
    """
    Parse les arguments de ligne de commande.
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Collecte et traite les données de différents indices.')
    
    parser.add_argument('--index', type=str, default='SP500', choices=INDICES.keys(),
                       help='Indice à traiter (SP500, NASDAQ, MSCIWORLD, EUROSTOXX)')
    
    parser.add_argument('--limit', type=int, default=None,
                       help='Nombre maximum de titres à traiter')
    
    parser.add_argument('--api_key', type=str, default=None,
                       help='Clé API Alpha Vantage (si non définie dans les variables d\'environnement)')
    
    return parser.parse_args()

def collect_data(index_name, limit=None, api_key=None):
    """
    Collecte les données pour un indice donné.
    
    Args:
        index_name (str): Nom de l'indice
        limit (int, optional): Limite de titres à traiter. Defaults to None.
        api_key (str, optional): Clé API Alpha Vantage. Defaults to None.
    
    Returns:
        tuple: (price_data, fundamental_data) - Données collectées
    """
    logger = logging.getLogger('stock_screener.collect')
    logger.info(f"Début de la collecte des données pour l'indice {index_name}")
    
    # Initialiser les collecteurs
    index_collector = IndexCollector()
    price_collector = PriceCollector(api_key=api_key)
    fundamental_collector = FundamentalCollector(api_key=api_key)
    
    # 1. Récupérer les composantes de l'indice
    logger.info(f"Récupération des composantes de l'indice {index_name}")
    constituents = index_collector.get_index_constituents(index_name)
    
    # Limiter le nombre de titres si demandé
    if limit is not None and limit > 0 and limit < len(constituents):
        logger.info(f"Limitation à {limit} titres (sur {len(constituents)} disponibles)")
        constituents = constituents.iloc[:limit]
    
    # Extraire les tickers
    tickers = constituents['ticker'].tolist()
    logger.info(f"Traitement de {len(tickers)} titres")
    
    # 2. Récupérer les données de prix
    logger.info("Récupération des données de prix quotidiennes")
    price_data = {}
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 5 == 0 or (i + 1) == len(tickers):
            logger.info(f"Prix: {i+1}/{len(tickers)} titres traités")
        
        # Récupérer les données de prix pour ce ticker
        df = price_collector.get_daily_prices(ticker)
        
        # Ajouter au dictionnaire si les données sont valides
        if df is not None and not df.empty:
            price_data[ticker] = df
    
    logger.info(f"Données de prix récupérées pour {len(price_data)}/{len(tickers)} titres")
    
    # 3. Récupérer les données fondamentales
    logger.info("Récupération des données fondamentales")
    fundamental_data = {}
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 5 == 0 or (i + 1) == len(tickers):
            logger.info(f"Fondamentales: {i+1}/{len(tickers)} titres traités")
        
        # Récupérer les données fondamentales pour ce ticker
        ticker_data = fundamental_collector.collect_all_fundamentals(ticker)
        
        # Ajouter au dictionnaire si les données sont valides
        if ticker_data:
            fundamental_data[ticker] = ticker_data
    
    logger.info(f"Données fondamentales récupérées pour {len(fundamental_data)}/{len(tickers)} titres")
    
    return price_data, fundamental_data

def process_data(price_data, fundamental_data):
    """
    Traite les données collectées.
    
    Args:
        price_data (dict): Données de prix
        fundamental_data (dict): Données fondamentales
    
    Returns:
        tuple: (processed_price_data, processed_fundamental_data, combined_data) - Données traitées
    """
    logger = logging.getLogger('stock_screener.process')
    logger.info("Début du traitement des données")
    
    # Initialiser le processeur de données
    processor = DataProcessor()
    
    # 1. Traiter les données de prix
    logger.info("Traitement des données de prix")
    processed_price_data = processor.process_price_data(price_data)
    
    # 2. Traiter les données fondamentales
    logger.info("Traitement des données fondamentales")
    processed_fundamental_data = processor.process_fundamental_data(fundamental_data)
    
    # 3. Combiner les données de prix et fondamentales
    logger.info("Combinaison des données de prix et fondamentales")
    combined_data = processor.combine_price_and_fundamental(processed_price_data, processed_fundamental_data)
    
    logger.info("Traitement des données terminé")
    
    return processed_price_data, processed_fundamental_data, combined_data

def save_data(processor, price_data, fundamental_data, combined_data, index_name):
    """
    Sauvegarde les données traitées.
    
    Args:
        processor (DataProcessor): Processeur de données
        price_data (dict): Données de prix traitées
        fundamental_data (dict): Données fondamentales traitées
        combined_data (dict): Données combinées
        index_name (str): Nom de l'indice
    """
    logger = logging.getLogger('stock_screener.save')
    logger.info("Sauvegarde des données traitées")
    
    # Générer un timestamp pour les noms de fichiers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Sauvegarder les données de prix
    price_filename = f"{index_name}_prices_{timestamp}.pkl"
    processor.save_processed_data(price_data, 'price', price_filename)
    
    # 2. Sauvegarder les données fondamentales
    fundamental_filename = f"{index_name}_fundamentals_{timestamp}.pkl"
    processor.save_processed_data(fundamental_data, 'fundamental', fundamental_filename)
    
    # 3. Sauvegarder les données combinées
    combined_filename = f"{index_name}_combined_{timestamp}.pkl"
    processor.save_processed_data(combined_data, 'combined', combined_filename)
    
    logger.info("Sauvegarde des données terminée")

def create_summary(price_data, fundamental_data, combined_data, index_name):
    """
    Crée un résumé des données collectées et traitées.
    
    Args:
        price_data (dict): Données de prix traitées
        fundamental_data (dict): Données fondamentales traitées
        combined_data (dict): Données combinées
        index_name (str): Nom de l'indice
    """
    logger = logging.getLogger('stock_screener.summary')
    logger.info("Création d'un résumé des données")
    
    # Nombre de titres avec des données de prix
    price_count = len(price_data)
    
    # Nombre de titres avec des données fondamentales
    fundamental_count = len(fundamental_data.get('key_metrics', pd.DataFrame()))
    
    # Nombre de titres avec des données combinées
    combined_count = len(combined_data)
    
    # Afficher le résumé
    print("\n" + "="*80)
    print(f"RÉSUMÉ DES DONNÉES POUR L'INDICE {index_name}")
    print("="*80)
    print(f"Nombre de titres avec des données de prix: {price_count}")
    print(f"Nombre de titres avec des données fondamentales: {fundamental_count}")
    print(f"Nombre de titres avec des données combinées: {combined_count}")
    print("="*80 + "\n")
    
    # Si des données fondamentales sont disponibles, afficher quelques statistiques
    if fundamental_count > 0:
        key_metrics = fundamental_data.get('key_metrics')
        
        print("STATISTIQUES DES MÉTRIQUES FONDAMENTALES")
        print("-"*50)
        
        # Sélectionner quelques colonnes numériques pour les statistiques
        numeric_cols = ['MarketCapitalization', 'PERatio', 'DividendYield', 'EPS', 
                       'PriceToBookRatio', 'EVToRevenue', 'Beta']
        
        for col in numeric_cols:
            if col in key_metrics.columns:
                # Convertir en numérique pour être sûr
                values = pd.to_numeric(key_metrics[col], errors='coerce')
                
                # Calculer les statistiques
                mean = values.mean()
                median = values.median()
                min_val = values.min()
                max_val = values.max()
                
                print(f"{col}:")
                print(f"  - Moyenne: {mean:.2f}")
                print(f"  - Médiane: {median:.2f}")
                print(f"  - Min: {min_val:.2f}")
                print(f"  - Max: {max_val:.2f}")
                print()
    
    logger.info("Résumé des données créé")

def main():
    """
    Fonction principale.
    """
    # Configuration du logging
    logger = setup_logging()
    logger.info("Démarrage du programme")
    
    # Parser les arguments de ligne de commande
    args = parse_arguments()
    
    # Définir la clé API Alpha Vantage (priorité: argument, variable d'environnement)
    api_key = args.api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("Clé API Alpha Vantage non spécifiée. Utilisez --api_key ou définissez ALPHA_VANTAGE_API_KEY")
        sys.exit(1)
    
    # Collecter les données
    price_data, fundamental_data = collect_data(args.index, args.limit, api_key)
    
    # Traiter les données
    processed_price, processed_fundamental, combined_data = process_data(price_data, fundamental_data)
    
    # Sauvegarder les données
    processor = DataProcessor()
    save_data(processor, processed_price, processed_fundamental, combined_data, args.index)
    
    # Créer un résumé
    create_summary(processed_price, processed_fundamental, combined_data, args.index)
    
    logger.info("Programme terminé avec succès")

if __name__ == "__main__":
    main()
