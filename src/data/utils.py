"""
Fonctions utilitaires pour le module de données.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.constants import REQUEST_DELAY

# Configuration du logging
logger = logging.getLogger('stock_screener.data.utils')

def ensure_dir_exists(directory):
    """
    S'assure que le répertoire existe, le crée sinon.
    
    Args:
        directory (str): Chemin du répertoire à vérifier/créer
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Répertoire créé: {directory}")

def rate_limited_request(func):
    """
    Décorateur pour limiter le taux de requêtes API.
    Introduit un délai entre les appels pour respecter les limites d'API.
    
    Args:
        func: La fonction à décorer
        
    Returns:
        La fonction décorée avec limitation de taux
    """
    last_call_time = [0]  # Utilisation d'une liste pour stocker le temps entre appels
    
    def wrapper(*args, **kwargs):
        # Calculer le temps écoulé depuis le dernier appel
        current_time = time.time()
        time_since_last_call = current_time - last_call_time[0]
        
        # Si le temps écoulé est inférieur au délai requis, attendre
        if time_since_last_call < REQUEST_DELAY:
            sleep_time = REQUEST_DELAY - time_since_last_call
            logger.debug(f"Délai d'attente de {sleep_time:.2f}s pour respecter les limites API")
            time.sleep(sleep_time)
        
        # Mettre à jour le temps du dernier appel et exécuter la fonction
        last_call_time[0] = time.time()
        return func(*args, **kwargs)
    
    return wrapper

def format_tickers(tickers, remove_suffix=True):
    """
    Formate une liste de tickers pour s'assurer de leur compatibilité avec Alpha Vantage.
    
    Args:
        tickers (list): Liste des tickers à formater
        remove_suffix (bool): Si True, supprime les suffixes de marché (.PA, .L, etc.)
        
    Returns:
        list: Liste des tickers formatés
    """
    formatted_tickers = []
    
    for ticker in tickers:
        # Remplacer les points par des tirets (utilisé dans certains indices)
        formatted = ticker.replace('.', '-')
        
        # Supprimer les suffixes de marché si demandé
        if remove_suffix and '.' in ticker:
            formatted = ticker.split('.')[0]
        
        formatted_tickers.append(formatted)
    
    return formatted_tickers

def save_to_csv(df, filepath, index=True):
    """
    Sauvegarde un DataFrame au format CSV.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
        filepath (str): Chemin du fichier CSV
        index (bool): Si True, inclut l'index dans le CSV
    """
    try:
        # Créer le répertoire parent si nécessaire
        parent_dir = os.path.dirname(filepath)
        ensure_dir_exists(parent_dir)
        
        # Sauvegarder au format CSV
        df.to_csv(filepath, index=index)
        logger.info(f"DataFrame sauvegardé dans {filepath}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde dans {filepath}: {e}")

def load_from_csv(filepath, index_col=0):
    """
    Charge un DataFrame depuis un fichier CSV.
    
    Args:
        filepath (str): Chemin du fichier CSV
        index_col (int or str): Index de la colonne à utiliser comme index
        
    Returns:
        pd.DataFrame: DataFrame chargé ou DataFrame vide en cas d'erreur
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Le fichier {filepath} n'existe pas")
            return pd.DataFrame()
        
        df = pd.DataFrame()
        df = pd.read_csv(filepath, index_col=index_col)
        logger.info(f"DataFrame chargé depuis {filepath}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement depuis {filepath}: {e}")
        return pd.DataFrame()

def save_to_pickle(data, filepath):
    """
    Sauvegarde des données au format pickle.
    
    Args:
        data: Données à sauvegarder (DataFrame, dictionnaire, etc.)
        filepath (str): Chemin du fichier pickle
    """
    try:
        # Créer le répertoire parent si nécessaire
        parent_dir = os.path.dirname(filepath)
        ensure_dir_exists(parent_dir)
        
        # Sauvegarder au format pickle
        pd.to_pickle(data, filepath)
        logger.info(f"Données sauvegardées dans {filepath}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde dans {filepath}: {e}")

def load_from_pickle(filepath):
    """
    Charge des données depuis un fichier pickle.
    
    Args:
        filepath (str): Chemin du fichier pickle
        
    Returns:
        Données chargées ou None en cas d'erreur
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Le fichier {filepath} n'existe pas")
            return None
        
        data = pd.read_pickle(filepath)
        logger.info(f"Données chargées depuis {filepath}")
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement depuis {filepath}: {e}")
        return None

def date_to_str(date):
    """
    Convertit une date en chaîne de caractères au format YYYY-MM-DD.
    
    Args:
        date (datetime): Date à convertir
        
    Returns:
        str: Date au format YYYY-MM-DD
    """
    if isinstance(date, str):
        return date
    return date.strftime("%Y-%m-%d")

def safe_float_convert(value):
    """
    Convertit une valeur en float de manière sécurisée.
    
    Args:
        value: Valeur à convertir
        
    Returns:
        float: Valeur convertie ou NaN en cas d'erreur
    """
    try:
        if pd.isna(value) or value is None or value == '':
            return np.nan
        return float(value)
    except (ValueError, TypeError):
        return np.nan
