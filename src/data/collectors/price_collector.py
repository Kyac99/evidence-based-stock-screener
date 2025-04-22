"""
Module pour collecter les données de prix depuis Alpha Vantage.
"""

import os
import logging
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

from src.data.constants import PRICES_DIR, REQUEST_DELAY
from src.data.utils import (
    ensure_dir_exists, save_to_csv, load_from_csv, 
    save_to_pickle, load_from_pickle, rate_limited_request
)

# Configuration du logging
logger = logging.getLogger('stock_screener.data.prices')

class PriceCollector:
    """
    Classe pour collecter et gérer les données de prix depuis Alpha Vantage.
    """
    
    def __init__(self, api_key=None):
        """
        Initialise le collecteur de prix.
        
        Args:
            api_key (str, optional): Clé API Alpha Vantage. Si None, cherche dans les variables d'environnement.
        """
        # Récupération de la clé API depuis les variables d'environnement si non fournie
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.error("Clé API Alpha Vantage non fournie")
            raise ValueError("La clé API Alpha Vantage est requise. Spécifiez-la en paramètre ou dans ALPHA_VANTAGE_API_KEY")
        
        # Assurer que le répertoire des prix existe
        ensure_dir_exists(PRICES_DIR)
        
        # Base URL pour l'API Alpha Vantage
        self.base_url = "https://www.alphavantage.co/query"
        
        # Dictionnaire pour stocker les derniers temps d'appel API et respecter les limites
        self.last_api_call = 0
    
    @rate_limited_request
    def _make_api_request(self, params):
        """
        Fait une requête à l'API Alpha Vantage avec gestion des limites d'API.
        
        Args:
            params (dict): Paramètres de la requête
            
        Returns:
            dict: Données JSON de la réponse ou None en cas d'erreur
        """
        try:
            # Ajouter la clé API aux paramètres
            params['apikey'] = self.api_key
            
            # Faire la requête
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Convertir la réponse en JSON
            data = response.json()
            
            # Vérifier les erreurs ou limitations d'API
            if 'Error Message' in data:
                logger.error(f"Erreur Alpha Vantage: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"Note Alpha Vantage: {data['Note']}")
                # Attendre un peu plus longtemps en cas de limite d'API
                if 'call frequency' in data['Note']:
                    logger.warning("Limite d'API atteinte, attente prolongée...")
                    time.sleep(60)  # Attendre 60 secondes en cas de limite atteinte
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de la requête API: {e}")
            return None
    
    def get_daily_prices(self, ticker, outputsize='full'):
        """
        Récupère les données de prix quotidiennes pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            outputsize (str, optional): Taille de la sortie ('compact' ou 'full'). Par défaut 'full'.
                'compact' retourne les 100 derniers points de données.
                'full' retourne toutes les données disponibles.
        
        Returns:
            pd.DataFrame: DataFrame avec les données quotidiennes ou None en cas d'erreur
        """
        logger.info(f"Récupération des prix quotidiens pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': outputsize
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération des prix pour {ticker}")
            return None
        
        # Extraire la série temporelle
        try:
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                logger.error(f"Clé de série temporelle non trouvée dans la réponse pour {ticker}")
                return None
            
            time_series = data[time_series_key]
            
            # Convertir en DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Renommer les colonnes (enlever les préfixes '1. ', '2. ', etc.)
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convertir les types de données
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Ajouter le ticker comme colonne
            df['ticker'] = ticker
            
            # Convertir l'index en datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            logger.info(f"Récupéré {len(df)} points de données quotidiennes pour {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données pour {ticker}: {e}")
            return None
    
    def get_intraday_prices(self, ticker, interval='5min', outputsize='full'):
        """
        Récupère les données de prix intraday pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            interval (str, optional): Intervalle de temps ('1min', '5min', '15min', '30min', '60min'). Par défaut '5min'.
            outputsize (str, optional): Taille de la sortie ('compact' ou 'full'). Par défaut 'full'.
        
        Returns:
            pd.DataFrame: DataFrame avec les données intraday ou None en cas d'erreur
        """
        logger.info(f"Récupération des prix intraday pour {ticker} (intervalle: {interval})...")
        
        # Paramètres de la requête
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'outputsize': outputsize
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération des prix intraday pour {ticker}")
            return None
        
        # Extraire la série temporelle
        try:
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                logger.error(f"Clé de série temporelle non trouvée dans la réponse pour {ticker}")
                return None
            
            time_series = data[time_series_key]
            
            # Convertir en DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Renommer les colonnes (enlever les préfixes '1. ', '2. ', etc.)
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convertir les types de données
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Ajouter le ticker comme colonne
            df['ticker'] = ticker
            
            # Convertir l'index en datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            logger.info(f"Récupéré {len(df)} points de données intraday pour {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données intraday pour {ticker}: {e}")
            return None
    
    def get_weekly_prices(self, ticker):
        """
        Récupère les données de prix hebdomadaires pour un titre.
        
        Args:
            ticker (str): Symbole du titre
        
        Returns:
            pd.DataFrame: DataFrame avec les données hebdomadaires ou None en cas d'erreur
        """
        logger.info(f"Récupération des prix hebdomadaires pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'TIME_SERIES_WEEKLY',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération des prix hebdomadaires pour {ticker}")
            return None
        
        # Extraire la série temporelle
        try:
            time_series_key = 'Weekly Time Series'
            if time_series_key not in data:
                logger.error(f"Clé de série temporelle non trouvée dans la réponse pour {ticker}")
                return None
            
            time_series = data[time_series_key]
            
            # Convertir en DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Renommer les colonnes (enlever les préfixes '1. ', '2. ', etc.)
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convertir les types de données
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Ajouter le ticker comme colonne
            df['ticker'] = ticker
            
            # Convertir l'index en datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            logger.info(f"Récupéré {len(df)} points de données hebdomadaires pour {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données hebdomadaires pour {ticker}: {e}")
            return None
    
    def get_monthly_prices(self, ticker):
        """
        Récupère les données de prix mensuelles pour un titre.
        
        Args:
            ticker (str): Symbole du titre
        
        Returns:
            pd.DataFrame: DataFrame avec les données mensuelles ou None en cas d'erreur
        """
        logger.info(f"Récupération des prix mensuels pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'TIME_SERIES_MONTHLY',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération des prix mensuels pour {ticker}")
            return None
        
        # Extraire la série temporelle
        try:
            time_series_key = 'Monthly Time Series'
            if time_series_key not in data:
                logger.error(f"Clé de série temporelle non trouvée dans la réponse pour {ticker}")
                return None
            
            time_series = data[time_series_key]
            
            # Convertir en DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Renommer les colonnes (enlever les préfixes '1. ', '2. ', etc.)
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convertir les types de données
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Ajouter le ticker comme colonne
            df['ticker'] = ticker
            
            # Convertir l'index en datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            logger.info(f"Récupéré {len(df)} points de données mensuelles pour {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des données mensuelles pour {ticker}: {e}")
            return None
    
    def download_batch_prices(self, tickers, time_series='daily', save=True):
        """
        Télécharge les données de prix pour une liste de tickers.
        
        Args:
            tickers (list): Liste des symboles des titres
            time_series (str, optional): Type de série temporelle ('daily', 'weekly', 'monthly'). Par défaut 'daily'.
            save (bool, optional): Si True, sauvegarde les données pour chaque ticker. Par défaut True.
        
        Returns:
            dict: Dictionnaire avec les tickers comme clés et les DataFrames comme valeurs
        """
        price_data = {}
        errors = []
        
        logger.info(f"Téléchargement des prix {time_series} pour {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers):
            try:
                # Afficher la progression
                if (i + 1) % 5 == 0 or (i + 1) == len(tickers):
                    logger.info(f"Progression: {i+1}/{len(tickers)} tickers ({((i+1)/len(tickers))*100:.1f}%)")
                
                # Déterminer la fonction à utiliser selon le type de série
                if time_series == 'daily':
                    df = self.get_daily_prices(ticker)
                elif time_series == 'weekly':
                    df = self.get_weekly_prices(ticker)
                elif time_series == 'monthly':
                    df = self.get_monthly_prices(ticker)
                else:
                    logger.error(f"Type de série temporelle non supporté: {time_series}")
                    return {}
                
                if df is not None and not df.empty:
                    price_data[ticker] = df
                    
                    # Sauvegarder les données si demandé
                    if save:
                        self._save_ticker_price(ticker, df, time_series)
                else:
                    errors.append(ticker)
                    
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement des données pour {ticker}: {e}")
                errors.append(ticker)
        
        # Afficher le résumé
        logger.info(f"Téléchargement terminé. {len(price_data)}/{len(tickers)} tickers récupérés avec succès.")
        
        if errors:
            logger.warning(f"{len(errors)} tickers avec erreurs: {', '.join(errors[:10])}" + 
                         (f" et {len(errors)-10} autres" if len(errors) > 10 else ""))
        
        return price_data
    
    def _save_ticker_price(self, ticker, df, time_series='daily'):
        """
        Sauvegarde les données de prix d'un ticker dans un fichier CSV.
        
        Args:
            ticker (str): Symbole du titre
            df (pd.DataFrame): DataFrame contenant les données de prix
            time_series (str, optional): Type de série temporelle. Par défaut 'daily'.
        """
        try:
            # Créer le sous-répertoire pour le type de série temporelle
            series_dir = os.path.join(PRICES_DIR, time_series)
            ensure_dir_exists(series_dir)
            
            # Nettoyer le ticker pour le nom de fichier (remplacer les caractères spéciaux)
            clean_ticker = ticker.replace('.', '_').replace('-', '_')
            
            # Chemin du fichier
            filepath = os.path.join(series_dir, f"{clean_ticker}.csv")
            
            # Sauvegarder au format CSV
            save_to_csv(df, filepath)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données pour {ticker}: {e}")
    
    def load_ticker_price(self, ticker, time_series='daily'):
        """
        Charge les données de prix d'un ticker depuis un fichier CSV.
        
        Args:
            ticker (str): Symbole du titre
            time_series (str, optional): Type de série temporelle. Par défaut 'daily'.
        
        Returns:
            pd.DataFrame: DataFrame contenant les données de prix ou None en cas d'erreur
        """
        try:
            # Créer le sous-répertoire pour le type de série temporelle
            series_dir = os.path.join(PRICES_DIR, time_series)
            
            # Nettoyer le ticker pour le nom de fichier (remplacer les caractères spéciaux)
            clean_ticker = ticker.replace('.', '_').replace('-', '_')
            
            # Chemin du fichier
            filepath = os.path.join(series_dir, f"{clean_ticker}.csv")
            
            # Vérifier si le fichier existe
            if not os.path.exists(filepath):
                logger.warning(f"Fichier de prix introuvable pour {ticker}: {filepath}")
                return None
            
            # Charger depuis le CSV
            df = load_from_csv(filepath)
            
            # Convertir l'index en datetime si ce n'est pas déjà fait
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
                df.index.name = 'date'
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {ticker}: {e}")
            return None
    
    def save_all_prices(self, price_data, time_series='daily'):
        """
        Sauvegarde toutes les données de prix dans un seul fichier pickle.
        
        Args:
            price_data (dict): Dictionnaire avec les tickers comme clés et les DataFrames comme valeurs
            time_series (str, optional): Type de série temporelle. Par défaut 'daily'.
        """
        try:
            filepath = os.path.join(PRICES_DIR, f"all_{time_series}_prices.pkl")
            save_to_pickle(price_data, filepath)
            logger.info(f"Toutes les données de prix {time_series} sauvegardées dans {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de toutes les données: {e}")
    
    def load_all_prices(self, time_series='daily'):
        """
        Charge toutes les données de prix depuis un fichier pickle.
        
        Args:
            time_series (str, optional): Type de série temporelle. Par défaut 'daily'.
        
        Returns:
            dict: Dictionnaire avec les tickers comme clés et les DataFrames comme valeurs
        """
        try:
            filepath = os.path.join(PRICES_DIR, f"all_{time_series}_prices.pkl")
            
            if not os.path.exists(filepath):
                logger.warning(f"Fichier de prix global introuvable: {filepath}")
                return {}
            
            price_data = load_from_pickle(filepath)
            logger.info(f"Toutes les données de prix {time_series} chargées depuis {filepath}")
            
            return price_data
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement de toutes les données: {e}")
            return {}
    
    def process_prices(self, price_data):
        """
        Traite les données de prix pour calculer des métriques dérivées.
        
        Args:
            price_data (dict): Dictionnaire avec les tickers comme clés et les DataFrames comme valeurs
        
        Returns:
            dict: Dictionnaire avec les données de prix traitées
        """
        processed_data = {}
        
        for ticker, df in price_data.items():
            try:
                # Créer une copie pour éviter de modifier les données originales
                processed_df = df.copy()
                
                # Calcul des rendements journaliers
                processed_df['daily_return'] = processed_df['close'].pct_change()
                
                # Rendements cumulés sur différentes périodes
                for period in [5, 10, 21, 63, 126, 252]:  # ~1 semaine, 2 semaines, 1 mois, 3 mois, 6 mois, 1 an
                    if len(processed_df) >= period:
                        col_name = f'return_{period}d'
                        processed_df[col_name] = processed_df['close'].pct_change(periods=period)
                
                # Calcul de la volatilité (écart-type des rendements journaliers)
                # Sur 21 jours (environ 1 mois de trading)
                processed_df['volatility_21d'] = processed_df['daily_return'].rolling(window=21).std() * np.sqrt(252)  # Annualisée
                
                # Volume relatif (par rapport à la moyenne)
                processed_df['rel_volume'] = processed_df['volume'] / processed_df['volume'].rolling(window=21).mean()
                
                # Indicateurs techniques de base
                # Moyennes mobiles
                processed_df['sma_50'] = processed_df['close'].rolling(window=50).mean()
                processed_df['sma_200'] = processed_df['close'].rolling(window=200).mean()
                
                # Différence relative entre le prix et les moyennes mobiles
                processed_df['price_to_sma50'] = processed_df['close'] / processed_df['sma_50'] - 1
                processed_df['price_to_sma200'] = processed_df['close'] / processed_df['sma_200'] - 1
                
                # Momentum (ratio des prix actuels par rapport aux prix d'il y a N périodes)
                for period in [21, 63, 126, 252]:
                    if len(processed_df) >= period:
                        col_name = f'momentum_{period}d'
                        processed_df[col_name] = processed_df['close'] / processed_df['close'].shift(period) - 1
                
                processed_data[ticker] = processed_df
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement des données pour {ticker}: {e}")
        
        logger.info(f"Traitement terminé pour {len(processed_data)} tickers")
        return processed_data
