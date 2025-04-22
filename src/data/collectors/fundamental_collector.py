"""
Module pour collecter les données fondamentales depuis Alpha Vantage.
"""

import os
import logging
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime

from src.data.constants import FUNDAMENTALS_DIR, REQUEST_DELAY
from src.data.utils import (
    ensure_dir_exists, save_to_csv, load_from_csv, 
    save_to_pickle, load_from_pickle, rate_limited_request,
    safe_float_convert
)

# Configuration du logging
logger = logging.getLogger('stock_screener.data.fundamentals')

class FundamentalCollector:
    """
    Classe pour collecter et gérer les données fondamentales depuis Alpha Vantage.
    """
    
    def __init__(self, api_key=None):
        """
        Initialise le collecteur de données fondamentales.
        
        Args:
            api_key (str, optional): Clé API Alpha Vantage. Si None, cherche dans les variables d'environnement.
        """
        # Récupération de la clé API depuis les variables d'environnement si non fournie
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.error("Clé API Alpha Vantage non fournie")
            raise ValueError("La clé API Alpha Vantage est requise. Spécifiez-la en paramètre ou dans ALPHA_VANTAGE_API_KEY")
        
        # Assurer que les répertoires existent
        ensure_dir_exists(FUNDAMENTALS_DIR)
        ensure_dir_exists(os.path.join(FUNDAMENTALS_DIR, 'overview'))
        ensure_dir_exists(os.path.join(FUNDAMENTALS_DIR, 'income'))
        ensure_dir_exists(os.path.join(FUNDAMENTALS_DIR, 'balance'))
        ensure_dir_exists(os.path.join(FUNDAMENTALS_DIR, 'cashflow'))
        ensure_dir_exists(os.path.join(FUNDAMENTALS_DIR, 'earnings'))
        
        # Base URL pour l'API Alpha Vantage
        self.base_url = "https://www.alphavantage.co/query"
    
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
    
    def get_company_overview(self, ticker):
        """
        Récupère les données d'aperçu de l'entreprise pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            
        Returns:
            dict: Dictionnaire avec les données d'aperçu ou None en cas d'erreur
        """
        logger.info(f"Récupération de l'aperçu pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération de l'aperçu pour {ticker}")
            return None
        
        # Vérifier que la réponse contient des données
        if not data or len(data) <= 1:  # Parfois l'API retourne juste {'Note': '...'}
            logger.warning(f"Pas de données d'aperçu disponibles pour {ticker}")
            return None
        
        # Ajouter la date de récupération
        data['RetrievalDate'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Aperçu récupéré pour {ticker}")
        
        return data
    
    def get_income_statement(self, ticker):
        """
        Récupère les données du compte de résultat pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            
        Returns:
            dict: Dictionnaire avec les données du compte de résultat ou None en cas d'erreur
        """
        logger.info(f"Récupération du compte de résultat pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération du compte de résultat pour {ticker}")
            return None
        
        # Vérifier que la réponse contient des données
        if 'annualReports' not in data and 'quarterlyReports' not in data:
            logger.warning(f"Pas de données de compte de résultat disponibles pour {ticker}")
            return None
        
        # Ajouter la date de récupération
        data['RetrievalDate'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Compte de résultat récupéré pour {ticker}")
        
        return data
    
    def get_balance_sheet(self, ticker):
        """
        Récupère les données du bilan pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            
        Returns:
            dict: Dictionnaire avec les données du bilan ou None en cas d'erreur
        """
        logger.info(f"Récupération du bilan pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération du bilan pour {ticker}")
            return None
        
        # Vérifier que la réponse contient des données
        if 'annualReports' not in data and 'quarterlyReports' not in data:
            logger.warning(f"Pas de données de bilan disponibles pour {ticker}")
            return None
        
        # Ajouter la date de récupération
        data['RetrievalDate'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Bilan récupéré pour {ticker}")
        
        return data
    
    def get_cash_flow(self, ticker):
        """
        Récupère les données du tableau de flux de trésorerie pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            
        Returns:
            dict: Dictionnaire avec les données des flux de trésorerie ou None en cas d'erreur
        """
        logger.info(f"Récupération des flux de trésorerie pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'CASH_FLOW',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération des flux de trésorerie pour {ticker}")
            return None
        
        # Vérifier que la réponse contient des données
        if 'annualReports' not in data and 'quarterlyReports' not in data:
            logger.warning(f"Pas de données de flux de trésorerie disponibles pour {ticker}")
            return None
        
        # Ajouter la date de récupération
        data['RetrievalDate'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Flux de trésorerie récupérés pour {ticker}")
        
        return data
    
    def get_earnings(self, ticker):
        """
        Récupère les données de bénéfices pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            
        Returns:
            dict: Dictionnaire avec les données de bénéfices ou None en cas d'erreur
        """
        logger.info(f"Récupération des bénéfices pour {ticker}...")
        
        # Paramètres de la requête
        params = {
            'function': 'EARNINGS',
            'symbol': ticker
        }
        
        # Faire la requête API
        data = self._make_api_request(params)
        
        if not data:
            logger.error(f"Échec de récupération des bénéfices pour {ticker}")
            return None
        
        # Vérifier que la réponse contient des données
        if 'annualEarnings' not in data and 'quarterlyEarnings' not in data:
            logger.warning(f"Pas de données de bénéfices disponibles pour {ticker}")
            return None
        
        # Ajouter la date de récupération
        data['RetrievalDate'] = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Bénéfices récupérés pour {ticker}")
        
        return data
    
    def save_fundamental_data(self, ticker, data_type, data):
        """
        Sauvegarde les données fondamentales dans un fichier JSON.
        
        Args:
            ticker (str): Symbole du titre
            data_type (str): Type de données ('overview', 'income', 'balance', 'cashflow', 'earnings')
            data (dict): Données à sauvegarder
        """
        if not data:
            logger.warning(f"Pas de données à sauvegarder pour {ticker} ({data_type})")
            return
        
        try:
            # Nettoyer le ticker pour le nom de fichier
            clean_ticker = ticker.replace('.', '_').replace('-', '_')
            
            # Répertoire pour ce type de données
            data_dir = os.path.join(FUNDAMENTALS_DIR, data_type)
            ensure_dir_exists(data_dir)
            
            # Chemin du fichier
            filepath = os.path.join(data_dir, f"{clean_ticker}.json")
            
            # Sauvegarder au format JSON
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Données {data_type} sauvegardées pour {ticker} dans {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données {data_type} pour {ticker}: {e}")
    
    def load_fundamental_data(self, ticker, data_type):
        """
        Charge les données fondamentales depuis un fichier JSON.
        
        Args:
            ticker (str): Symbole du titre
            data_type (str): Type de données ('overview', 'income', 'balance', 'cashflow', 'earnings')
            
        Returns:
            dict: Données chargées ou None en cas d'erreur
        """
        try:
            # Nettoyer le ticker pour le nom de fichier
            clean_ticker = ticker.replace('.', '_').replace('-', '_')
            
            # Répertoire pour ce type de données
            data_dir = os.path.join(FUNDAMENTALS_DIR, data_type)
            
            # Chemin du fichier
            filepath = os.path.join(data_dir, f"{clean_ticker}.json")
            
            # Vérifier si le fichier existe
            if not os.path.exists(filepath):
                logger.warning(f"Fichier de données {data_type} introuvable pour {ticker}: {filepath}")
                return None
            
            # Charger depuis le fichier JSON
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Données {data_type} chargées pour {ticker} depuis {filepath}")
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données {data_type} pour {ticker}: {e}")
            return None
    
    def collect_all_fundamentals(self, ticker, save=True):
        """
        Collecte toutes les données fondamentales pour un titre.
        
        Args:
            ticker (str): Symbole du titre
            save (bool, optional): Si True, sauvegarde les données. Par défaut True.
            
        Returns:
            dict: Dictionnaire avec tous les types de données fondamentales
        """
        logger.info(f"Collecte de toutes les données fondamentales pour {ticker}...")
        
        all_data = {}
        
        # Récupérer l'aperçu de l'entreprise
        overview = self.get_company_overview(ticker)
        if overview:
            all_data['overview'] = overview
            if save:
                self.save_fundamental_data(ticker, 'overview', overview)
        
        # Récupérer le compte de résultat
        income = self.get_income_statement(ticker)
        if income:
            all_data['income'] = income
            if save:
                self.save_fundamental_data(ticker, 'income', income)
        
        # Récupérer le bilan
        balance = self.get_balance_sheet(ticker)
        if balance:
            all_data['balance'] = balance
            if save:
                self.save_fundamental_data(ticker, 'balance', balance)
        
        # Récupérer les flux de trésorerie
        cashflow = self.get_cash_flow(ticker)
        if cashflow:
            all_data['cashflow'] = cashflow
            if save:
                self.save_fundamental_data(ticker, 'cashflow', cashflow)
        
        # Récupérer les bénéfices
        earnings = self.get_earnings(ticker)
        if earnings:
            all_data['earnings'] = earnings
            if save:
                self.save_fundamental_data(ticker, 'earnings', earnings)
        
        return all_data
    
    def collect_batch_fundamentals(self, tickers, save=True):
        """
        Collecte les données fondamentales pour une liste de tickers.
        
        Args:
            tickers (list): Liste des symboles des titres
            save (bool, optional): Si True, sauvegarde les données. Par défaut True.
            
        Returns:
            dict: Dictionnaire avec les tickers comme clés et les données fondamentales comme valeurs
        """
        all_data = {}
        errors = []
        
        logger.info(f"Collecte des données fondamentales pour {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers):
            try:
                # Afficher la progression
                if (i + 1) % 5 == 0 or (i + 1) == len(tickers):
                    logger.info(f"Progression: {i+1}/{len(tickers)} tickers ({((i+1)/len(tickers))*100:.1f}%)")
                
                # Collecter toutes les données fondamentales
                ticker_data = self.collect_all_fundamentals(ticker, save=save)
                
                if ticker_data:
                    all_data[ticker] = ticker_data
                else:
                    errors.append(ticker)
                    
            except Exception as e:
                logger.error(f"Erreur lors de la collecte des données pour {ticker}: {e}")
                errors.append(ticker)
        
        # Afficher le résumé
        logger.info(f"Collecte terminée. {len(all_data)}/{len(tickers)} tickers récupérés avec succès.")
        
        if errors:
            logger.warning(f"{len(errors)} tickers avec erreurs: {', '.join(errors[:10])}" + 
                         (f" et {len(errors)-10} autres" if len(errors) > 10 else ""))
        
        return all_data
    
    def convert_to_dataframe(self, fundamental_data, data_type, report_type='annual'):
        """
        Convertit les données fondamentales en DataFrame.
        
        Args:
            fundamental_data (dict): Données fondamentales pour un ticker
            data_type (str): Type de données ('income', 'balance', 'cashflow', 'earnings')
            report_type (str): Type de rapport ('annual' ou 'quarterly'). Par défaut 'annual'.
            
        Returns:
            pd.DataFrame: DataFrame formaté ou None en cas d'erreur
        """
        try:
            if not fundamental_data or data_type not in fundamental_data:
                logger.warning(f"Pas de données {data_type} disponibles")
                return None
            
            data = fundamental_data[data_type]
            
            # Déterminer la clé pour le type de rapport
            if data_type == 'earnings':
                reports_key = 'annualEarnings' if report_type == 'annual' else 'quarterlyEarnings'
            else:
                reports_key = 'annualReports' if report_type == 'annual' else 'quarterlyReports'
            
            if reports_key not in data:
                logger.warning(f"Pas de rapports {report_type} disponibles pour {data_type}")
                return None
            
            # Extraire les rapports
            reports = data[reports_key]
            
            # Convertir en DataFrame
            df = pd.DataFrame(reports)
            
            # Ajouter le symbole si disponible
            if 'symbol' in data:
                df['symbol'] = data['symbol']
            
            # Convertir les colonnes numériques en float
            for col in df.columns:
                if col not in ['fiscalDateEnding', 'reportedDate', 'reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage', 'symbol']:
                    df[col] = df[col].apply(safe_float_convert)
            
            # Si c'est un DataFrame de bénéfices, convertir ces colonnes spécifiques aussi
            if data_type == 'earnings':
                for col in ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']:
                    if col in df.columns:
                        df[col] = df[col].apply(safe_float_convert)
            
            # Convertir les dates en datetime
            if 'fiscalDateEnding' in df.columns:
                df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
                # Définir comme index
                df.set_index('fiscalDateEnding', inplace=True)
                df.sort_index(ascending=False, inplace=True)
            
            if 'reportedDate' in df.columns:
                df['reportedDate'] = pd.to_datetime(df['reportedDate'])
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la conversion en DataFrame: {e}")
            return None
    
    def extract_key_metrics(self, ticker):
        """
        Extrait les métriques clés pour un titre à partir des données fondamentales.
        
        Args:
            ticker (str): Symbole du titre
            
        Returns:
            dict: Dictionnaire avec les métriques clés ou None en cas d'erreur
        """
        try:
            # Charger les données fondamentales
            overview = self.load_fundamental_data(ticker, 'overview')
            income = self.load_fundamental_data(ticker, 'income')
            balance = self.load_fundamental_data(ticker, 'balance')
            cashflow = self.load_fundamental_data(ticker, 'cashflow')
            earnings = self.load_fundamental_data(ticker, 'earnings')
            
            # Si aucune donnée n'est disponible, retourner None
            if not any([overview, income, balance, cashflow, earnings]):
                logger.warning(f"Aucune donnée fondamentale disponible pour {ticker}")
                return None
            
            # Dictionnaire pour stocker les métriques
            metrics = {'ticker': ticker}
            
            # Extraire les métriques de l'aperçu
            if overview:
                for key in ['MarketCapitalization', 'EBITDA', 'PERatio', 'PEGRatio', 'BookValue', 
                          'DividendPerShare', 'DividendYield', 'EPS', 'RevenuePerShareTTM', 
                          'ProfitMargin', 'OperatingMarginTTM', 'ReturnOnAssetsTTM', 'ReturnOnEquityTTM',
                          'RevenueTTM', 'GrossProfitTTM', 'QuarterlyEarningsGrowthYOY',
                          'QuarterlyRevenueGrowthYOY', 'AnalystTargetPrice', 'TrailingPE',
                          'ForwardPE', 'PriceToSalesRatioTTM', 'PriceToBookRatio', 'EVToRevenue',
                          'EVToEBITDA', 'Beta', '52WeekHigh', '52WeekLow']:
                    
                    if key in overview:
                        metrics[key] = safe_float_convert(overview[key])
                
                # Ajouter des métriques non numériques
                for key in ['Name', 'Industry', 'Sector', 'Currency', 'Country', 'Exchange']:
                    if key in overview:
                        metrics[key] = overview[key]
            
            # Extraire les métriques du dernier rapport annuel
            if income and 'annualReports' in income and len(income['annualReports']) > 0:
                latest_income = income['annualReports'][0]
                for key in ['TotalRevenue', 'GrossProfit', 'OperatingIncome', 'NetIncome',
                          'EBITDA', 'Research-Development']:
                    if key in latest_income:
                        metrics[f'Latest{key}'] = safe_float_convert(latest_income[key])
            
            if balance and 'annualReports' in balance and len(balance['annualReports']) > 0:
                latest_balance = balance['annualReports'][0]
                for key in ['TotalAssets', 'TotalCurrentAssets', 'CashAndCashEquivalentsAtCarryingValue',
                          'TotalLiabilities', 'TotalCurrentLiabilities', 'TotalShareholderEquity',
                          'RetainedEarnings', 'LongTermDebt']:
                    if key in latest_balance:
                        metrics[f'Latest{key}'] = safe_float_convert(latest_balance[key])
            
            if cashflow and 'annualReports' in cashflow and len(cashflow['annualReports']) > 0:
                latest_cashflow = cashflow['annualReports'][0]
                for key in ['OperatingCashflow', 'CashflowFromInvestment', 'CashflowFromFinancing',
                          'CapitalExpenditures', 'DividendPayout', 'NetIncome']:
                    if key in latest_cashflow:
                        metrics[f'Latest{key}'] = safe_float_convert(latest_cashflow[key])
            
            # Calculer des ratios supplémentaires si les données nécessaires sont disponibles
            if 'LatestTotalAssets' in metrics and 'LatestTotalLiabilities' in metrics and metrics['LatestTotalAssets'] > 0:
                metrics['DebtToAssets'] = metrics['LatestTotalLiabilities'] / metrics['LatestTotalAssets']
            
            if 'LatestTotalLiabilities' in metrics and 'LatestTotalShareholderEquity' in metrics and metrics['LatestTotalShareholderEquity'] > 0:
                metrics['DebtToEquity'] = metrics['LatestTotalLiabilities'] / metrics['LatestTotalShareholderEquity']
            
            if 'LatestTotalCurrentAssets' in metrics and 'LatestTotalCurrentLiabilities' in metrics and metrics['LatestTotalCurrentLiabilities'] > 0:
                metrics['CurrentRatio'] = metrics['LatestTotalCurrentAssets'] / metrics['LatestTotalCurrentLiabilities']
            
            if 'LatestCashAndCashEquivalentsAtCarryingValue' in metrics and 'LatestTotalCurrentLiabilities' in metrics and metrics['LatestTotalCurrentLiabilities'] > 0:
                metrics['QuickRatio'] = metrics['LatestCashAndCashEquivalentsAtCarryingValue'] / metrics['LatestTotalCurrentLiabilities']
            
            if 'LatestOperatingCashflow' in metrics and 'LatestCapitalExpenditures' in metrics:
                metrics['FreeCashFlow'] = metrics['LatestOperatingCashflow'] - metrics['LatestCapitalExpenditures']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des métriques clés pour {ticker}: {e}")
            return None
    
    def create_fundamental_dataset(self, tickers):
        """
        Crée un dataset avec les métriques fondamentales clés pour tous les tickers.
        
        Args:
            tickers (list): Liste des symboles des titres
            
        Returns:
            pd.DataFrame: DataFrame avec les métriques fondamentales pour tous les tickers
        """
        all_metrics = []
        
        logger.info(f"Création d'un dataset fondamental pour {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers):
            try:
                # Afficher la progression
                if (i + 1) % 10 == 0 or (i + 1) == len(tickers):
                    logger.info(f"Progression: {i+1}/{len(tickers)} tickers ({((i+1)/len(tickers))*100:.1f}%)")
                
                # Extraire les métriques clés
                metrics = self.extract_key_metrics(ticker)
                
                if metrics:
                    all_metrics.append(metrics)
                    
            except Exception as e:
                logger.error(f"Erreur lors de la création des métriques pour {ticker}: {e}")
        
        # Créer un DataFrame à partir de toutes les métriques
        if all_metrics:
            df = pd.DataFrame(all_metrics)
            
            # Définir le ticker comme index
            if 'ticker' in df.columns:
                df.set_index('ticker', inplace=True)
            
            logger.info(f"Dataset fondamental créé pour {len(df)} tickers")
            
            return df
        else:
            logger.warning("Aucune métrique trouvée pour créer le dataset")
            return pd.DataFrame()
