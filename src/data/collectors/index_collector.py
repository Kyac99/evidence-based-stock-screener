"""
Module pour collecter les composantes des indices boursiers.
"""

import os
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data.constants import (
    INDICES, INDICES_DIR, INDEX_URLS, INDEX_SCRAPE_CONFIG
)
from src.data.utils import ensure_dir_exists, save_to_csv, load_from_csv

# Configuration du logging
logger = logging.getLogger('stock_screener.data.indices')

class IndexCollector:
    """
    Classe pour collecter et gérer les composantes des indices boursiers.
    """
    
    def __init__(self):
        """
        Initialise le collecteur d'indices.
        """
        # Assurer que le répertoire des indices existe
        ensure_dir_exists(INDICES_DIR)
        
        # Noms des fichiers pour chaque indice
        self.index_files = {
            'SP500': os.path.join(INDICES_DIR, 'sp500_constituents.csv'),
            'NASDAQ': os.path.join(INDICES_DIR, 'nasdaq_constituents.csv'),
            'MSCIWORLD': os.path.join(INDICES_DIR, 'msci_world_constituents.csv'),
            'EUROSTOXX': os.path.join(INDICES_DIR, 'eurostoxx_constituents.csv')
        }
    
    def scrape_sp500(self):
        """
        Récupère la liste des composantes du S&P 500 depuis Wikipedia.
        
        Returns:
            pd.DataFrame: DataFrame contenant les composantes du S&P 500
        """
        try:
            logger.info("Récupération des composantes du S&P 500...")
            url = INDEX_URLS['SP500']
            config = INDEX_SCRAPE_CONFIG['SP500']
            
            # Récupérer les tables HTML
            tables = pd.read_html(url)
            
            # Obtenir la table des composantes
            df = tables[config['table_index']]
            
            # Sélectionner et renommer les colonnes importantes
            constituents = df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
            constituents = constituents.rename(columns={
                'Symbol': 'ticker',
                'Security': 'name',
                'GICS Sector': 'sector',
                'GICS Sub-Industry': 'industry'
            })
            
            # Nettoyer les symboles (remplacer les points par des tirets)
            constituents['ticker'] = constituents['ticker'].str.replace('.', '-')
            
            logger.info(f"Récupéré {len(constituents)} composantes du S&P 500")
            return constituents
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des composantes du S&P 500: {e}")
            return pd.DataFrame()
    
    def scrape_nasdaq(self):
        """
        Récupère la liste des composantes du NASDAQ-100 depuis Wikipedia.
        
        Returns:
            pd.DataFrame: DataFrame contenant les composantes du NASDAQ-100
        """
        try:
            logger.info("Récupération des composantes du NASDAQ-100...")
            url = INDEX_URLS['NASDAQ100']
            config = INDEX_SCRAPE_CONFIG['NASDAQ100']
            
            # Récupérer les tables HTML
            tables = pd.read_html(url)
            
            # Obtenir la table des composantes
            df = tables[config['table_index']]
            
            # Sélectionner et renommer les colonnes importantes
            constituents = df[[config['symbol_column'], 'Company', 'GICS Sector', 'GICS Sub-Industry']]
            constituents = constituents.rename(columns={
                config['symbol_column']: 'ticker',
                'Company': 'name',
                'GICS Sector': 'sector',
                'GICS Sub-Industry': 'industry'
            })
            
            logger.info(f"Récupéré {len(constituents)} composantes du NASDAQ-100")
            return constituents
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des composantes du NASDAQ-100: {e}")
            
            # Tentative alternative: récupérer simplement les tickers NASDAQ
            try:
                logger.info("Tentative alternative: récupération des principaux tickers NASDAQ...")
                
                # URL pour un listing des principales entreprises du NASDAQ
                url = "https://www.nasdaq.com/market-activity/stocks/screener"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Cette approche est simplifiée et pourrait nécessiter des ajustements
                # selon la structure réelle du site NASDAQ
                ticker_elements = soup.select('a[href^="/market-activity/stocks/"]')
                tickers = []
                
                for element in ticker_elements:
                    ticker = element.get_text().strip()
                    if ticker and len(ticker) < 6:  # Filtrer les tickers valides
                        tickers.append(ticker)
                
                # Supprimer les doublons
                tickers = list(set(tickers))
                
                # Créer un DataFrame simple
                constituents = pd.DataFrame({
                    'ticker': tickers,
                    'name': [''] * len(tickers),
                    'sector': [''] * len(tickers),
                    'industry': [''] * len(tickers)
                })
                
                logger.info(f"Récupéré {len(constituents)} tickers NASDAQ (méthode alternative)")
                return constituents
                
            except Exception as alt_e:
                logger.error(f"Échec de la méthode alternative pour NASDAQ: {alt_e}")
                return pd.DataFrame()
    
    def scrape_eurostoxx(self):
        """
        Récupère la liste des composantes de l'EURO STOXX 50 depuis Wikipedia.
        
        Returns:
            pd.DataFrame: DataFrame contenant les composantes de l'EURO STOXX 50
        """
        try:
            logger.info("Récupération des composantes de l'EURO STOXX 50...")
            url = INDEX_URLS['EUROSTOXX']
            config = INDEX_SCRAPE_CONFIG['EUROSTOXX']
            
            # Récupérer les tables HTML
            tables = pd.read_html(url)
            
            # Obtenir la table des composantes
            df = tables[config['table_index']]
            
            # Le format de la table peut varier, nous adaptons
            # Chercher les colonnes contenant le ticker et le nom
            columns = df.columns.tolist()
            ticker_col = next((col for col in columns if 'Ticker' in col or 'Symbol' in col), None)
            name_col = next((col for col in columns if 'Company' in col or 'Name' in col), None)
            sector_col = next((col for col in columns if 'Sector' in col or 'Industry' in col), None)
            
            if not ticker_col or not name_col:
                logger.warning("Colonnes requises non trouvées dans la table EURO STOXX 50")
                # Créer une liste manuelle si les colonnes ne sont pas trouvées
                return self._get_manual_eurostoxx()
            
            # Sélectionner et renommer les colonnes importantes
            cols_to_select = [ticker_col, name_col]
            if sector_col:
                cols_to_select.append(sector_col)
            
            constituents = df[cols_to_select].copy()
            
            # Renommer les colonnes
            column_mapping = {ticker_col: 'ticker', name_col: 'name'}
            if sector_col:
                column_mapping[sector_col] = 'sector'
            
            constituents = constituents.rename(columns=column_mapping)
            
            # Ajouter les colonnes manquantes si nécessaire
            if 'sector' not in constituents.columns:
                constituents['sector'] = ''
            constituents['industry'] = ''
            
            # Ajouter les suffixes de marché
            constituents = self._add_eurostoxx_market_suffixes(constituents)
            
            logger.info(f"Récupéré {len(constituents)} composantes de l'EURO STOXX 50")
            return constituents
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des composantes de l'EURO STOXX 50: {e}")
            return self._get_manual_eurostoxx()
    
    def _get_manual_eurostoxx(self):
        """
        Retourne une liste manuelle des composantes de l'EURO STOXX 50.
        Utilisé comme solution de secours si le scraping échoue.
        
        Returns:
            pd.DataFrame: DataFrame contenant les composantes manuelles de l'EURO STOXX 50
        """
        logger.info("Utilisation de la liste manuelle des composantes de l'EURO STOXX 50")
        
        # Liste manuelle des composantes (mise à jour: avril 2025)
        eurostoxx_data = [
            {"ticker": "ADS.DE", "name": "Adidas", "sector": "Consumer Cyclical", "industry": "Footwear & Accessories"},
            {"ticker": "AIR.PA", "name": "Airbus", "sector": "Industrials", "industry": "Aerospace & Defense"},
            {"ticker": "ALV.DE", "name": "Allianz", "sector": "Financial Services", "industry": "Insurance"},
            {"ticker": "ASML.AS", "name": "ASML Holding", "sector": "Technology", "industry": "Semiconductors"},
            {"ticker": "CS.PA", "name": "AXA", "sector": "Financial Services", "industry": "Insurance"},
            {"ticker": "SAN.MC", "name": "Banco Santander", "sector": "Financial Services", "industry": "Banks"},
            {"ticker": "BAS.DE", "name": "BASF", "sector": "Basic Materials", "industry": "Chemicals"},
            {"ticker": "BAYN.DE", "name": "Bayer", "sector": "Healthcare", "industry": "Drug Manufacturers"},
            {"ticker": "BMW.DE", "name": "BMW", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            {"ticker": "BNP.PA", "name": "BNP Paribas", "sector": "Financial Services", "industry": "Banks"},
            {"ticker": "CRH.PA", "name": "CRH plc", "sector": "Basic Materials", "industry": "Building Materials"},
            {"ticker": "DTG.DE", "name": "Daimler Truck", "sector": "Industrials", "industry": "Farm & Heavy Construction Machinery"},
            {"ticker": "DPW.DE", "name": "Deutsche Post", "sector": "Industrials", "industry": "Integrated Freight & Logistics"},
            {"ticker": "DTE.DE", "name": "Deutsche Telekom", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "ENEL.MI", "name": "Enel", "sector": "Utilities", "industry": "Utilities—Regulated Electric"},
            {"ticker": "ENI.MI", "name": "Eni", "sector": "Energy", "industry": "Oil & Gas Integrated"},
            {"ticker": "EL.PA", "name": "EssilorLuxottica", "sector": "Healthcare", "industry": "Medical Instruments & Supplies"},
            {"ticker": "IBE.MC", "name": "Iberdrola", "sector": "Utilities", "industry": "Utilities—Regulated Electric"},
            {"ticker": "ITX.MC", "name": "Inditex", "sector": "Consumer Cyclical", "industry": "Apparel Retail"},
            {"ticker": "IFX.DE", "name": "Infineon", "sector": "Technology", "industry": "Semiconductors"},
            {"ticker": "INGA.AS", "name": "ING Group", "sector": "Financial Services", "industry": "Banks"},
            {"ticker": "ISP.MI", "name": "Intesa Sanpaolo", "sector": "Financial Services", "industry": "Banks"},
            {"ticker": "KER.PA", "name": "Kering", "sector": "Consumer Cyclical", "industry": "Luxury Goods"},
            {"ticker": "OR.PA", "name": "L'Oréal", "sector": "Consumer Defensive", "industry": "Household & Personal Products"},
            {"ticker": "LIN.DE", "name": "Linde", "sector": "Basic Materials", "industry": "Specialty Chemicals"},
            {"ticker": "MC.PA", "name": "LVMH", "sector": "Consumer Cyclical", "industry": "Luxury Goods"},
            {"ticker": "MBG.DE", "name": "Mercedes-Benz Group", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            {"ticker": "MUV2.DE", "name": "Munich Re", "sector": "Financial Services", "industry": "Insurance"},
            {"ticker": "NOKIA.HE", "name": "Nokia", "sector": "Technology", "industry": "Communication Equipment"},
            {"ticker": "ORA.PA", "name": "Orange", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "PHIA.AS", "name": "Philips", "sector": "Healthcare", "industry": "Medical Devices"},
            {"ticker": "SAF.PA", "name": "Safran", "sector": "Industrials", "industry": "Aerospace & Defense"},
            {"ticker": "SAN.PA", "name": "Sanofi", "sector": "Healthcare", "industry": "Drug Manufacturers"},
            {"ticker": "SAP.DE", "name": "SAP", "sector": "Technology", "industry": "Software—Application"},
            {"ticker": "SHL.DE", "name": "Siemens Healthineers", "sector": "Healthcare", "industry": "Medical Devices"},
            {"ticker": "SIE.DE", "name": "Siemens", "sector": "Industrials", "industry": "Specialty Industrial Machinery"},
            {"ticker": "SU.PA", "name": "Schneider Electric", "sector": "Industrials", "industry": "Electrical Equipment & Parts"},
            {"ticker": "TEF.MC", "name": "Telefónica", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "FP.PA", "name": "TotalEnergies", "sector": "Energy", "industry": "Oil & Gas Integrated"},
            {"ticker": "URW.AS", "name": "Unibail-Rodamco-Westfield", "sector": "Real Estate", "industry": "REIT—Retail"},
            {"ticker": "UNA.AS", "name": "Unilever", "sector": "Consumer Defensive", "industry": "Household & Personal Products"},
            {"ticker": "DG.PA", "name": "Vinci", "sector": "Industrials", "industry": "Engineering & Construction"},
            {"ticker": "VOW3.DE", "name": "Volkswagen Group", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            {"ticker": "VNA.DE", "name": "Vonovia", "sector": "Real Estate", "industry": "Real Estate—Development"}
        ]
        
        constituents = pd.DataFrame(eurostoxx_data)
        return constituents
    
    def _add_eurostoxx_market_suffixes(self, df):
        """
        Ajoute les suffixes de marché aux tickers EURO STOXX 50 si nécessaire.
        
        Args:
            df (pd.DataFrame): DataFrame contenant les tickers sans suffixes
            
        Returns:
            pd.DataFrame: DataFrame avec les tickers complets
        """
        # Mapping des entreprises vers leurs suffixes de marché
        market_mapping = {
            'Adidas': 'ADS.DE',
            'Airbus': 'AIR.PA',
            'Allianz': 'ALV.DE',
            'ASML Holding': 'ASML.AS',
            'AXA': 'CS.PA',
            'Banco Santander': 'SAN.MC',
            'BASF': 'BAS.DE',
            'Bayer': 'BAYN.DE',
            'BMW': 'BMW.DE',
            'BNP Paribas': 'BNP.PA',
            'CRH': 'CRH.PA',
            'Daimler Truck': 'DTG.DE',
            'Deutsche Post': 'DPW.DE',
            'Deutsche Telekom': 'DTE.DE',
            'Enel': 'ENEL.MI',
            'Eni': 'ENI.MI',
            'EssilorLuxottica': 'EL.PA',
            'Iberdrola': 'IBE.MC',
            'Inditex': 'ITX.MC',
            'Infineon': 'IFX.DE',
            'ING Group': 'INGA.AS',
            'Intesa Sanpaolo': 'ISP.MI',
            'Kering': 'KER.PA',
            'L\'Oréal': 'OR.PA',
            'Linde': 'LIN.DE',
            'LVMH': 'MC.PA',
            'Mercedes-Benz': 'MBG.DE',
            'Munich Re': 'MUV2.DE',
            'Nokia': 'NOKIA.HE',
            'Orange': 'ORA.PA',
            'Philips': 'PHIA.AS',
            'Safran': 'SAF.PA',
            'Sanofi': 'SAN.PA',
            'SAP': 'SAP.DE',
            'Siemens Healthineers': 'SHL.DE',
            'Siemens': 'SIE.DE',
            'Schneider Electric': 'SU.PA',
            'Telefónica': 'TEF.MC',
            'TotalEnergies': 'FP.PA',
            'Unibail-Rodamco-Westfield': 'URW.AS',
            'Unilever': 'UNA.AS',
            'Vinci': 'DG.PA',
            'Volkswagen': 'VOW3.DE',
            'Vonovia': 'VNA.DE'
        }
        
        # Fonction pour trouver le bon suffixe
        def find_market_suffix(row):
            name = row['name']
            ticker = row['ticker']
            
            # Vérifier si le ticker contient déjà un suffixe
            if '.' in ticker:
                return ticker
            
            # Chercher une correspondance exacte
            if name in market_mapping:
                return market_mapping[name]
            
            # Chercher une correspondance partielle
            for company, suffix in market_mapping.items():
                if company in name or name in company:
                    return suffix
            
            # Par défaut, retourner le ticker original
            return ticker
        
        # Appliquer la fonction à chaque ligne
        df['ticker'] = df.apply(find_market_suffix, axis=1)
        
        return df
    
    def create_msci_world_list(self):
        """
        Crée une liste approximative des composantes du MSCI World.
        Comme Alpha Vantage ne fournit pas cette liste, nous utilisons une approche alternative.
        
        Returns:
            pd.DataFrame: DataFrame contenant les composantes approximatives du MSCI World
        """
        logger.info("Création d'une liste approximative des composantes du MSCI World...")
        
        # Approche simplifiée : combiner les plus grandes entreprises des marchés développés
        
        # 1. Récupérer le S&P 500 (représente environ 80% du marché US)
        sp500 = self.scrape_sp500()
        
        # 2. Ajouter des titres européens majeurs (EuroStoxx 50)
        eurostoxx = self.scrape_eurostoxx()
        
        # 3. Ajouter des titres japonais et asiatiques majeurs
        # Liste basique des principaux titres japonais et asiatiques
        asia_pacific_data = [
            {"ticker": "7203.T", "name": "Toyota Motor", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            {"ticker": "9984.T", "name": "SoftBank Group", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "6758.T", "name": "Sony Group", "sector": "Technology", "industry": "Consumer Electronics"},
            {"ticker": "9432.T", "name": "Nippon Telegraph & Telephone", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "6501.T", "name": "Hitachi", "sector": "Industrials", "industry": "Specialty Industrial Machinery"},
            {"ticker": "4502.T", "name": "Takeda Pharmaceutical", "sector": "Healthcare", "industry": "Drug Manufacturers"},
            {"ticker": "8306.T", "name": "Mitsubishi UFJ Financial", "sector": "Financial Services", "industry": "Banks"},
            {"ticker": "9433.T", "name": "KDDI", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "6861.T", "name": "Keyence", "sector": "Technology", "industry": "Electronic Equipment"},
            {"ticker": "7974.T", "name": "Nintendo", "sector": "Communication Services", "industry": "Electronic Gaming & Multimedia"},
            {"ticker": "005930.KS", "name": "Samsung Electronics", "sector": "Technology", "industry": "Consumer Electronics"},
            {"ticker": "000660.KS", "name": "SK Hynix", "sector": "Technology", "industry": "Semiconductors"},
            {"ticker": "035420.KS", "name": "NAVER", "sector": "Communication Services", "industry": "Internet Content & Information"},
            {"ticker": "051910.KS", "name": "LG Chem", "sector": "Basic Materials", "industry": "Specialty Chemicals"},
            {"ticker": "005380.KS", "name": "Hyundai Motor", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
            {"ticker": "0700.HK", "name": "Tencent Holdings", "sector": "Communication Services", "industry": "Internet Content & Information"},
            {"ticker": "0941.HK", "name": "China Mobile", "sector": "Communication Services", "industry": "Telecom Services"},
            {"ticker": "9988.HK", "name": "Alibaba Group", "sector": "Consumer Cyclical", "industry": "Internet Retail"},
            {"ticker": "1299.HK", "name": "AIA Group", "sector": "Financial Services", "industry": "Insurance"},
            {"ticker": "3988.HK", "name": "Bank of China", "sector": "Financial Services", "industry": "Banks"}
        ]
        
        asia_pacific = pd.DataFrame(asia_pacific_data)
        
        # Combiner toutes les sources
        msci_world = pd.concat([sp500, eurostoxx, asia_pacific], ignore_index=True)
        
        # Supprimer les doublons potentiels
        msci_world = msci_world.drop_duplicates(subset=['ticker'])
        
        logger.info(f"Créé une liste de {len(msci_world)} titres pour le MSCI World")
        
        return msci_world
    
    def get_index_constituents(self, index_name, force_refresh=False):
        """
        Récupère les constituants d'un indice, soit depuis le cache local soit en les scrapant.
        
        Args:
            index_name (str): Nom de l'indice (clé dans INDICES)
            force_refresh (bool, optional): Si True, force le rafraîchissement des données. Par défaut False.
            
        Returns:
            pd.DataFrame: DataFrame contenant les constituants de l'indice
        """
        if index_name not in INDICES:
            logger.error(f"Indice {index_name} non supporté")
            return pd.DataFrame()
        
        # Chemin du fichier cache
        cache_file = self.index_files.get(index_name)
        
        # Vérifier si le cache existe et si on ne force pas le rafraîchissement
        if not force_refresh and os.path.exists(cache_file):
            logger.info(f"Chargement des constituants de {index_name} depuis le cache")
            constituents = load_from_csv(cache_file)
            
            if not constituents.empty:
                return constituents
            
            logger.warning(f"Cache vide ou invalide pour {index_name}, récupération des données fraîches")
        
        # Récupérer les constituants selon l'indice
        if index_name == 'SP500':
            constituents = self.scrape_sp500()
        elif index_name == 'NASDAQ':
            constituents = self.scrape_nasdaq()
        elif index_name == 'EUROSTOXX':
            constituents = self.scrape_eurostoxx()
        elif index_name == 'MSCIWORLD':
            constituents = self.create_msci_world_list()
        else:
            logger.error(f"Méthode de récupération non implémentée pour {index_name}")
            return pd.DataFrame()
        
        # Sauvegarder les constituants dans le cache si non vide
        if not constituents.empty:
            save_to_csv(constituents, cache_file)
            logger.info(f"Constituants de {index_name} sauvegardés dans {cache_file}")
        
        return constituents
    
    def get_all_indices_constituents(self, force_refresh=False):
        """
        Récupère les constituants de tous les indices supportés.
        
        Args:
            force_refresh (bool, optional): Si True, force le rafraîchissement des données. Par défaut False.
            
        Returns:
            dict: Dictionnaire avec les noms d'indices comme clés et les DataFrames comme valeurs
        """
        constituents = {}
        
        for index_name in INDICES.keys():
            logger.info(f"Récupération des constituants de {index_name}")
            constituents[index_name] = self.get_index_constituents(index_name, force_refresh)
        
        return constituents
