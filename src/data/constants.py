"""
Constantes utilisées dans le module de données.
"""

import os

# Chemins des répertoires
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Sous-répertoires pour les données brutes
INDICES_DIR = os.path.join(RAW_DATA_DIR, "indices")
PRICES_DIR = os.path.join(RAW_DATA_DIR, "prices")
FUNDAMENTALS_DIR = os.path.join(RAW_DATA_DIR, "fundamentals")

# Définition des indices supportés
INDICES = {
    "SP500": "S&P 500",
    "NASDAQ": "NASDAQ Composite", 
    "MSCIWORLD": "MSCI World",
    "EUROSTOXX": "EURO STOXX 50"
}

# URLs pour le scraping des indices
INDEX_URLS = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ100": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "EUROSTOXX": "https://en.wikipedia.org/wiki/EURO_STOXX_50",
    # Pour le MSCI World, nous utiliserons une approche différente car pas de source simple
}

# Correspondance entre les indices et les tables/colonnes pour le scraping
INDEX_SCRAPE_CONFIG = {
    "SP500": {"table_index": 0, "symbol_column": "Symbol"},
    "NASDAQ100": {"table_index": 4, "symbol_column": "Ticker"},
    "EUROSTOXX": {"table_index": 2, "symbol_column": "Ticker"}
}

# Nombre maximum de requêtes par minute pour Alpha Vantage (API gratuite)
MAX_REQUESTS_PER_MINUTE = 5

# Délai entre les requêtes pour ne pas dépasser la limite (en secondes)
REQUEST_DELAY = 60 / MAX_REQUESTS_PER_MINUTE
