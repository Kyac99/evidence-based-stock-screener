"""
Module de collecte et de traitement de données pour le stock screener.
"""

# Import des principaux collecteurs et processeurs qui seront créés
from src.data.collectors.index_collector import IndexCollector
from src.data.collectors.price_collector import PriceCollector
from src.data.collectors.fundamental_collector import FundamentalCollector
from src.data.processors.data_processor import DataProcessor
