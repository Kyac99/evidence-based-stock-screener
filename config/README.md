# Configuration du Stock Screener

Ce document explique comment configurer correctement les différentes API et paramètres nécessaires au bon fonctionnement du screener.

## Configuration des API

Le screener utilise plusieurs API externes pour récupérer les données. Vous devez obtenir des clés API pour chacune de ces sources.

### Fichier de configuration

Créez un fichier `config.ini` dans ce répertoire avec la structure suivante :

```ini
[API_KEYS]
YAHOO_FINANCE = votre_clé_ici
ALPHA_VANTAGE = votre_clé_ici
FINANCIAL_MODELING_PREP = votre_clé_ici
SIMFIN = votre_clé_ici
QUANDL = votre_clé_ici

[PROXY]
USE_PROXY = False
HTTP_PROXY = http://proxy.example.com:8080
HTTPS_PROXY = https://proxy.example.com:8080

[DATA]
CACHE_DIR = data/cache
EXPIRY_DAYS = 7
```

### Obtention des clés API

#### Yahoo Finance
- Visitez [Yahoo Finance API](https://www.yahoofinanceapi.com/)
- Inscrivez-vous pour obtenir une clé API gratuite ou premium

#### Alpha Vantage
- Visitez [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- Inscrivez-vous pour obtenir une clé API gratuite

#### Financial Modeling Prep
- Visitez [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
- Inscrivez-vous pour obtenir une clé API

#### SimFin
- Visitez [SimFin](https://simfin.com/api/v2/documentation/)
- Créez un compte et obtenez votre clé API

#### Quandl
- Visitez [Quandl](https://www.quandl.com/tools/api)
- Inscrivez-vous pour obtenir une clé API

## Paramètres du screener

### Fichier de paramètres du screener

Créez un fichier `screener_settings.json` dans ce répertoire avec la structure suivante :

```json
{
  "indices": {
    "SP500": {
      "source": "yahoo",
      "symbol": "^GSPC",
      "components_source": "wikipedia"
    },
    "NASDAQ": {
      "source": "yahoo",
      "symbol": "^IXIC",
      "components_source": "nasdaq_api"
    },
    "EUROSTOXX": {
      "source": "yahoo",
      "symbol": "^STOXX",
      "components_source": "csv",
      "components_file": "data/raw/eurostoxx_components.csv"
    }
  },
  "factors": {
    "momentum": {
      "short_term_window": 63,
      "medium_term_window": 126,
      "long_term_window": 252,
      "earnings_revision_weight": 0.3,
      "price_momentum_weight": 0.7
    },
    "quality": {
      "min_roe": 10,
      "min_roce": 8,
      "max_debt_to_equity": 1.5,
      "min_profit_margin": 5
    }
  },
  "scoring": {
    "momentum_weight": 0.5,
    "quality_weight": 0.5,
    "normalization_method": "minmax",
    "ranking_method": "percentile"
  },
  "filters": {
    "min_market_cap": 100000000,
    "min_daily_volume": 500000,
    "exclude_sectors": ["Financials"],
    "max_pe_ratio": 50
  }
}
```

## Personnalisation avancée

### Ajout de nouveaux indices

Pour ajouter un nouvel indice à analyser, ajoutez une nouvelle entrée dans la section "indices" du fichier `screener_settings.json`.

Exemple pour ajouter l'indice CAC 40 :

```json
"CAC40": {
  "source": "yahoo",
  "symbol": "^FCHI",
  "components_source": "csv",
  "components_file": "data/raw/cac40_components.csv"
}
```

### Format des fichiers CSV pour les composants d'indice

Si vous utilisez un fichier CSV pour définir les composants d'un indice (`components_source": "csv"`), le fichier doit avoir la structure suivante :

```
ticker,name,sector,industry
AAPL,"Apple Inc.","Technology","Consumer Electronics"
MSFT,"Microsoft Corporation","Technology","Software—Infrastructure"
...
```

## Exécution avec des paramètres personnalisés

Vous pouvez également spécifier un fichier de paramètres personnalisé lors de l'exécution du screener :

```python
from src.screener import StockScreener

# Initialiser le screener avec des paramètres personnalisés
screener = StockScreener(config_path="config/my_custom_settings.json")
```
