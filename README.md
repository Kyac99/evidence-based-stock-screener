# Evidence-Based Stock Screener

Ce projet est un screener d'actions basé sur les facteurs Quality et Momentum pour sélectionner les meilleurs actifs à travers différents indices mondiaux (S&P 500, NASDAQ, MSCI World, Euro Stoxx 50).

## Description

Le screener utilise l'API Alpha Vantage pour récupérer les données financières et de prix des actions. Le projet est structuré pour collecter, traiter et analyser les données de manière efficace, en permettant d'appliquer des filtres basés sur des facteurs de qualité (fondamentaux) et de momentum (technique).

## Fonctionnalités

- Collecte de données pour les composantes des principaux indices boursiers
- Récupération des données de prix et des données fondamentales via l'API Alpha Vantage
- Traitement des données pour calculer différentes métriques et indicateurs
- Combinaison des données de prix et fondamentales
- Filtrage des actions selon des critères de qualité et de momentum
- Sauvegarde et chargement des données traitées

## Structure du projet

```
evidence-based-stock-screener/
├── data/                    # Répertoire pour stocker les données
│   ├── raw/                 # Données brutes récupérées des APIs
│   └── processed/           # Données traitées et prêtes à l'emploi
├── logs/                    # Logs d'exécution
├── src/                     # Code source
│   ├── data/                # Module de gestion des données
│   │   ├── collectors/      # Collecteurs de données
│   │   │   ├── index_collector.py       # Récupération des composantes des indices
│   │   │   ├── price_collector.py       # Récupération des données de prix
│   │   │   └── fundamental_collector.py # Récupération des données fondamentales
│   │   ├── processors/      # Processeurs de données
│   │   │   └── data_processor.py        # Traitement et formatage des données
│   │   ├── constants.py     # Constantes pour le module de données
│   │   └── utils.py         # Fonctions utilitaires
│   ├── factors/             # Module pour les facteurs de sélection
│   ├── screener/            # Module de filtrage des actions
│   └── visualization/       # Module de visualisation des résultats
└── main.py                  # Script principal
```

## Prérequis

- Python 3.8 ou supérieur
- Clé API Alpha Vantage (obtenir sur [alphavantage.co](https://www.alphavantage.co/))

## Installation

1. Clonez ce dépôt :
   ```
   git clone https://github.com/Kyac99/evidence-based-stock-screener.git
   cd evidence-based-stock-screener
   ```

2. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

3. Configuration de l'API Alpha Vantage :
   ```
   export ALPHA_VANTAGE_API_KEY=votre_clé_api
   ```
   Ou sous Windows :
   ```
   set ALPHA_VANTAGE_API_KEY=votre_clé_api
   ```

## Utilisation

### Collection de données

```
python main.py --index SP500 --limit 10 --api_key YOUR_API_KEY
```

Arguments :
- `--index` : Indice à traiter (SP500, NASDAQ, MSCIWORLD, EUROSTOXX)
- `--limit` : Nombre maximum de titres à traiter (optionnel)
- `--api_key` : Clé API Alpha Vantage (si non définie dans les variables d'environnement)

### Limites à considérer

- L'API gratuite d'Alpha Vantage limite les requêtes à 5 par minute et 500 par jour.
- Le traitement des données de l'ensemble des titres d'un indice peut prendre du temps.
- Pour les tests initiaux, il est recommandé d'utiliser l'argument `--limit` pour limiter le nombre de titres.

## Développement futur

- Implémentation des facteurs de filtrage Quality et Momentum
- Création d'un système de scoring pour les actions
- Mise en place de tests unitaires
- Ajout de visualisations pour l'analyse des résultats
- Optimisation de la collecte de données avec des mécanismes de cache

## Licence

Ce projet est sous licence MIT.
