# Evidence-Based Stock Screener

Un outil de sélection d'actions basé sur l'Evidence-Based Investing, se concentrant sur les facteurs **Quality** et **Momentum** pour identifier les opportunités d'investissement à travers différents indices mondiaux.

## Présentation

Ce screener s'appuie sur des recherches scientifiques ayant prouvé la pertinence de certaines stratégies d'investissement sur le long terme. L'approche Evidence-Based Investing permet de rationaliser le processus d'investissement en s'appuyant sur des facteurs qui ont démontré leur efficacité à travers le temps.

### Facteurs de sélection

#### Facteur Momentum

Le momentum combine deux dimensions :

1. **Momentum Technique** : 
   - Tendance haussière du titre à court terme (1-3 mois)
   - Tendance haussière à moyen terme (6 mois)
   - Tendance haussière à long terme (12 mois)

2. **Momentum Fondamental** :
   - Révisions des bénéfices par les analystes
   - Révisions du chiffre d'affaires
   - Pondération selon la visibilité du business model
   - Prise en compte de la dispersion des estimations des analystes

#### Facteur Quality

Le facteur qualité privilégie les sociétés ayant :

- Profitabilité positive 
- Rentabilité élevée (ROE, ROCE, etc.)
- Bilan comptable sain (faible endettement)
- Faible volatilité des marges
- Historique de qualité en matière de publication de résultats
- Bonne visibilité sur les résultats futurs

## Indices analysés

Le screener permet de filtrer les actions des indices suivants :

- Nasdaq
- S&P 500
- Eurostoxx
- Indices small et mid cap européennes
- MSCI World Tech
- Et d'autres indices configurables

## Fonctionnalités

- Calcul de scores Momentum et Quality pour chaque action
- Classement des actions selon un score combiné personnalisable
- Filtres avancés pour affiner la sélection
- Visualisation des tendances et performances des actions sélectionnées
- Export des résultats en CSV et Excel
- Mise à jour automatique des données

### Visualisations avancées

Le module de visualisation intégré permet de générer des graphiques interactifs pour faciliter l'analyse :

- **Factor Scores** : Visualisation des scores Momentum, Quality et Combined pour les meilleures actions
- **Sector Distribution** : Analyse de la répartition sectorielle des actions sélectionnées
- **Factor Heatmap** : Carte de chaleur positionnant les actions selon leurs scores Quality et Momentum
- **Performance Comparison** : Comparaison des performances historiques des actions sélectionnées vs benchmark
- **Rolling Correlations** : Analyse des corrélations roulantes avec un indice de référence
- **Metric Comparison** : Comparaison des métriques fondamentales entre différentes actions
- **Score Distributions** : Visualisation de la distribution statistique des scores

Ces visualisations sont disponibles en versions interactives (Plotly) et statiques (Matplotlib).

## Architecture du projet

```
evidence-based-stock-screener/
├── data/                  # Stockage des données
│   ├── raw/               # Données brutes téléchargées
│   └── processed/         # Données traitées et prêtes à l'analyse
├── src/                   # Code source
│   ├── data/              # Modules de collecte et traitement des données
│   │   ├── collectors/    # Collecteurs de données pour différentes sources
│   │   └── processors/    # Traitement des données brutes
│   ├── factors/           # Calcul des facteurs d'investissement
│   │   ├── momentum.py    # Calcul des métriques de momentum
│   │   └── quality.py     # Calcul des métriques de qualité
│   ├── screener/          # Logique du screener
│   │   ├── filters.py     # Filtres avancés
│   │   └── scoring.py     # Système de scoring des actions
│   └── visualization/     # Outils de visualisation des résultats
│       └── chart_generator.py  # Générateur de graphiques interactifs et statiques
├── notebooks/             # Notebooks Jupyter pour l'analyse exploratoire
│   └── screener_visualization_demo.ipynb  # Démonstration des visualisations
├── config/                # Fichiers de configuration
├── tests/                 # Tests unitaires et d'intégration
└── reports/               # Rapports générés et résultats d'analyse
    └── visualizations/    # Graphiques exportés
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Kyac99/evidence-based-stock-screener.git
cd evidence-based-stock-screener

# Installer les dépendances
pip install -r requirements.txt

# Configuration des API (voir instructions dans config/README.md)
```

## Utilisation

### Utilisation basique

```python
from src.screener import StockScreener

# Initialiser le screener
screener = StockScreener()

# Charger les données pour un indice spécifique
screener.load_index_data("SP500")

# Calculer les scores pour chaque action
screener.calculate_scores()

# Obtenir les N meilleures actions selon les critères
top_stocks = screener.get_top_stocks(n=20)

# Afficher les résultats
print(top_stocks)

# Exporter les résultats
screener.export_results("top_stocks.xlsx")
```

### Utilisation avec visualisations

```python
from src.screener import StockScreener
from src.visualization import ChartGenerator

# Initialiser le screener et calculer les scores
screener = StockScreener()
screener.load_index_data("SP500")
screener.calculate_scores()
top_stocks = screener.get_top_stocks(n=20)

# Initialiser le générateur de graphiques
chart_gen = ChartGenerator(output_dir='reports/visualizations')

# Créer diverses visualisations
# Visualisation des scores
scores_fig = chart_gen.plot_factor_scores(
    scores_df=screener.combined_scores,
    top_n=15,
    interactive=True  # Utiliser Plotly pour des graphiques interactifs
)

# Visualisation de la distribution sectorielle
sector_fig = chart_gen.plot_sector_distribution(
    scores_df=top_stocks,
    top_n=30
)

# Comparaison des performances
perf_fig = chart_gen.plot_performance_comparison(
    price_data=screener.price_data,
    tickers=top_stocks.index.tolist()[:5],
    benchmark_ticker='SPY'
)

# Afficher les graphiques (dans un notebook Jupyter)
scores_fig.show()
sector_fig.show()
perf_fig.show()
```

Pour un exemple complet d'utilisation des visualisations, consultez le notebook `notebooks/screener_visualization_demo.ipynb`.

## Sources de données

Le screener utilise plusieurs sources de données pour ses analyses :

- Données de prix historiques (Yahoo Finance, Alpha Vantage)
- Données fondamentales (Financial Modeling Prep, SimFin)
- Estimations des analystes (Refinitiv, FactSet)
- Ratios financiers (QuandL, Morningstar)

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Avertissement

Ce screener est un outil d'aide à la décision et ne constitue pas une recommandation d'investissement. Les performances passées ne préjugent pas des performances futures.
