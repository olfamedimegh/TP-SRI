# TP2 - Systèmes de Recherche d'Information

## 📋 Description

Ce TP implémente et compare différents modèles de recherche d'information vus en cours :

1. **Modèle Booléen** - Recherche basée sur la logique booléenne (AND, OR, NOT)
2. **Modèle Vectoriel (TF-IDF)** - Représentation vectorielle avec similarité cosinus
3. **Modèle Probabiliste (BM25)** - Fonction de scoring probabiliste
4. **Modèle de Langue** - Estimation de la probabilité de génération de la requête

L'évaluation comparative utilise l'approche **LLM as a Judge** pour évaluer automatiquement la qualité des résultats.

## 🏗️ Structure du Projet

```
TP2/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
├── main.py                      # Script principal
├── corpus.py                    # Corpus de documents de test
├── preprocessing.py             # Prétraitement et indexation
├── llm_judge.py                 # Système d'évaluation LLM as a Judge
└── models/
    ├── __init__.py
    ├── boolean_model.py         # Modèle Booléen
    ├── vector_model.py          # Modèle Vectoriel (TF-IDF)
    ├── probabilistic_model.py   # Modèle BM25 et BM25+
    └── language_model.py        # Modèles de Langue (JM & Dirichlet)
```

## 🚀 Installation

```bash
# Cloner ou se placer dans le répertoire du TP
cd TP2

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Utilisation

### Mode Démonstration

Exécute une démonstration des différents modèles :

```bash
python main.py --mode demo
```

### Mode Interactif

Permet de tester les modèles avec vos propres requêtes :

```bash
python main.py --mode interactive
```

### Mode Évaluation (LLM as a Judge)

Lance l'évaluation comparative des modèles :

```bash
# Avec le juge simulé (par défaut)
python main.py --mode evaluate

# Avec OpenAI GPT
export OPENAI_API_KEY="votre-clé-api"
python main.py --mode evaluate --provider openai

# Avec Ollama (local)
python main.py --mode evaluate --provider ollama
```

### Mode Comparaison

Compare deux modèles en détail :

```bash
python main.py --mode compare --query "intelligence artificielle"
```

## 📚 Les Modèles Implémentés

### 1. Modèle Booléen

Le modèle booléen traite les requêtes comme des expressions logiques. Un document est soit pertinent (1) soit non pertinent (0).

- **Opérateurs** : AND, OR, NOT
- **Avantages** : Simple, précis pour les requêtes exactes
- **Inconvénients** : Pas de classement, tout ou rien

### 2. Modèle Vectoriel (TF-IDF)

Représente les documents et requêtes comme des vecteurs dans un espace multidimensionnel.

- **Pondération** : TF-IDF (Term Frequency - Inverse Document Frequency)
- **Similarité** : Cosinus entre les vecteurs
- **Avantages** : Classement graduel, performant
- **Formules** :
  - TF(t,d) = 1 + log(freq(t,d))
  - IDF(t) = log(N/df(t))
  - Similarité = cos(q, d) = (q·d) / (||q|| × ||d||)

### 3. Modèle Probabiliste (BM25)

Fonction de scoring probabiliste améliorant le TF-IDF.

- **Paramètres** : k1 (saturation TF), b (normalisation longueur)
- **Avantages** : Très performant, standard industriel
- **Formule** :
  ```
  BM25(D,Q) = Σ IDF(qi) × (tf × (k1+1)) / (tf + k1 × (1-b + b×|D|/avgdl))
  ```

### 4. Modèle de Langue

Estime la probabilité qu'un document génère la requête.

#### Variante Jelinek-Mercer
- Interpolation entre modèle document et corpus
- P(t|D) = λ × P(t|D) + (1-λ) × P(t|C)

#### Variante Dirichlet
- Lissage adaptatif selon la longueur du document
- P(t|D) = (tf + μ×P(t|C)) / (|D| + μ)

## 🤖 LLM as a Judge

L'approche "LLM as a Judge" utilise un grand modèle de langage pour évaluer la pertinence des résultats.

### Fonctionnalités

1. **Évaluation de pertinence** : Score de 0 à 5 pour chaque document retourné
2. **Comparaison pairée** : Compare les classements de deux modèles
3. **Rapport final** : Classement global des modèles avec scores combinés

### Providers Supportés

- **Mock** : Simulation pour les tests (par défaut)
- **OpenAI** : GPT-3.5/GPT-4
- **Ollama** : Modèles locaux (Llama2, Mistral, etc.)

## 📊 Exemple de Résultats

```
╔══════════════════════════════════════════════════════════════════════╗
║            TP2 - SYSTÈMES DE RECHERCHE D'INFORMATION                 ║
╚══════════════════════════════════════════════════════════════════════╝

🔬 ÉVALUATION COMPARATIVE

📊 SCORES MOYENS DE PERTINENCE (sur 5):
   BM25: 4.20
   Vectoriel (TF-IDF): 4.05
   LM (Dirichlet): 3.95
   LM (Jelinek-Mercer): 3.85
   Booléen: 3.50

🥇 CLASSEMENT FINAL:
   🥇 1. BM25 - Score: 4.20
   🥈 2. Vectoriel (TF-IDF) - Score: 4.05
   🥉 3. LM (Dirichlet) - Score: 3.95
```

## 📝 Corpus de Test

Le corpus inclut 12 documents en français sur les thèmes :
- Intelligence Artificielle
- Machine Learning
- Bases de Données et Big Data
- Recherche d'Information
- Traitement du Langage Naturel
- Sécurité Informatique
- Cloud Computing
- Réseaux de Neurones
- Internet des Objets
- Éthique de l'IA

## 🔧 Extension

Pour ajouter vos propres documents :

```python
from corpus import DOCUMENTS

# Ajouter un nouveau document
DOCUMENTS["doc13"] = {
    "title": "Mon nouveau document",
    "content": "Le contenu de mon document..."
}
```

Pour ajouter un nouveau modèle :

```python
class MonModele:
    def __init__(self, index):
        self.index = index
        self.name = "Mon Modèle"
    
    def get_ranking(self, query, top_k=10):
        # Implémenter votre algorithme
        return [(doc_id, score), ...]
```

## 📚 Références

- Robertson, S. E., & Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. SIGIR.
- Ponte, J. M., & Croft, W. B. (1998). A language modeling approach to information retrieval. SIGIR.
- Zhai, C., & Lafferty, J. (2004). A study of smoothing methods for language models applied to information retrieval. TOIS.

## 👨‍🎓 Auteur

INSAT - GL5 - Systèmes de Recherche d'Information - 2025

