# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Corpus de documents pour les tests
"""

# Corpus de documents en français sur le thème de l'intelligence artificielle
# et des technologies de l'information

DOCUMENTS = {
    "doc1": {
        "title": "Introduction à l'Intelligence Artificielle",
        "content": """L'intelligence artificielle est un domaine de l'informatique qui vise à créer 
        des systèmes capables de simuler l'intelligence humaine. Les applications de l'IA incluent 
        la reconnaissance vocale, la vision par ordinateur, le traitement du langage naturel et 
        les systèmes de recommandation. Le machine learning est une branche importante de l'IA 
        qui permet aux machines d'apprendre à partir de données sans être explicitement programmées.
        Les réseaux de neurones artificiels sont inspirés du fonctionnement du cerveau humain."""
    },
    "doc2": {
        "title": "Le Machine Learning et ses Applications",
        "content": """Le machine learning ou apprentissage automatique est une technique d'intelligence 
        artificielle permettant aux ordinateurs d'apprendre à partir de données. Il existe trois types 
        principaux d'apprentissage : supervisé, non supervisé et par renforcement. L'apprentissage 
        supervisé utilise des données étiquetées pour entraîner des modèles de classification et de 
        régression. Les algorithmes populaires incluent les forêts aléatoires, les SVM et les réseaux 
        de neurones profonds. Le deep learning a révolutionné la reconnaissance d'images et le 
        traitement du langage naturel."""
    },
    "doc3": {
        "title": "Les Bases de Données et le Big Data",
        "content": """Les bases de données sont des systèmes organisés pour stocker et gérer de grandes 
        quantités d'informations. Les bases de données relationnelles utilisent le langage SQL pour 
        les requêtes. Avec l'avènement du Big Data, de nouvelles technologies comme Hadoop et Spark 
        ont émergé pour traiter des volumes massifs de données. NoSQL offre des alternatives flexibles 
        pour le stockage de données non structurées. L'analyse de données permet d'extraire des 
        informations précieuses pour la prise de décision en entreprise."""
    },
    "doc4": {
        "title": "La Recherche d'Information",
        "content": """La recherche d'information est le processus de localisation de documents pertinents 
        dans une collection. Les moteurs de recherche utilisent des techniques d'indexation pour 
        accélérer les requêtes. Le modèle vectoriel représente les documents et les requêtes comme 
        des vecteurs dans un espace multidimensionnel. La similarité cosinus mesure la pertinence 
        entre une requête et un document. Le TF-IDF est une méthode de pondération qui favorise 
        les termes importants et discriminants."""
    },
    "doc5": {
        "title": "Le Traitement du Langage Naturel",
        "content": """Le traitement du langage naturel permet aux ordinateurs de comprendre et générer 
        du texte humain. Les techniques incluent la tokenisation, la lemmatisation et l'analyse 
        syntaxique. Les modèles de langue comme BERT et GPT ont révolutionné le domaine grâce aux 
        transformers. La traduction automatique et les chatbots sont des applications majeures du NLP.
        L'analyse de sentiment permet d'extraire les opinions et émotions exprimées dans les textes."""
    },
    "doc6": {
        "title": "La Sécurité Informatique",
        "content": """La sécurité informatique protège les systèmes contre les menaces et les attaques.
        Le chiffrement assure la confidentialité des données transmises sur les réseaux. Les pare-feu 
        filtrent le trafic réseau pour bloquer les accès non autorisés. L'authentification vérifie 
        l'identité des utilisateurs avant d'accorder l'accès aux ressources. Les cyberattaques 
        incluent le phishing, les ransomwares et les attaques par déni de service."""
    },
    "doc7": {
        "title": "Le Cloud Computing",
        "content": """Le cloud computing fournit des ressources informatiques à la demande via Internet.
        Les trois modèles de service sont IaaS, PaaS et SaaS. Amazon AWS, Microsoft Azure et Google 
        Cloud sont les principaux fournisseurs de services cloud. La virtualisation permet de créer 
        des machines virtuelles sur des serveurs physiques. Le cloud offre évolutivité, flexibilité 
        et réduction des coûts pour les entreprises de toutes tailles."""
    },
    "doc8": {
        "title": "Les Réseaux de Neurones Profonds",
        "content": """Les réseaux de neurones profonds sont des architectures avec plusieurs couches 
        cachées. Le deep learning excelle dans la reconnaissance d'images grâce aux réseaux convolutifs.
        Les réseaux récurrents LSTM sont utilisés pour le traitement de séquences et le langage naturel.
        La rétropropagation permet d'ajuster les poids du réseau pendant l'apprentissage. Les GPU 
        accélèrent considérablement l'entraînement des modèles de deep learning."""
    },
    "doc9": {
        "title": "L'Internet des Objets",
        "content": """L'Internet des Objets connecte des milliards d'appareils physiques au réseau.
        Les capteurs collectent des données sur l'environnement comme la température et l'humidité.
        Les protocoles MQTT et CoAP sont optimisés pour les communications IoT. Les applications 
        incluent les maisons intelligentes, les villes connectées et l'industrie 4.0. L'edge computing 
        permet de traiter les données localement pour réduire la latence."""
    },
    "doc10": {
        "title": "L'Éthique de l'Intelligence Artificielle",
        "content": """L'éthique de l'IA aborde les questions morales soulevées par les systèmes 
        intelligents. Les biais algorithmiques peuvent conduire à des décisions discriminatoires.
        La transparence des modèles est essentielle pour comprendre les décisions automatisées.
        La protection de la vie privée est menacée par la collecte massive de données personnelles.
        La régulation de l'IA vise à encadrer le développement responsable de ces technologies."""
    },
    "doc11": {
        "title": "Les Algorithmes de Recherche",
        "content": """Les algorithmes de recherche parcourent des structures de données pour trouver 
        des éléments. La recherche binaire est efficace sur les données triées avec une complexité 
        logarithmique. Les algorithmes de graphes comme BFS et DFS explorent les nœuds connectés.
        Les tables de hachage offrent un accès en temps constant pour les recherches par clé.
        L'optimisation des algorithmes est cruciale pour les performances des moteurs de recherche."""
    },
    "doc12": {
        "title": "La Visualisation de Données",
        "content": """La visualisation de données transforme les informations en représentations 
        graphiques. Les graphiques statistiques comme les histogrammes et les nuages de points 
        révèlent des patterns dans les données. Les tableaux de bord interactifs permettent 
        l'exploration dynamique des métriques. Les bibliothèques comme D3.js et Matplotlib 
        facilitent la création de visualisations. Une bonne visualisation aide à la compréhension 
        et à la communication des insights."""
    }
}

# Requêtes de test pour l'évaluation des modèles
TEST_QUERIES = [
    "intelligence artificielle et machine learning",
    "apprentissage profond réseaux neurones",
    "recherche information indexation documents",
    "bases de données big data analyse",
    "traitement langage naturel NLP",
    "sécurité informatique chiffrement",
    "cloud computing services virtuels",
    "éthique intelligence artificielle biais",
    "algorithmes recherche optimisation",
    "Internet des objets capteurs connectés"
]

def get_corpus():
    """Retourne le corpus de documents."""
    return DOCUMENTS

def get_test_queries():
    """Retourne les requêtes de test."""
    return TEST_QUERIES

if __name__ == "__main__":
    print(f"Corpus contenant {len(DOCUMENTS)} documents")
    print(f"{len(TEST_QUERIES)} requêtes de test disponibles")

