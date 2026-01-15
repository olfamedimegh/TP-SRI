#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Script principal pour l'évaluation comparative des modèles

INSAT - GL5 - Systèmes de Recherche d'Information
"""

import sys
import os
import argparse
from typing import Dict, List

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import get_corpus, get_test_queries
from preprocessing import create_index, InvertedIndex
from models.boolean_model import BooleanModel
from models.vector_model import VectorModel
from models.probabilistic_model import ProbabilisticModel, BM25Plus
from models.language_model import LanguageModel, DirichletLanguageModel
from llm_judge import (
    LLMJudge, LLMProvider, ComparativeEvaluator, quick_evaluate
)


def print_header():
    """Affiche l'en-tête du programme."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║            TP2 - SYSTÈMES DE RECHERCHE D'INFORMATION                 ║
║                                                                      ║
║     Évaluation Comparative des Modèles avec LLM as a Judge           ║
║                                                                      ║
║                     INSAT - GL5 - 2025                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def create_models(index: InvertedIndex) -> Dict:
    """
    Crée et retourne tous les modèles de recherche.
    
    Args:
        index: L'index inversé
    
    Returns:
        Dictionnaire {nom: modèle}
    """
    models = {
        "1. Booléen": BooleanModel(index),
        "2. Vectoriel (TF-IDF)": VectorModel(index),
        "3. BM25": ProbabilisticModel(index, k1=1.5, b=0.75),
        "4. BM25+": BM25Plus(index, k1=1.5, b=0.75, delta=1.0),
        "5. LM (Jelinek-Mercer)": LanguageModel(index, lambda_param=0.7),
        "6. LM (Dirichlet)": DirichletLanguageModel(index, mu=2000)
    }
    return models


def demo_single_query(models: Dict, corpus: Dict, query: str):
    """
    Démontre les résultats de tous les modèles pour une requête.
    """
    print(f"\n{'='*70}")
    print(f"DÉMONSTRATION: Résultats pour la requête")
    print(f"Requête: \"{query}\"")
    print(f"{'='*70}")
    
    for model_name, model in models.items():
        print(f"\n📌 {model_name}")
        print("-" * 50)
        
        results = model.get_ranking(query, top_k=5)
        
        if not results:
            print("  Aucun résultat trouvé.")
        else:
            for rank, (doc_id, score) in enumerate(results, 1):
                doc = corpus.get(doc_id, {})
                title = doc.get('title', 'Sans titre')
                print(f"  {rank}. [{doc_id}] {title}")
                print(f"     Score: {score:.4f}")


def interactive_search(models: Dict, corpus: Dict):
    """
    Mode de recherche interactive.
    """
    print("\n" + "="*70)
    print("MODE RECHERCHE INTERACTIVE")
    print("="*70)
    print("Entrez vos requêtes (tapez 'quit' pour quitter)")
    print("-"*70)
    
    model_names = list(models.keys())
    
    while True:
        try:
            query = input("\n🔍 Requête: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Au revoir!")
                break
            
            if not query:
                continue
            
            # Afficher les résultats de tous les modèles
            demo_single_query(models, corpus, query)
            
        except KeyboardInterrupt:
            print("\n\nInterruption. Au revoir!")
            break
        except EOFError:
            break


def run_evaluation(models: Dict, corpus: Dict, queries: List[str], 
                  provider: str = 'mock', api_key: str = None,
                  num_queries: int = None):
    """
    Lance l'évaluation comparative des modèles.
    """
    # Sélectionner le provider
    if provider == 'openai':
        llm_provider = LLMProvider.OPENAI
    elif provider == 'ollama':
        llm_provider = LLMProvider.OLLAMA
    else:
        llm_provider = LLMProvider.MOCK
    
    # Limiter le nombre de requêtes si spécifié
    if num_queries:
        queries = queries[:num_queries]
    
    print(f"\n🔬 Lancement de l'évaluation avec {provider.upper()}")
    print(f"   Nombre de requêtes: {len(queries)}")
    print(f"   Nombre de modèles: {len(models)}")
    
    # Lancer l'évaluation
    report = quick_evaluate(
        models=models,
        documents=corpus,
        queries=queries,
        provider=llm_provider,
        api_key=api_key
    )
    
    return report


def compare_two_models(model1, model2, model1_name: str, model2_name: str,
                       corpus: Dict, query: str):
    """
    Compare en détail deux modèles pour une requête.
    """
    print(f"\n{'='*70}")
    print(f"COMPARAISON DÉTAILLÉE")
    print(f"Requête: \"{query}\"")
    print(f"{'='*70}")
    
    results1 = model1.get_ranking(query, top_k=5)
    results2 = model2.get_ranking(query, top_k=5)
    
    # Afficher côte à côte
    print(f"\n{'─'*34} vs {'─'*34}")
    print(f"   {model1_name:^30}     {model2_name:^30}")
    print(f"{'─'*34}    {'─'*34}")
    
    max_len = max(len(results1), len(results2))
    
    for i in range(max_len):
        left = ""
        right = ""
        
        if i < len(results1):
            doc_id, score = results1[i]
            title = corpus.get(doc_id, {}).get('title', doc_id)[:25]
            left = f"{i+1}. {title} ({score:.3f})"
        
        if i < len(results2):
            doc_id, score = results2[i]
            title = corpus.get(doc_id, {}).get('title', doc_id)[:25]
            right = f"{i+1}. {title} ({score:.3f})"
        
        print(f"   {left:<32}     {right:<32}")
    
    # Calculer le chevauchement
    docs1 = set(d[0] for d in results1)
    docs2 = set(d[0] for d in results2)
    overlap = docs1.intersection(docs2)
    
    print(f"\n📊 Statistiques:")
    print(f"   Chevauchement: {len(overlap)}/{max(len(docs1), len(docs2))} documents")
    print(f"   Documents communs: {', '.join(overlap) if overlap else 'Aucun'}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="TP2 - Évaluation comparative des modèles de recherche d'information"
    )
    parser.add_argument(
        '--mode', 
        choices=['demo', 'interactive', 'evaluate', 'compare'],
        default='demo',
        help="Mode d'exécution (default: demo)"
    )
    parser.add_argument(
        '--query', 
        type=str,
        help="Requête pour le mode demo ou compare"
    )
    parser.add_argument(
        '--provider',
        choices=['mock', 'openai', 'ollama'],
        default='mock',
        help="Fournisseur LLM pour l'évaluation (default: mock)"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help="Clé API OpenAI (ou variable d'env OPENAI_API_KEY)"
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=5,
        help="Nombre de requêtes pour l'évaluation (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Afficher l'en-tête
    print_header()
    
    # Charger le corpus
    print("📚 Chargement du corpus...")
    corpus = get_corpus()
    queries = get_test_queries()
    print(f"   {len(corpus)} documents chargés")
    print(f"   {len(queries)} requêtes de test disponibles")
    
    # Créer l'index
    print("\n📑 Construction de l'index inversé...")
    index = create_index(corpus)
    
    # Créer les modèles
    print("\n🔧 Initialisation des modèles...")
    models = create_models(index)
    for name in models.keys():
        print(f"   ✓ {name}")
    
    # Exécution selon le mode
    if args.mode == 'demo':
        # Mode démonstration
        query = args.query or queries[0]
        demo_single_query(models, corpus, query)
        
        # Aussi montrer quelques autres requêtes
        print("\n" + "="*70)
        print("AUTRES EXEMPLES")
        print("="*70)
        for q in queries[1:4]:
            demo_single_query(models, corpus, q)
    
    elif args.mode == 'interactive':
        # Mode interactif
        interactive_search(models, corpus)
    
    elif args.mode == 'evaluate':
        # Mode évaluation avec LLM as a Judge
        api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
        run_evaluation(
            models=models,
            corpus=corpus,
            queries=queries,
            provider=args.provider,
            api_key=api_key,
            num_queries=args.num_queries
        )
    
    elif args.mode == 'compare':
        # Comparaison détaillée entre deux modèles
        query = args.query or queries[0]
        model_list = list(models.items())
        
        # Comparer BM25 vs TF-IDF
        compare_two_models(
            model_list[2][1], model_list[1][1],
            model_list[2][0], model_list[1][0],
            corpus, query
        )
        
        # Comparer BM25 vs LM
        compare_two_models(
            model_list[2][1], model_list[4][1],
            model_list[2][0], model_list[4][0],
            corpus, query
        )


if __name__ == "__main__":
    main()

