# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
LLM as a Judge - Évaluation comparative des modèles
"""

import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Pour l'API OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Pour l'API Ollama (local)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LLMProvider(Enum):
    """Fournisseurs LLM supportés."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    MOCK = "mock"  # Pour les tests sans LLM


@dataclass
class JudgmentResult:
    """Résultat d'une évaluation par le LLM."""
    query: str
    model_name: str
    doc_id: str
    doc_title: str
    relevance_score: float  # 0 à 5
    reasoning: str
    is_relevant: bool  # Score >= 3


@dataclass
class ModelComparison:
    """Comparaison entre deux modèles pour une requête."""
    query: str
    model1_name: str
    model2_name: str
    winner: str  # 'model1', 'model2', ou 'tie'
    reasoning: str
    score_model1: float
    score_model2: float


class LLMJudge:
    """
    Classe pour évaluer la qualité des résultats de recherche
    en utilisant un LLM comme juge.
    """
    
    def __init__(self, provider: LLMProvider = LLMProvider.MOCK, 
                 api_key: str = None, 
                 model_name: str = None,
                 ollama_url: str = "http://localhost:11434"):
        """
        Initialise le juge LLM.
        
        Args:
            provider: Le fournisseur LLM à utiliser
            api_key: Clé API (pour OpenAI)
            model_name: Nom du modèle à utiliser
            ollama_url: URL du serveur Ollama (si provider=OLLAMA)
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.model_name = model_name
        self.ollama_url = ollama_url
        self._warning_shown = False  # Pour n'afficher l'avertissement qu'une fois
        
        # Configuration par défaut selon le provider
        if provider == LLMProvider.OPENAI:
            self.model_name = model_name or "gpt-3.5-turbo"
            if OPENAI_AVAILABLE and self.api_key:
                openai.api_key = self.api_key
        elif provider == LLMProvider.OLLAMA:
            self.model_name = model_name or "llama2"
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM avec le prompt donné."""
        
        if self.provider == LLMProvider.OPENAI:
            return self._call_openai(prompt)
        elif self.provider == LLMProvider.OLLAMA:
            return self._call_ollama(prompt)
        else:
            return self._mock_response(prompt)
    
    def _call_openai(self, prompt: str) -> str:
        """Appelle l'API OpenAI."""
        if not OPENAI_AVAILABLE:
            if not self._warning_shown:
                print("⚠️  Package openai non installé. Utilisation du mode MOCK.")
                print("   Installez avec: pip install openai")
                self._warning_shown = True
            return self._mock_response(prompt)
        
        if not self.api_key:
            if not self._warning_shown:
                print("⚠️  Clé API OpenAI non configurée. Utilisation du mode MOCK.")
                print("   Configurez avec: export OPENAI_API_KEY='votre-clé'")
                self._warning_shown = True
            return self._mock_response(prompt)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Tu es un expert en recherche d'information. Tu évalues la pertinence des documents retournés par des moteurs de recherche."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Erreur OpenAI: {e}")
            return self._mock_response(prompt)
    
    def _call_ollama(self, prompt: str) -> str:
        """Appelle le serveur Ollama local."""
        if not REQUESTS_AVAILABLE:
            if not self._warning_shown:
                print("⚠️  Package requests non installé. Utilisation du mode MOCK.")
                print("   Installez avec: pip install requests")
                self._warning_shown = True
            return self._mock_response(prompt)
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=60
            )
            return response.json().get('response', '')
        except Exception as e:
            print(f"Erreur Ollama: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Génère une réponse simulée pour les tests."""
        # Analyse simple basée sur les mots-clés
        if "évalue la pertinence" in prompt.lower():
            return json.dumps({
                "score": 4,
                "reasoning": "Le document semble pertinent par rapport à la requête basé sur les termes communs."
            })
        elif "compare" in prompt.lower():
            return json.dumps({
                "winner": "model1",
                "score_model1": 4.2,
                "score_model2": 3.8,
                "reasoning": "Le premier modèle retourne des résultats légèrement plus pertinents."
            })
        return "{}"
    
    def evaluate_document_relevance(self, query: str, doc_id: str, 
                                    doc_title: str, doc_content: str,
                                    model_name: str) -> JudgmentResult:
        """
        Évalue la pertinence d'un document pour une requête.
        
        Args:
            query: La requête de recherche
            doc_id: Identifiant du document
            doc_title: Titre du document
            doc_content: Contenu du document
            model_name: Nom du modèle qui a retourné ce document
        
        Returns:
            JudgmentResult avec le score et le raisonnement
        """
        prompt = f"""Évalue la pertinence du document suivant par rapport à la requête.

REQUÊTE: {query}

DOCUMENT:
Titre: {doc_title}
Contenu: {doc_content[:500]}...

Réponds en JSON avec le format suivant:
{{
    "score": <nombre entre 0 et 5>,
    "reasoning": "<explication de ton évaluation>"
}}

Échelle de score:
- 0: Pas du tout pertinent
- 1: Très peu pertinent
- 2: Légèrement pertinent
- 3: Modérément pertinent
- 4: Très pertinent
- 5: Parfaitement pertinent

JSON:"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            score = float(result.get('score', 0))
            reasoning = result.get('reasoning', 'Pas d\'explication fournie')
        except (json.JSONDecodeError, ValueError):
            # Fallback si le parsing échoue
            score = 3.0
            reasoning = "Évaluation automatique par défaut"
        
        return JudgmentResult(
            query=query,
            model_name=model_name,
            doc_id=doc_id,
            doc_title=doc_title,
            relevance_score=score,
            reasoning=reasoning,
            is_relevant=score >= 3
        )
    
    def compare_rankings(self, query: str, 
                        ranking1: List[Tuple[str, float]], 
                        ranking2: List[Tuple[str, float]],
                        documents: Dict,
                        model1_name: str,
                        model2_name: str) -> ModelComparison:
        """
        Compare les classements de deux modèles pour une requête.
        
        Args:
            query: La requête de recherche
            ranking1: Classement du premier modèle [(doc_id, score), ...]
            ranking2: Classement du second modèle
            documents: Dictionnaire des documents
            model1_name: Nom du premier modèle
            model2_name: Nom du second modèle
        
        Returns:
            ModelComparison avec le gagnant et le raisonnement
        """
        # Préparer les descriptions des résultats
        def format_ranking(ranking, limit=5):
            result = []
            for i, (doc_id, score) in enumerate(ranking[:limit], 1):
                doc = documents.get(doc_id, {})
                result.append(f"{i}. {doc.get('title', doc_id)} (score: {score:.4f})")
            return "\n".join(result)
        
        ranking1_str = format_ranking(ranking1)
        ranking2_str = format_ranking(ranking2)
        
        prompt = f"""Compare les résultats de deux modèles de recherche pour la requête suivante.

REQUÊTE: {query}

RÉSULTATS {model1_name}:
{ranking1_str}

RÉSULTATS {model2_name}:
{ranking2_str}

Évalue quel modèle retourne les meilleurs résultats pour cette requête.
Réponds en JSON:
{{
    "winner": "<{model1_name}|{model2_name}|tie>",
    "score_model1": <note sur 5>,
    "score_model2": <note sur 5>,
    "reasoning": "<explication détaillée>"
}}

JSON:"""
        
        response = self._call_llm(prompt)
        
        try:
            result = json.loads(response)
            winner = result.get('winner', 'tie')
            score1 = float(result.get('score_model1', 3))
            score2 = float(result.get('score_model2', 3))
            reasoning = result.get('reasoning', 'Pas d\'explication fournie')
        except (json.JSONDecodeError, ValueError):
            winner = 'tie'
            score1 = score2 = 3.0
            reasoning = "Comparaison automatique par défaut"
        
        return ModelComparison(
            query=query,
            model1_name=model1_name,
            model2_name=model2_name,
            winner=winner,
            reasoning=reasoning,
            score_model1=score1,
            score_model2=score2
        )


class ComparativeEvaluator:
    """
    Évaluateur comparatif utilisant LLM as a Judge.
    """
    
    def __init__(self, models: Dict[str, Any], documents: Dict, 
                 llm_judge: LLMJudge = None):
        """
        Initialise l'évaluateur.
        
        Args:
            models: Dictionnaire {nom: modèle}
            documents: Corpus de documents
            llm_judge: Instance de LLMJudge (utilise MOCK par défaut)
        """
        self.models = models
        self.documents = documents
        self.judge = llm_judge or LLMJudge(provider=LLMProvider.MOCK)
        
        # Stockage des résultats
        self.relevance_results = {}
        self.comparison_results = []
        self.model_scores = {name: [] for name in models.keys()}
    
    def evaluate_query(self, query: str, top_k: int = 5) -> Dict:
        """
        Évalue tous les modèles pour une requête donnée.
        
        Args:
            query: La requête de recherche
            top_k: Nombre de documents à évaluer par modèle
        
        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        results = {
            'query': query,
            'rankings': {},
            'relevance_scores': {},
            'mean_scores': {}
        }
        
        # Obtenir les classements de chaque modèle
        for model_name, model in self.models.items():
            ranking = model.get_ranking(query, top_k)
            results['rankings'][model_name] = ranking
            
            # Évaluer la pertinence de chaque document
            relevance_scores = []
            for doc_id, score in ranking:
                doc = self.documents.get(doc_id, {})
                judgment = self.judge.evaluate_document_relevance(
                    query=query,
                    doc_id=doc_id,
                    doc_title=doc.get('title', ''),
                    doc_content=doc.get('content', ''),
                    model_name=model_name
                )
                relevance_scores.append(judgment)
                
            results['relevance_scores'][model_name] = relevance_scores
            
            # Score moyen
            if relevance_scores:
                mean_score = sum(j.relevance_score for j in relevance_scores) / len(relevance_scores)
            else:
                mean_score = 0
            results['mean_scores'][model_name] = mean_score
            self.model_scores[model_name].append(mean_score)
        
        return results
    
    def run_pairwise_comparisons(self, query: str, top_k: int = 5) -> List[ModelComparison]:
        """
        Compare tous les modèles deux à deux pour une requête.
        """
        comparisons = []
        model_names = list(self.models.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1_name = model_names[i]
                model2_name = model_names[j]
                
                ranking1 = self.models[model1_name].get_ranking(query, top_k)
                ranking2 = self.models[model2_name].get_ranking(query, top_k)
                
                comparison = self.judge.compare_rankings(
                    query=query,
                    ranking1=ranking1,
                    ranking2=ranking2,
                    documents=self.documents,
                    model1_name=model1_name,
                    model2_name=model2_name
                )
                comparisons.append(comparison)
                self.comparison_results.append(comparison)
        
        return comparisons
    
    def evaluate_all_queries(self, queries: List[str], top_k: int = 5) -> Dict:
        """
        Évalue tous les modèles sur un ensemble de requêtes.
        
        Args:
            queries: Liste de requêtes
            top_k: Nombre de documents à évaluer par requête
        
        Returns:
            Rapport complet d'évaluation
        """
        all_results = []
        
        print(f"\n{'='*60}")
        print("ÉVALUATION COMPARATIVE DES MODÈLES")
        print(f"{'='*60}")
        print(f"Nombre de modèles: {len(self.models)}")
        print(f"Nombre de requêtes: {len(queries)}")
        print(f"Top-K: {top_k}")
        print(f"{'='*60}\n")
        
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] Évaluation: '{query[:50]}...'")
            
            # Évaluation de la pertinence
            result = self.evaluate_query(query, top_k)
            all_results.append(result)
            
            # Comparaisons pairées
            self.run_pairwise_comparisons(query, top_k)
        
        # Générer le rapport final
        report = self.generate_report(queries, all_results)
        
        return report
    
    def generate_report(self, queries: List[str], all_results: List[Dict]) -> Dict:
        """
        Génère un rapport d'évaluation complet.
        """
        report = {
            'summary': {},
            'per_query_results': all_results,
            'pairwise_summary': {},
            'final_ranking': []
        }
        
        # Score moyen global par modèle
        for model_name, scores in self.model_scores.items():
            if scores:
                report['summary'][model_name] = {
                    'mean_relevance': sum(scores) / len(scores),
                    'num_queries': len(scores)
                }
        
        # Résumé des comparaisons pairées
        wins = {name: 0 for name in self.models.keys()}
        for comp in self.comparison_results:
            if comp.winner in wins:
                wins[comp.winner] += 1
        
        report['pairwise_summary'] = {
            'wins': wins,
            'total_comparisons': len(self.comparison_results)
        }
        
        # Classement final
        final_scores = []
        for model_name in self.models.keys():
            mean_rel = report['summary'].get(model_name, {}).get('mean_relevance', 0)
            win_rate = wins.get(model_name, 0) / max(1, len(self.comparison_results))
            # Score combiné: 70% pertinence + 30% victoires
            combined = 0.7 * mean_rel + 0.3 * (win_rate * 5)
            final_scores.append((model_name, combined, mean_rel, wins.get(model_name, 0)))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        report['final_ranking'] = final_scores
        
        return report
    
    def print_report(self, report: Dict):
        """Affiche le rapport de manière formatée."""
        print("\n" + "=" * 70)
        print("RAPPORT D'ÉVALUATION COMPARATIVE")
        print("=" * 70)
        
        print("\n📊 SCORES MOYENS DE PERTINENCE (sur 5):")
        print("-" * 50)
        for model_name, data in report['summary'].items():
            print(f"  {model_name}: {data['mean_relevance']:.2f}")
        
        print("\n🏆 VICTOIRES EN COMPARAISONS PAIRÉES:")
        print("-" * 50)
        for model_name, win_count in report['pairwise_summary']['wins'].items():
            print(f"  {model_name}: {win_count} victoires")
        
        print("\n🥇 CLASSEMENT FINAL:")
        print("-" * 50)
        for rank, (model_name, combined, mean_rel, wins) in enumerate(report['final_ranking'], 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"  {medal} {rank}. {model_name}")
            print(f"      Score combiné: {combined:.2f} | Pertinence: {mean_rel:.2f} | Victoires: {wins}")
        
        print("\n" + "=" * 70)


# Fonction utilitaire pour une évaluation rapide
def quick_evaluate(models: Dict, documents: Dict, queries: List[str], 
                   provider: LLMProvider = LLMProvider.MOCK,
                   api_key: str = None) -> Dict:
    """
    Fonction utilitaire pour une évaluation rapide.
    
    Args:
        models: Dictionnaire {nom: modèle}
        documents: Corpus de documents
        queries: Liste de requêtes de test
        provider: Fournisseur LLM
        api_key: Clé API (optionnel)
    
    Returns:
        Rapport d'évaluation
    """
    judge = LLMJudge(provider=provider, api_key=api_key)
    evaluator = ComparativeEvaluator(models, documents, judge)
    report = evaluator.evaluate_all_queries(queries)
    evaluator.print_report(report)
    return report


if __name__ == "__main__":
    # Test du système d'évaluation
    from corpus import get_corpus, get_test_queries
    from preprocessing import create_index
    from models.boolean_model import BooleanModel
    from models.vector_model import VectorModel
    from models.probabilistic_model import ProbabilisticModel
    from models.language_model import LanguageModel
    
    # Charger le corpus
    corpus = get_corpus()
    queries = get_test_queries()[:3]  # Seulement 3 requêtes pour le test
    
    # Créer l'index
    index = create_index(corpus)
    
    # Initialiser les modèles
    models = {
        "Booléen": BooleanModel(index),
        "Vectoriel": VectorModel(index),
        "BM25": ProbabilisticModel(index),
        "LM": LanguageModel(index)
    }
    
    # Évaluation avec le juge simulé (MOCK)
    print("Test avec juge MOCK (simulé)...")
    report = quick_evaluate(models, corpus, queries)

