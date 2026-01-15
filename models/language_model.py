# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Modèle de Langue (Language Model)
"""

import math
from collections import Counter
from preprocessing import TextPreprocessor


class LanguageModel:
    """
    Modèle de Langue pour la Recherche d'Information.
    
    Ce modèle estime la probabilité qu'une requête soit générée par le
    modèle de langue d'un document. Utilise le lissage de Jelinek-Mercer
    pour interpoler entre le modèle du document et le modèle du corpus.
    """
    
    def __init__(self, index, lambda_param=0.5):
        """
        Initialise le modèle de langue.
        
        Args:
            index: L'index inversé
            lambda_param: Paramètre de lissage Jelinek-Mercer (0 à 1)
                         λ proche de 1 = plus de poids au document
                         λ proche de 0 = plus de poids au corpus
        """
        self.index = index
        self.preprocessor = TextPreprocessor()
        self.lambda_param = lambda_param
        self.name = f"Modèle de Langue (λ={lambda_param})"
        
        # Calcul du modèle de langue du corpus
        self.corpus_term_counts = Counter()
        self.corpus_total_terms = 0
        self._build_corpus_model()
    
    def _build_corpus_model(self):
        """Construit le modèle de langue du corpus entier."""
        for doc_id, term_freqs in self.index.doc_term_freqs.items():
            for term, freq in term_freqs.items():
                self.corpus_term_counts[term] += freq
                self.corpus_total_terms += freq
    
    def get_corpus_probability(self, term):
        """
        Calcule P(t|C) - probabilité du terme dans le corpus.
        """
        count = self.corpus_term_counts.get(term, 0)
        if self.corpus_total_terms == 0:
            return 0
        return count / self.corpus_total_terms
    
    def get_document_probability(self, term, doc_id):
        """
        Calcule P(t|D) - probabilité du terme dans le document.
        """
        tf = self.index.get_term_frequency(term, doc_id)
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0
        
        return tf / doc_length
    
    def get_smoothed_probability(self, term, doc_id):
        """
        Calcule la probabilité lissée avec Jelinek-Mercer.
        
        P(t|D,C) = λ * P(t|D) + (1-λ) * P(t|C)
        """
        p_doc = self.get_document_probability(term, doc_id)
        p_corpus = self.get_corpus_probability(term)
        
        return self.lambda_param * p_doc + (1 - self.lambda_param) * p_corpus
    
    def compute_query_likelihood(self, terms, doc_id):
        """
        Calcule la vraisemblance de la requête étant donné le modèle du document.
        
        P(Q|D) = Π P(t|D) pour tous les termes t dans Q
        
        En pratique, on utilise le log pour éviter les problèmes numériques:
        log P(Q|D) = Σ log P(t|D)
        """
        log_likelihood = 0
        
        for term in terms:
            prob = self.get_smoothed_probability(term, doc_id)
            
            if prob > 0:
                log_likelihood += math.log(prob)
            else:
                # Terme non présent ni dans le document ni dans le corpus
                # Utiliser une très petite valeur pour éviter log(0)
                log_likelihood += math.log(1e-10)
        
        return log_likelihood
    
    def search(self, query):
        """
        Effectue une recherche avec le modèle de langue.
        
        Args:
            query: La requête de recherche
        
        Returns:
            Liste de tuples (doc_id, score) triés par vraisemblance décroissante
        """
        terms = self.preprocessor.preprocess(query)
        
        if not terms:
            return []
        
        results = []
        for doc_id in self.index.doc_lengths.keys():
            score = self.compute_query_likelihood(terms, doc_id)
            results.append((doc_id, score))
        
        # Trier par score décroissant (log-vraisemblance)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_ranking(self, query, top_k=10):
        """
        Retourne le classement des documents pour une requête.
        
        Args:
            query: La requête de recherche
            top_k: Nombre maximum de résultats
        
        Returns:
            Liste de tuples (doc_id, score)
        """
        results = self.search(query)
        return results[:top_k]
    
    def explain_ranking(self, query, doc_id):
        """
        Explique le score d'un document pour une requête.
        """
        terms = self.preprocessor.preprocess(query)
        
        explanation = {
            'query_terms': terms,
            'lambda': self.lambda_param,
            'doc_length': self.index.doc_lengths.get(doc_id, 0),
            'term_probabilities': {}
        }
        
        for term in terms:
            p_doc = self.get_document_probability(term, doc_id)
            p_corpus = self.get_corpus_probability(term)
            p_smoothed = self.get_smoothed_probability(term, doc_id)
            
            explanation['term_probabilities'][term] = {
                'P(t|D)': p_doc,
                'P(t|C)': p_corpus,
                'P_smoothed': p_smoothed,
                'log_prob': math.log(p_smoothed) if p_smoothed > 0 else float('-inf')
            }
        
        return explanation


class DirichletLanguageModel(LanguageModel):
    """
    Modèle de Langue avec lissage de Dirichlet.
    
    Le lissage de Dirichlet adapte automatiquement le paramètre de lissage
    en fonction de la longueur du document.
    """
    
    def __init__(self, index, mu=2000):
        """
        Args:
            index: L'index inversé
            mu: Paramètre de Dirichlet (typiquement entre 1000 et 2500)
        """
        super().__init__(index, lambda_param=0)  # lambda non utilisé
        self.mu = mu
        self.name = f"Modèle de Langue Dirichlet (μ={mu})"
    
    def get_smoothed_probability(self, term, doc_id):
        """
        Calcule la probabilité avec lissage de Dirichlet.
        
        P(t|D) = (tf + μ * P(t|C)) / (|D| + μ)
        """
        tf = self.index.get_term_frequency(term, doc_id)
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        p_corpus = self.get_corpus_probability(term)
        
        numerator = tf + self.mu * p_corpus
        denominator = doc_length + self.mu
        
        if denominator == 0:
            return 0
        
        return numerator / denominator


if __name__ == "__main__":
    # Test du modèle de langue
    from corpus import get_corpus
    from preprocessing import create_index
    
    # Création de l'index
    corpus = get_corpus()
    index = create_index(corpus)
    
    # Test du modèle Jelinek-Mercer
    print("\n" + "=" * 60)
    print("Modèle de Langue (Jelinek-Mercer)")
    print("=" * 60)
    model_jm = LanguageModel(index, lambda_param=0.7)
    
    query = "intelligence artificielle machine learning"
    print(f"\nRequête: {query}")
    print("-" * 50)
    
    results = model_jm.get_ranking(query)
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f} - {corpus[doc_id]['title']}")
    
    # Test du modèle Dirichlet
    print("\n" + "=" * 60)
    print("Modèle de Langue (Dirichlet)")
    print("=" * 60)
    model_dir = DirichletLanguageModel(index, mu=2000)
    
    print(f"\nRequête: {query}")
    print("-" * 50)
    
    results = model_dir.get_ranking(query)
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f} - {corpus[doc_id]['title']}")

