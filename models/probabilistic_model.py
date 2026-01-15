# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Modèle Probabiliste (BM25)
"""

import math
from preprocessing import TextPreprocessor


class ProbabilisticModel:
    """
    Modèle Probabiliste BM25 (Best Matching 25).
    
    BM25 est une fonction de scoring basée sur le framework probabiliste.
    Elle améliore le TF-IDF en ajoutant des paramètres de saturation (k1)
    et de normalisation de longueur (b).
    """
    
    def __init__(self, index, k1=1.5, b=0.75):
        """
        Initialise le modèle BM25.
        
        Args:
            index: L'index inversé
            k1: Paramètre de saturation du TF (typiquement 1.2 à 2.0)
            b: Paramètre de normalisation de longueur (0 = pas de normalisation, 1 = normalisation complète)
        """
        self.index = index
        self.preprocessor = TextPreprocessor()
        self.name = f"Modèle Probabiliste (BM25, k1={k1}, b={b})"
        
        # Paramètres BM25
        self.k1 = k1
        self.b = b
        
        # Précalcul des IDF
        self.idf_cache = {}
        self._precompute_idf()
    
    def _precompute_idf(self):
        """Précalcule les scores IDF pour tous les termes."""
        N = self.index.total_docs
        
        for term in self.index.vocabulary:
            df = self.index.get_document_frequency(term)
            # Formule IDF de BM25
            # IDF = log((N - df + 0.5) / (df + 0.5))
            self.idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def compute_bm25_score(self, terms, doc_id):
        """
        Calcule le score BM25 d'un document pour une liste de termes.
        
        BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
        
        où:
        - f(qi, D) = fréquence du terme qi dans le document D
        - |D| = longueur du document D
        - avgdl = longueur moyenne des documents
        """
        score = 0
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.index.avg_doc_length
        
        for term in terms:
            if term not in self.idf_cache:
                continue
            
            # Fréquence du terme dans le document
            tf = self.index.get_term_frequency(term, doc_id)
            
            if tf == 0:
                continue
            
            # IDF
            idf = self.idf_cache[term]
            
            # Normalisation de la longueur
            length_norm = 1 - self.b + self.b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1
            
            # Score BM25 pour ce terme
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    def search(self, query):
        """
        Effectue une recherche avec le modèle BM25.
        
        Args:
            query: La requête de recherche
        
        Returns:
            Liste de tuples (doc_id, score) triés par score décroissant
        """
        terms = self.preprocessor.preprocess(query)
        
        if not terms:
            return []
        
        results = []
        for doc_id in self.index.doc_lengths.keys():
            score = self.compute_bm25_score(terms, doc_id)
            if score > 0:
                results.append((doc_id, score))
        
        # Trier par score décroissant
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
        Explique le score BM25 d'un document pour une requête.
        """
        terms = self.preprocessor.preprocess(query)
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        
        explanation = {
            'query_terms': terms,
            'doc_length': doc_length,
            'avg_doc_length': self.index.avg_doc_length,
            'k1': self.k1,
            'b': self.b,
            'term_scores': {}
        }
        
        for term in terms:
            tf = self.index.get_term_frequency(term, doc_id)
            idf = self.idf_cache.get(term, 0)
            df = self.index.get_document_frequency(term)
            
            length_norm = 1 - self.b + self.b * (doc_length / self.index.avg_doc_length) if self.index.avg_doc_length > 0 else 1
            
            if tf > 0:
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * length_norm
                term_score = idf * (numerator / denominator)
            else:
                term_score = 0
            
            explanation['term_scores'][term] = {
                'tf': tf,
                'df': df,
                'idf': idf,
                'term_score': term_score
            }
        
        return explanation


class BM25Plus(ProbabilisticModel):
    """
    Variante BM25+ qui ajoute un terme delta pour éviter les scores négatifs
    et améliorer les performances sur les documents longs.
    """
    
    def __init__(self, index, k1=1.5, b=0.75, delta=1.0):
        super().__init__(index, k1, b)
        self.delta = delta
        self.name = f"BM25+ (k1={k1}, b={b}, δ={delta})"
    
    def compute_bm25_score(self, terms, doc_id):
        """
        Score BM25+ = BM25 + delta * (nombre de termes correspondants)
        """
        base_score = super().compute_bm25_score(terms, doc_id)
        
        # Ajouter le bonus delta pour chaque terme correspondant
        matching_terms = sum(
            1 for term in terms 
            if self.index.get_term_frequency(term, doc_id) > 0
        )
        
        return base_score + self.delta * matching_terms


if __name__ == "__main__":
    # Test du modèle probabiliste
    from corpus import get_corpus
    from preprocessing import create_index
    
    # Création de l'index
    corpus = get_corpus()
    index = create_index(corpus)
    
    # Test du modèle BM25
    model = ProbabilisticModel(index)
    
    query = "intelligence artificielle machine learning"
    print(f"\nRequête: {query}")
    print("-" * 50)
    
    results = model.get_ranking(query)
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f} - {corpus[doc_id]['title']}")
    
    # Test de l'explication
    print("\n" + "=" * 50)
    print("Explication du score pour doc1:")
    explanation = model.explain_ranking(query, "doc1")
    for term, details in explanation['term_scores'].items():
        print(f"  {term}: tf={details['tf']}, idf={details['idf']:.4f}, score={details['term_score']:.4f}")

