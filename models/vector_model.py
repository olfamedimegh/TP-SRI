# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Modèle Vectoriel (TF-IDF avec Similarité Cosinus)
"""

import math
from preprocessing import TextPreprocessor


class VectorModel:
    """
    Modèle Vectoriel de Recherche d'Information.
    
    Ce modèle représente les documents et les requêtes comme des vecteurs
    dans un espace multidimensionnel où chaque dimension correspond à un terme.
    La pertinence est mesurée par la similarité cosinus entre les vecteurs.
    """
    
    def __init__(self, index):
        self.index = index
        self.preprocessor = TextPreprocessor()
        self.name = "Modèle Vectoriel (TF-IDF)"
        
        # Cache des vecteurs de documents pour optimisation
        self.doc_vectors = {}
        self.doc_norms = {}
        self._precompute_doc_vectors()
    
    def _precompute_doc_vectors(self):
        """Précalcule les vecteurs TF-IDF des documents."""
        for doc_id in self.index.doc_lengths.keys():
            vector = {}
            norm_squared = 0
            
            for term in self.index.doc_term_freqs.get(doc_id, {}).keys():
                tfidf = self.index.compute_tfidf(term, doc_id)
                if tfidf > 0:
                    vector[term] = tfidf
                    norm_squared += tfidf ** 2
            
            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = math.sqrt(norm_squared) if norm_squared > 0 else 0
    
    def compute_query_vector(self, query):
        """
        Calcule le vecteur TF-IDF de la requête.
        """
        terms = self.preprocessor.preprocess(query)
        
        if not terms:
            return {}, 0
        
        # Calcul des fréquences dans la requête
        term_freqs = {}
        for term in terms:
            term_freqs[term] = term_freqs.get(term, 0) + 1
        
        # Vecteur TF-IDF de la requête
        vector = {}
        norm_squared = 0
        
        for term, freq in term_freqs.items():
            # TF normalisé pour la requête
            tf_weight = 1 + math.log(freq) if freq > 0 else 0
            idf = self.index.compute_idf(term)
            tfidf = tf_weight * idf
            
            if tfidf > 0:
                vector[term] = tfidf
                norm_squared += tfidf ** 2
        
        norm = math.sqrt(norm_squared) if norm_squared > 0 else 0
        
        return vector, norm
    
    def cosine_similarity(self, query_vector, query_norm, doc_id):
        """
        Calcule la similarité cosinus entre la requête et un document.
        
        cos(q, d) = (q · d) / (||q|| * ||d||)
        """
        doc_vector = self.doc_vectors.get(doc_id, {})
        doc_norm = self.doc_norms.get(doc_id, 0)
        
        if query_norm == 0 or doc_norm == 0:
            return 0
        
        # Produit scalaire
        dot_product = 0
        for term, query_weight in query_vector.items():
            if term in doc_vector:
                dot_product += query_weight * doc_vector[term]
        
        # Similarité cosinus
        similarity = dot_product / (query_norm * doc_norm)
        
        return similarity
    
    def search(self, query):
        """
        Effectue une recherche avec le modèle vectoriel.
        
        Args:
            query: La requête de recherche
        
        Returns:
            Liste de tuples (doc_id, score) triés par similarité décroissante
        """
        query_vector, query_norm = self.compute_query_vector(query)
        
        if not query_vector:
            return []
        
        results = []
        for doc_id in self.index.doc_lengths.keys():
            score = self.cosine_similarity(query_vector, query_norm, doc_id)
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
        Explique le score d'un document pour une requête.
        """
        query_vector, query_norm = self.compute_query_vector(query)
        doc_vector = self.doc_vectors.get(doc_id, {})
        
        explanation = {
            'query_terms': list(query_vector.keys()),
            'query_weights': query_vector,
            'query_norm': query_norm,
            'doc_norm': self.doc_norms.get(doc_id, 0),
            'term_contributions': {}
        }
        
        for term, query_weight in query_vector.items():
            doc_weight = doc_vector.get(term, 0)
            contribution = query_weight * doc_weight
            explanation['term_contributions'][term] = {
                'query_weight': query_weight,
                'doc_weight': doc_weight,
                'contribution': contribution
            }
        
        return explanation


if __name__ == "__main__":
    # Test du modèle vectoriel
    from corpus import get_corpus
    from preprocessing import create_index
    
    # Création de l'index
    corpus = get_corpus()
    index = create_index(corpus)
    
    # Test du modèle
    model = VectorModel(index)
    
    query = "intelligence artificielle machine learning"
    print(f"\nRequête: {query}")
    print("-" * 50)
    
    results = model.get_ranking(query)
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f} - {corpus[doc_id]['title']}")

