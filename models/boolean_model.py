# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Modèle Booléen
"""

from preprocessing import TextPreprocessor


class BooleanModel:
    """
    Modèle Booléen de Recherche d'Information.
    
    Ce modèle traite les requêtes comme des expressions booléennes.
    Les documents sont considérés comme pertinents ou non (binaire).
    Supporte les opérateurs AND, OR, NOT.
    """
    
    def __init__(self, index):
        self.index = index
        self.preprocessor = TextPreprocessor()
        self.name = "Modèle Booléen"
    
    def parse_query(self, query):
        """
        Parse une requête et retourne les termes.
        Par défaut, tous les termes sont combinés avec AND.
        """
        return self.preprocessor.preprocess(query)
    
    def search_and(self, terms):
        """
        Recherche avec opérateur AND.
        Retourne les documents contenant TOUS les termes.
        """
        if not terms:
            return set()
        
        # Commencer avec les documents du premier terme
        result = self.index.get_documents_containing(terms[0])
        
        # Intersection avec les documents des autres termes
        for term in terms[1:]:
            result = result.intersection(self.index.get_documents_containing(term))
        
        return result
    
    def search_or(self, terms):
        """
        Recherche avec opérateur OR.
        Retourne les documents contenant AU MOINS UN des termes.
        """
        result = set()
        for term in terms:
            result = result.union(self.index.get_documents_containing(term))
        return result
    
    def search_not(self, terms, excluded_terms):
        """
        Recherche avec opérateur NOT.
        Retourne les documents contenant les termes mais pas les termes exclus.
        """
        result = self.search_and(terms)
        excluded = self.search_or(excluded_terms)
        return result.difference(excluded)
    
    def search(self, query, operator='AND'):
        """
        Effectue une recherche avec le modèle booléen.
        
        Args:
            query: La requête de recherche
            operator: 'AND' ou 'OR' (par défaut 'AND')
        
        Returns:
            Liste de tuples (doc_id, score) triés par pertinence
        """
        terms = self.parse_query(query)
        
        if not terms:
            return []
        
        if operator.upper() == 'AND':
            matching_docs = self.search_and(terms)
        else:
            matching_docs = self.search_or(terms)
        
        # Score basé sur le nombre de termes correspondants
        results = []
        for doc_id in matching_docs:
            # Calculer le nombre de termes de la requête présents
            matching_terms = sum(1 for t in terms if self.index.get_term_frequency(t, doc_id) > 0)
            score = matching_terms / len(terms)  # Score normalisé entre 0 et 1
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
        # Utiliser OR pour avoir plus de résultats
        results = self.search(query, operator='OR')
        return results[:top_k]
    
    def explain_ranking(self, query, doc_id):
        """
        Explique pourquoi un document a été retourné pour une requête.
        """
        terms = self.parse_query(query)
        explanation = {
            'query_terms': terms,
            'matched_terms': [],
            'missing_terms': []
        }
        
        for term in terms:
            if self.index.get_term_frequency(term, doc_id) > 0:
                explanation['matched_terms'].append(term)
            else:
                explanation['missing_terms'].append(term)
        
        return explanation


if __name__ == "__main__":
    # Test du modèle booléen
    from corpus import get_corpus
    from preprocessing import create_index
    
    # Création de l'index
    corpus = get_corpus()
    index = create_index(corpus)
    
    # Test du modèle
    model = BooleanModel(index)
    
    query = "intelligence artificielle machine learning"
    print(f"\nRequête: {query}")
    print("-" * 50)
    
    results = model.get_ranking(query)
    for doc_id, score in results:
        print(f"  {doc_id}: score={score:.4f} - {corpus[doc_id]['title']}")

