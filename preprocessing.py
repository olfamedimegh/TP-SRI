# -*- coding: utf-8 -*-
"""
TP2 - Système de Recherche d'Information
Module de prétraitement des documents
"""

import re
import math
from collections import Counter, defaultdict

# Liste de stop words en français
FRENCH_STOP_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux',
    'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
    'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
    'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
    'qui', 'que', 'quoi', 'dont', 'où', 'quel', 'quelle', 'quels', 'quelles',
    'et', 'ou', 'mais', 'donc', 'car', 'ni', 'or', 'si', 'comme',
    'pour', 'par', 'avec', 'sans', 'sous', 'sur', 'dans', 'en', 'vers',
    'chez', 'entre', 'parmi', 'contre', 'avant', 'après', 'pendant',
    'être', 'avoir', 'faire', 'pouvoir', 'vouloir', 'devoir', 'savoir',
    'est', 'sont', 'était', 'été', 'a', 'ont', 'avait', 'eu',
    'fait', 'peut', 'peuvent', 'doit', 'doivent', 'sait', 'savent',
    'plus', 'moins', 'très', 'bien', 'mal', 'trop', 'peu', 'assez',
    'tout', 'tous', 'toute', 'toutes', 'autre', 'autres', 'même', 'mêmes',
    'aussi', 'ainsi', 'alors', 'encore', 'toujours', 'jamais', 'déjà',
    'ne', 'pas', 'point', 'rien', 'personne', 'aucun', 'aucune',
    'y', 'ci', 'là', 'ceci', 'cela', 'ça',
    'quand', 'comment', 'pourquoi', 'combien',
    'd', 'l', 'n', 's', 'c', 'j', 'qu', 'm', 't'
}


class TextPreprocessor:
    """Classe pour le prétraitement de texte."""
    
    def __init__(self, stop_words=None, min_word_length=2):
        self.stop_words = stop_words or FRENCH_STOP_WORDS
        self.min_word_length = min_word_length
    
    def tokenize(self, text):
        """Tokenise un texte en mots."""
        # Conversion en minuscules
        text = text.lower()
        # Suppression de la ponctuation et extraction des mots
        tokens = re.findall(r'\b[a-zàâäéèêëïîôùûüœæç]+\b', text)
        return tokens
    
    def remove_stop_words(self, tokens):
        """Supprime les stop words."""
        return [t for t in tokens if t not in self.stop_words]
    
    def filter_by_length(self, tokens):
        """Filtre les tokens trop courts."""
        return [t for t in tokens if len(t) >= self.min_word_length]
    
    def preprocess(self, text):
        """Pipeline complet de prétraitement."""
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.filter_by_length(tokens)
        return tokens


class InvertedIndex:
    """Index inversé pour la recherche d'information."""
    
    def __init__(self):
        self.index = defaultdict(dict)  # terme -> {doc_id: tf}
        self.doc_lengths = {}  # doc_id -> longueur
        self.doc_term_freqs = {}  # doc_id -> {term: freq}
        self.total_docs = 0
        self.avg_doc_length = 0
        self.vocabulary = set()
        self.preprocessor = TextPreprocessor()
    
    def build_index(self, documents):
        """Construit l'index inversé à partir d'un corpus."""
        self.total_docs = len(documents)
        total_length = 0
        
        for doc_id, doc in documents.items():
            # Prétraitement du texte
            text = doc.get('title', '') + ' ' + doc.get('content', '')
            tokens = self.preprocessor.preprocess(text)
            
            # Calcul des fréquences de termes
            term_freqs = Counter(tokens)
            self.doc_term_freqs[doc_id] = dict(term_freqs)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # Mise à jour de l'index inversé
            for term, freq in term_freqs.items():
                self.index[term][doc_id] = freq
                self.vocabulary.add(term)
        
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        print(f"Index construit: {len(self.vocabulary)} termes uniques, {self.total_docs} documents")
    
    def get_document_frequency(self, term):
        """Retourne le nombre de documents contenant le terme."""
        return len(self.index.get(term, {}))
    
    def get_term_frequency(self, term, doc_id):
        """Retourne la fréquence d'un terme dans un document."""
        return self.index.get(term, {}).get(doc_id, 0)
    
    def get_documents_containing(self, term):
        """Retourne les IDs des documents contenant le terme."""
        return set(self.index.get(term, {}).keys())
    
    def compute_idf(self, term):
        """Calcule l'IDF d'un terme."""
        df = self.get_document_frequency(term)
        if df == 0:
            return 0
        return math.log(self.total_docs / df)
    
    def compute_tfidf(self, term, doc_id):
        """Calcule le TF-IDF d'un terme dans un document."""
        tf = self.get_term_frequency(term, doc_id)
        if tf == 0:
            return 0
        # TF normalisé (1 + log(tf))
        tf_weight = 1 + math.log(tf)
        idf = self.compute_idf(term)
        return tf_weight * idf


def create_index(documents):
    """Fonction utilitaire pour créer un index."""
    index = InvertedIndex()
    index.build_index(documents)
    return index

