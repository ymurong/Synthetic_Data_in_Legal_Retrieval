import sys
import logging
import itertools
from tqdm import tqdm

import math
import numpy as np
from statistics import mean
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from utils.common import log_step


class TFIDFRetriever:
    def __init__(self, retrieval_corpus):
        self.retrieval_corpus = retrieval_corpus
        self.N = len(retrieval_corpus)
        self.vocab = self._build_vocabulary()
        self.idfs = self._compute_idfs()

    def __repr__(self):
        return f"{self.__class__.__name__}".lower()

    @log_step
    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id, _ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        results = dict()
        for i, doc in enumerate(self.retrieval_corpus):
            results[i + 1] = self.score(q, doc)  # NB: '+1' because doc_ids in BSARD start at 1.
        return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def score(self, q, d):
        score = 0.0
        for t in q.split():
            score += self._compute_tfidf(t, d)
        return score

    def _build_vocabulary(self):
        return sorted(set(itertools.chain.from_iterable([doc.lower().split() for doc in self.retrieval_corpus])))

    def _compute_idfs(self):
        idfs = dict.fromkeys(self.vocab, 0)
        for word, _ in idfs.items():
            idfs[word] = self._compute_idf(word)
        return idfs

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10(self.N / (df + 1))

    def _compute_tf(self, t, d):
        return d.split().count(t)

    def _compute_tfidf(self, t, d):
        tf = self._compute_tf(t, d)
        idf = self.idfs[t] if t in self.idfs else math.log10(self.N)
        return tf * idf


class BM25Retriever(TFIDFRetriever):
    """
    When experimenting with b and k1, you should first consider their bounds. I would also suggest looking into past experiments to give you a rough feel for the type of experimentation you may be interested in doing — especially if you’re just getting into this for the first time:

    b needs to be between 0 and 1. Many experiments test values in increments of around 0.1 and most experiments seem to show the optimal b to be in a range of 0.3-0.9 (Lipani, Lupu, Hanbury, Aizawa (2015); Taylor, Zaragoza, Craswell, Robertson, Burges (2006); Trotman, Puurula, Burgess (2014); etc.)

    k1 is typically evaluated in the 0 to 3 range, though there’s nothing to stop it from being higher. Many experiments have focused on increments of 0.1 to 0.2 and most experiments seem to show the optimal k1 to be in a range of 0.5-2.0

    For k1, you should be asking, “when do we think a term is likely to be saturated?” For very long documents like books — especially fictional or multi-topic books — it’s very likely to have a lot of different terms several times in a work, even when the term isn’t highly relevant to the work as a whole. For example, “eye” or “eyes” can appear hundreds of times in a fictional book even when “eyes” are not one of the the primary subjects of the book. A book that mentions “eyes” a thousand times, though, likely has a lot more to do with eyes. You may not want terms to be saturated as quickly in this situation, so there’s some suggestion that k1 should generally trend toward larger numbers when the text is a lot longer and more diverse. For the inverse situation, it’s been suggested to set k1 on the lower side. It’s very unlikely that a collection of short news articles would have “eyes” dozens to hundreds of times without being highly related to eyes as a subject.

    For b, you should be asking, “when do we think a document is likely to be very long, and when should that hinder its relevance to a term?” Documents which are highly specific like engineering specifications or patents are lengthy in order to be more specific about a subject. Their length is unlikely to be detrimental to the relevance and b may be more appropriate to be lower. On the opposite side, documents which touch on several different topics in a broad way — news articles (a political article may touch on economics, international affairs, and certain corporations), user reviews, etc. — often benefit by choosing a larger b so that irrelevant topics to a user’s search, including spam and the like, are penalized.

    https://www.elastic.co/fr/blog/practical-bm25-part-3-considerations-for-picking-b-and-k1-in-elasticsearch
    """
    def __init__(self, retrieval_corpus, k1, b):
        super().__init__(retrieval_corpus)
        self.k1 = k1
        self.b = b
        self.avgdl = self._compute_avgdl()

    def score(self, q, d):
        score = 0.0
        for t in q.split():
            tf = self._compute_tf(t, d)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5) / 0.5)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(d.split()) / self.avgdl))
        return score

    def _compute_avgdl(self):
        return mean([len(doc.split()) for doc in self.retrieval_corpus])

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10((self.N - df + 0.5) / (df + 0.5))


class SWSNRetriever(BM25Retriever):
    def __init__(self, retrieval_corpus, k1, b, model):
        super().__init__(retrieval_corpus, k1, b)
        self.model = model

    def score(self, q, d):
        score = 0.0
        for t in d.split():
            sem = self._compute_sem(t, q)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5) / 0.5)
            score += idf * (sem * (self.k1 + 1)) / (sem + self.k1 * (1 - self.b + self.b * len(q.split()) / self.avgdl))
        return score

    def _compute_sem(self, t, q):
        term_embedding = self._get_word_embedding(t)
        query_embeddings = [self._get_word_embedding(w) for w in q.split()]
        cosines = [cosine_similarity([term_embedding], [embedding])[0, 0] for embedding in query_embeddings]
        return np.max(cosines)

    def _get_word_embedding(self, w):
        return self.model[w] if w in self.model.vocab else np.zeros(self.model.vector_size)