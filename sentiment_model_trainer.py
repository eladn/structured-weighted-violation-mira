import random
from scipy import sparse

import numpy as np
import time
from itertools import islice
from random import shuffle

from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, zero_one_loss
from warnings import warn
import sys

from constants import DEBUG, DATA_PATH, STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH, DOCUMENT_LABELS, SENTENCE_LABELS, MODELS, TEST_PATH, NR_SENTENCE_LABELS
from corpus import Corpus
from document import Document
from sentence import Sentence
from corpus_features_vector import CorpusFeaturesVector
from sentiment_model_tester import SentimentModelTester
from utils import ProgressBar, print_title, shuffle_iter
from config import Config


class SentimentModelTrainer:
    def __init__(self, corpus: Corpus, features_vector: CorpusFeaturesVector, config: Config):
        config.verify()
        if DEBUG:
            if not isinstance(corpus, Corpus):
                raise Exception('The corpus argument is not a Corpus object')
        self.corpus = corpus
        self.features_vector = features_vector
        self.empirical_counts = None
        self.evaluated_feature_vectors = []
        self.w = None
        self.config = config

    def generate_features(self):
        start_time = time.time()
        # self.features_vector.initialize_features()
        self.w = np.zeros(self.features_vector.size)
        print("generate_feature done: {0:.3f} seconds".format(time.time() - start_time))
        print("Features count: {}".format(self.features_vector.size))

    def evaluate_feature_vectors(self):
        start_time = time.time()
        print("Evaluating features")
        pb = ProgressBar(len(self.corpus.documents))
        for document in self.corpus.documents:
            pb.start_next_task()
            self.evaluated_feature_vectors.append(self.evaluate_document_feature_vector(document))
        pb.finish()
        print("evaluate_feature_vectors done: {0:.3f} seconds".format(time.time() - start_time))

    def evaluate_document_feature_vector(self, document: Document, y_tag=None):
        # TODO if not structured - should not count from 1
        y_document = y_tag[0] if y_tag is not None else None
        row_ind, col_ind = [], []
        for sentence in islice(document.sentences, 1, None):  # TODO
            y_sentence = y_tag[sentence.index + 1] if y_tag is not None else None
            y_pre_sentence = y_tag[sentence.index] if y_tag is not None else None
            feature_indices = self.features_vector.evaluate_clique_feature_vector(
                document, sentence, y_document, y_sentence, y_pre_sentence)
            col_ind += list(feature_indices)  # TODO: consider stacking it all in array (size can be known)
            row_ind += [sentence.index - 1 for _ in feature_indices]
            #print(len(feature_indices))
        #print(len(col_ind))
        return csr_matrix(([1 for _ in col_ind], (row_ind, col_ind)),
                          shape=(document.count_sentences() - 1, self.features_vector.size))  # TODO

    def mira_algorithm(self, iterations=5, k_best_viterbi_labelings=2, k_random_labelings=0):
        optimization_time = time.time()
        print_title("Training model: {model}, k-best-viterbi = {k_viterbi}, k-random = {k_rnd}, iterations = {iter}".format(
            model=self.config.model_type,
            k_viterbi=k_best_viterbi_labelings,
            k_rnd=k_random_labelings,
            iter=iterations))
        pb = ProgressBar(iterations * len(self.corpus.documents))
        tester = SentimentModelTester(self.corpus.clone(), self.features_vector, self.config.model_type)
        for cur_iter in range(1, iterations+1):
            for document_nr, (document, test_document, feature_vectors) \
                in enumerate(shuffle_iter(self.corpus.documents,
                                          tester.corpus.documents,
                                          self.evaluated_feature_vectors), start=1):
                task_str = 'iter: {cur_iter}/{nr_iters} -- document: {cur_doc}/{nr_docs}'.format(
                    cur_iter=cur_iter, nr_iters=iterations, cur_doc=document_nr+1, nr_docs=len(self.corpus.documents)
                )
                pb.start_next_task(task_str)

                c = []
                if k_random_labelings > 0:
                    c = self.extract_random_labeling_subset(document, k_random_labelings, use_document_tag=False)
                if k_best_viterbi_labelings > 0:
                    tester.w = self.w
                    top_k = min(k_best_viterbi_labelings, NR_SENTENCE_LABELS ** document.count_sentences())
                    labelings = tester.viterbi_inference(test_document, top_k=top_k)
                    c += labelings
                    shuffle(c)
                assert(len(c) >= 1)

                P, q, G, h = self.extract_qp_matrices(document, feature_vectors, document.y(), c)
                w = solve_qp(P, q, G, h, solver="osqp")
                if np.any(np.equal(w, None)):
                    print(file=sys.stderr)
                    warn_msg = "QP solver returned `None`s solution vector. Weights vector `w` has not been updated. [{}]".format(task_str)
                    warn(warn_msg)
                else:
                    self.w = w
        pb.finish()
        print("Total execution time: {0:.3f} seconds".format(time.time() - optimization_time))

    def extract_qp_matrices(self, document: Document, feature_vectors, y, c):
        M = sparse.eye(self.features_vector.size)
        q = np.copy(self.w) * -1
        y_fv = feature_vectors.sum(axis=0)
        y_fv = np.asarray(y_fv).reshape((y_fv.shape[1],))
        G = None
        h = []
        for y_tag in c:
            y_tag_fv = self.evaluate_document_feature_vector(document, y_tag).sum(axis=0)
            y_tag_fv = np.asarray(y_tag_fv)
            if G is None:
                G = y_tag_fv - y_fv
            else:
                G = np.vstack((G, y_tag_fv - y_fv))

            # y_tag_loss = hamming_loss(y_true=y[1:], y_pred=y_tag[1:]) * zero_one_loss([y[0]], [y_tag[0]])
            y_tag_document_loss = zero_one_loss([y[0]], [y_tag[0]])
            y_tag_sentences_loss = hamming_loss(y_true=y[1:], y_pred=y_tag[1:])
            # y_tag_loss = y_tag_sentences_loss * y_tag_document_loss  # original loss

            y_tag_loss = y_tag_sentences_loss

            if self.config.model_type in {DOCUMENT_CLASSIFIER, STRUCTURED_JOINT}:
                if self.config.loss_type == 'mult':
                    y_tag_loss *= y_tag_document_loss
                elif self.config.loss_type == 'plus':
                    y_tag_loss += self.config.doc_loss_factor * y_tag_document_loss

            h.append(-y_tag_loss)
        G = csr_matrix(G)
        h = np.array(h).reshape(len(c), )
        return M, q.reshape(self.features_vector.size, ), G, h

    @staticmethod
    def extract_random_labeling_subset(document: Document, k: int, use_document_tag: bool=False):
        return [
            [document.label if use_document_tag else random.choice(DOCUMENT_LABELS)] +
            [random.choice(SENTENCE_LABELS) for _ in range(document.count_sentences())]
            for _ in range(k)]

    def save_model(self, model_name: str):
        np.savetxt(MODELS_PATH + model_name, self.w)

    def load_model(self, model_name: str):
        self.w = np.loadtxt(MODELS_PATH + model_name)

