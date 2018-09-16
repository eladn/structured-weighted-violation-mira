import random
from scipy import sparse

import numpy as np
import time
from random import shuffle

from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, zero_one_loss
from warnings import warn
import sys

from constants import STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED, \
    DOCUMENT_LABELS, SENTENCE_LABELS, NR_SENTENCE_LABELS
from corpus import Corpus
from document import Document
from sentence import Sentence
from corpus_features_extractor import CorpusFeaturesExtractor
from sentiment_model import SentimentModel
from utils import ProgressBar, print_title, shuffle_iterate_over_batches
from sentiment_model_configuration import SentimentModelConfiguration


class SentimentModelTrainer:
    def __init__(self, train_corpus: Corpus, features_extractor: CorpusFeaturesExtractor,
                 model_config: SentimentModelConfiguration):

        if not isinstance(train_corpus, Corpus):
            raise ValueError('The `train_corpus` argument is not a `Corpus` object')
        if not isinstance(features_extractor, CorpusFeaturesExtractor):
            raise ValueError('The `features_extractor` argument is not a `CorpusFeaturesExtractor` object')
        if not isinstance(model_config, SentimentModelConfiguration):
            raise ValueError('The `model_config` argument is not a `SentimentModelConfiguration` object')
        model_config.verify()

        self.train_corpus = train_corpus
        self.features_extractor = features_extractor
        self.evaluated_feature_vectors = []
        self.model_config = model_config

    def evaluate_feature_vectors(self):
        start_time = time.time()
        print("Evaluating features")
        pb = ProgressBar(len(self.train_corpus.documents))
        for document in self.train_corpus.documents:
            pb.start_next_task()
            self.evaluated_feature_vectors.append(self.features_extractor.evaluate_document_feature_vector(document))
        pb.finish()
        print("evaluate_feature_vectors done: {0:.3f} seconds".format(time.time() - start_time))

    def fit_using_mira_algorithm(self, save_model_after_every_iteration: bool = False,
                                 datasets_to_evaluate_after_every_iteration: list = None):
        optimization_time = time.time()
        print_title("Training model: {model}, k-best-viterbi = {k_viterbi}, k-random = {k_rnd}, iterations = {iter}".format(
            model=self.model_config.model_type,
            k_viterbi=self.model_config.mira_k_best_viterbi_labelings,
            k_rnd=self.model_config.mira_k_random_labelings,
            iter=self.model_config.mira_iterations))
        nr_batchs_per_iteration = int(np.ceil(len(self.train_corpus.documents) / self.model_config.mira_batch_size))
        pb = ProgressBar(self.model_config.mira_iterations * nr_batchs_per_iteration)
        corpus_without_labels = self.train_corpus.clone(copy_document_labels=False, copy_sentence_labels=False)
        initial_weights = np.zeros(self.features_extractor.nr_features)
        model = SentimentModel(self.features_extractor, self.model_config, initial_weights)
        use_batchs = (self.model_config.mira_batch_size and self.model_config.mira_batch_size > 1)
        batch_size = self.model_config.mira_batch_size if use_batchs else 1
        for cur_iter in range(1, self.model_config.mira_iterations+1):
            for document_nr, (batch_start_idx, documents_batch, test_documents_batch, feature_vectors_batch) \
                in enumerate(shuffle_iterate_over_batches(self.train_corpus.documents,
                                                          corpus_without_labels.documents,
                                                          self.evaluated_feature_vectors,
                                                          batch_size=batch_size), start=1):
                to_doc_nr = min(self.train_corpus.count_documents(), batch_start_idx + batch_size)
                flg_infer_top_k_per_each_document_label = self.model_config.infer_document_label and (2 * cur_iter < self.model_config.mira_iterations)
                task_str = 'iter: {cur_iter}/{nr_iters} -- document{plural_if_batch}: {cur_doc_nr}{to_doc_nr}/{nr_docs}'.format(
                    cur_iter=cur_iter,
                    nr_iters=self.model_config.mira_iterations,
                    plural_if_batch='s' if use_batchs else '',
                    cur_doc_nr=batch_start_idx+1,
                    to_doc_nr='-{}'.format(to_doc_nr),
                    nr_docs=len(self.train_corpus.documents)
                )
                pb.start_next_task(task_str)

                # Generate labelings for each document.
                mira_labelings_batch = []
                for document, test_document in zip(documents_batch, test_documents_batch):
                    document_generated_labelings = []
                    if self.model_config.mira_k_random_labelings > 0:
                        document_generated_labelings = self.extract_random_labeling_subset(
                            document, self.model_config.mira_k_random_labelings)
                    if self.model_config.mira_k_best_viterbi_labelings > 0:
                        top_k = min(self.model_config.mira_k_best_viterbi_labelings, NR_SENTENCE_LABELS ** document.count_sentences())
                        viterbi_labelings = model.viterbi_inference(
                            test_document,
                            infer_document_label=self.model_config.infer_document_label,
                            top_k=top_k,
                            assign_best_labeling=False,
                            infer_top_k_per_each_document_label=flg_infer_top_k_per_each_document_label)
                        document_generated_labelings += viterbi_labelings
                        shuffle(document_generated_labelings)
                    assert(len(document_generated_labelings) >= 1)
                    mira_labelings_batch.append(document_generated_labelings)

                P, q, G, h = self.extract_qp_matrices(
                    model.w, documents_batch, feature_vectors_batch, mira_labelings_batch)
                w = solve_qp(P, q, G, h, solver="osqp")
                if np.any(np.equal(w, None)):
                    print(file=sys.stderr)
                    warn_msg = "QP solver returned `None`s solution vector. Weights vector `w` has not been updated. [{}]".format(task_str)
                    warn(warn_msg)
                else:
                    model.w = w
            if save_model_after_every_iteration:
                cnf = self.model_config.clone()
                cnf.mira_iterations = cur_iter
                model.save(cnf.model_weights_filename)

            if datasets_to_evaluate_after_every_iteration:
                print()
                for evaluation_dataset_name, evaluation_dataset in datasets_to_evaluate_after_every_iteration:
                    print_title("Model evaluation over {} set:".format(evaluation_dataset_name))

                    inferred_dataset = evaluation_dataset.clone(
                        copy_document_labels=False, copy_sentence_labels=False)
                    model.inference(inferred_dataset)

                    evaluation_set_ground_truth = evaluation_dataset.clone()
                    print(model.evaluate_model(inferred_dataset, evaluation_set_ground_truth))
        pb.finish()
        print("MIRA training completed. Total execution time: {0:.3f} seconds".format(time.time() - optimization_time))
        return model

    def extract_qp_matrices(self, w, documents_batch, feature_vectors_batch, mira_labelings_batch):
        M = sparse.eye(self.features_extractor.nr_features)
        q = np.copy(w) * -1
        nr_labelings = sum(len(labelings) for labelings in mira_labelings_batch)
        G = np.zeros((nr_labelings, self.features_extractor.nr_features))
        h = []

        next_G_line_idx = 0
        for document, feature_vectors, mira_labelings in zip(documents_batch, feature_vectors_batch, mira_labelings_batch):
            y = document.y()
            y_fv = feature_vectors.sum(axis=0)
            y_fv = np.asarray(y_fv).reshape((y_fv.shape[1],))
            for y_tag in mira_labelings:
                y_tag_fv = self.features_extractor.evaluate_document_feature_vector(document, y_tag).sum(axis=0)
                y_tag_fv = np.asarray(y_tag_fv)

                G[next_G_line_idx, :] = (y_tag_fv - y_fv)
                next_G_line_idx += 1

                # y_tag_loss = hamming_loss(y_true=y[1:], y_pred=y_tag[1:]) * zero_one_loss([y[0]], [y_tag[0]])
                # y_tag_loss = y_tag_sentences_loss * y_tag_document_loss  # original loss

                y_tag_document_loss = 0
                if self.model_config.infer_document_label:
                    y_tag_document_loss = zero_one_loss([y[0]], [y_tag[0]])

                if not self.model_config.infer_sentences_labels:
                    y_tag_loss = y_tag_document_loss
                else:
                    y_tag_sentences_loss = hamming_loss(y_true=y[1:], y_pred=y_tag[1:])
                    y_tag_loss = y_tag_sentences_loss

                    if self.model_config.infer_document_label:
                        if self.model_config.loss_type == 'mult':
                            y_tag_loss *= y_tag_document_loss
                        elif self.model_config.loss_type == 'plus':
                            y_tag_loss += self.model_config.doc_loss_factor * y_tag_document_loss
                        elif self.model_config.loss_type == 'max':
                            y_tag_loss = max(y_tag_loss, y_tag_document_loss)

                h.append(-y_tag_loss)
        G = csr_matrix(G)
        h = np.array(h).reshape(-1, )
        return M, q.reshape(self.features_extractor.nr_features, ), G, h

    def extract_random_labeling_subset(self, document: Document, k: int):
        use_document_tag = not self.model_config.infer_document_label
        return [
            [document.label if use_document_tag else random.choice(DOCUMENT_LABELS)] +
            [random.choice(SENTENCE_LABELS) for _ in range(document.count_sentences())]
            for _ in range(k)]
