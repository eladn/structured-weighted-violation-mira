
import numpy as np
import time
from itertools import chain

from constants import DEBUG, DATA_PATH, STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH, DOCUMENT_LABELS, SENTENCE_LABELS, NR_SENTENCE_LABELS, MODELS, TEST_PATH
from corpus import Corpus
from document import Document
from sentence import Sentence
from feature_vector import FeatureVector
from utils import ProgressBar, print_title


class SentimentModelTester:
    def __init__(self, corpus: Corpus, feature_vector: FeatureVector, model, w: np.ndarray=None):
        assert model in MODELS
        if DEBUG:
            if not isinstance(corpus, Corpus):
                raise Exception('The corpus argument is not a Corpus object')
        self.matrix = np.zeros((len(SENTENCE_LABELS), len(SENTENCE_LABELS)))
        self.corpus = corpus
        self.feature_vector = feature_vector
        self.w = w
        self.model = model

    def sentence_score(self, document: Document, sen_index: int, document_label, sentence_label, pre_sentence_label):
        assert sentence_label in SENTENCE_LABELS
        assert document_label is None or document_label in DOCUMENT_LABELS
        assert pre_sentence_label is None or pre_sentence_label in SENTENCE_LABELS
        fv = self.feature_vector.evaluate_clique_feature_vector(document, sen_index, self.model,
                                                                document_label=document_label,
                                                                sentence_label=sentence_label,
                                                                pre_sentence_label=pre_sentence_label)
        return np.sum(np.take(self.w, fv))

    # if `infer_document_label` is False, `document.label` will be used.
    # Otherwise, all possible document labels will be tested, and the one with
    # the maximal total document score will be assigned to `document.label`.
    def viterbi_inference(self, document: Document, infer_document_label: bool=True, top_k: int=7, assign_best_labeling=True):
        nr_sentences = document.count_sentences()
        assert(NR_SENTENCE_LABELS ** nr_sentences >= top_k)

        # Forward pass: calculate `pi` and `bp`.
        bp, pi = None, None
        if infer_document_label:
            for document_label in DOCUMENT_LABELS:
                cur_bp, cur_pi = self.viterbi_forward_pass(document, document_label=document_label, top_k=top_k)
                if (bp is None and pi is None) or (cur_pi[-1].max() > pi[-1].max()):
                    bp, pi = cur_bp, cur_pi
                    document.label = document_label
        else:
            # `document.label` will be used as document label.
            bp, pi = self.viterbi_forward_pass(document, document_label=None, top_k=top_k)

        # Backward pass: assign optimal sentence labels using calculated `pi` and `bp`.
        # TODO: fix to adapt top-k !
        labelings = [[document.label] + [None for _ in document.sentences] for _ in range(top_k)]

        for k in range(top_k):
            sentence_rank = k
            following_sentence_label_idx = 0
            for sentence_idx in range(nr_sentences - 1, -1, -1):
                # Assign the label for sentence #sentence_idx, for the k-th labeling.
                following_sentence_idx = sentence_idx + 1
                sentence_label_idx = bp[following_sentence_idx, following_sentence_label_idx, sentence_rank, 0]
                sentence_rank = bp[following_sentence_idx, following_sentence_label_idx, sentence_rank, 1]
                labelings[k][sentence_idx+1] = SENTENCE_LABELS[sentence_label_idx]
                if k == 0 and assign_best_labeling:
                    document.sentences[sentence_idx].label = SENTENCE_LABELS[sentence_label_idx]
                following_sentence_label_idx = sentence_label_idx
            assert(sentence_rank == 0)  # TODO: explain

        # last_sentence_label_index = np.where(pi[-1] == pi[-1].max())
        # document.sentences[-1].label = SENTENCE_LABELS[last_sentence_label_index[0][0]]
        #
        # for sentence_idx in range(nr_sentences - 2, -1, -1):
        #     next_sentence_label_idx = SENTENCE_LABELS.index(document.sentences[sentence_idx + 1].label)
        #     best_sentence_label_idx = bp[sentence_idx + 1, next_sentence_label_idx, 0, 0]
        #     document.sentences[sentence_idx].label = SENTENCE_LABELS[best_sentence_label_idx]

        return labelings

    def viterbi_forward_pass(self, document: Document, document_label=None, top_k: int=1):
        assert document_label is None or document_label in DOCUMENT_LABELS

        nr_sentences = document.count_sentences()
        pi = np.zeros((nr_sentences+1, NR_SENTENCE_LABELS, top_k))
        # For each sentence_idx>1, sentence_label_idx, and k in {0, ..., top_k-1}:
        #   bp[sentence_idx, sentence_label_idx, k, 0] is the pre_sentence_label_idx
        #   bp[sentence_idx, sentence_label_idx, k, 1] is the pre_sentence_rank (which is in {0, ..., top_k-1})
        # So that the k'th best rank for `sentence_idx` and `sentence_label_idx` is with the path
        # that ends in bp[sentence_idx-1, pre_sentence_label_idx, pre_sentence_rank].
        bp = np.zeros((nr_sentences+1, NR_SENTENCE_LABELS, top_k, 2), dtype=int)

        # Note:
        # We perform one more iteration, with a None sentence, only in order to calculate `bp` for last_sentence_idx+1.
        # This is going to help us finding the top_k best labelings in the backward pass.
        for sentence_idx, sentence in enumerate(chain(document.sentences, [None])):
            for sentence_label_idx, sentence_label in enumerate(SENTENCE_LABELS):
                if sentence_idx == 0:
                    # In order to avoid duplicating each final labeling vector for top_k times, we enforce that the
                    # first sentence will have positive scores only for rank=0 (for each labeling).
                    pi[sentence_idx, sentence_label_idx, :] = -np.inf
                    pi[sentence_idx, sentence_label_idx, 0] = self.sentence_score(
                        document, sentence_idx, document_label, sentence_label, pre_sentence_label=None)
                else:
                    # TODO: fix to adapt top-k !
                    # TODO: doc!
                    # TODO: test!
                    if sentence is not None:
                        # Remember that the label of the previous sentence may affect the score of the current sentence.
                        # For each possible label (for the previous sentence), evaluate the score of the current
                        # sentence, when using `sentence_label` as the label for the current sentence.
                        sentence_scores_per_pre_sentence_label = [(pre_sentence_label_idx, self.sentence_score(
                            document, sentence_idx, document_label, sentence_label, pre_sentence_label))
                                for pre_sentence_label_idx, pre_sentence_label in enumerate(SENTENCE_LABELS)]
                    else:
                        # It is the last iteration.
                        # TODO: DOC!
                        sentence_scores_per_pre_sentence_label = [
                            (pre_sentence_label_idx, 0)
                            for pre_sentence_label_idx, pre_sentence_label in enumerate(SENTENCE_LABELS)]

                    # Create a matrix of size (NR_LABELS * top_k), such that the cell M[pre_lbl_idx, pre_rank] contains
                    # the total score of the sentences 1...current, when using the (pre_rank+1)-th best path of the
                    # previous sentence when its label is lbl_pre_lbl_idx.
                    total_scores_per_pre_sentence_label_and_rank = np.array(
                        [[sentence_score + pi[sentence_idx - 1, pre_sentence_label_idx, pre_sentence_rank]
                          for pre_sentence_rank in range(top_k)]
                         for pre_sentence_label_idx, sentence_score in sentence_scores_per_pre_sentence_label])

                    # Find (flattened) indeces of top_k elements with highest score in the above created matrix,
                    #   (out of its NR_SENTENCE_LABELS * top_k elements).
                    # What is a "FLATTENED" index?
                    #   For example, in the matrix [[8 3 1] [7 9 4]], the flattened index of the element `9` is 4.
                    # Why do we use flattened indeces and not simply (i,j) indeces (this is a matrix)?
                    #   It is just much simpler to perform the following tasks with a single value index.
                    # Time complexity:
                    #   linear! O(total_scores_per_pre_sentence_label_and_rank.size) = O(NR_SENTENCE_LABELS * top_k)
                    top_k_flattened_indexes = np.argpartition(total_scores_per_pre_sentence_label_and_rank,
                                                              -top_k, axis=None)[-top_k:]

                    # Find the scores of the actual top_k elements.
                    top_k_elements = total_scores_per_pre_sentence_label_and_rank.flat[top_k_flattened_indexes]

                    # Now lets sort these best found top_k elements by their score (descending order).
                    # Time complexity: O(top_k * log(top_k))
                    # Notice: Out of (NR_SENTENCE_LABELS * top_k) elements we found and sorted the best top_k elements
                    #   using time complexity (NR_SENTENCE_LABELS * top_k) + (top_k * log(top_k)), which is optimal.
                    top_k_elements_sorting_flattened_indexes = np.argsort(top_k_elements, kind='heapsort')[::-1]
                    top_k_flattened_indexes_sorted = top_k_flattened_indexes[top_k_elements_sorting_flattened_indexes]
                    top_k_elements_sorted = top_k_elements[top_k_elements_sorting_flattened_indexes]

                    # Update the dynamic-programming arrays (scores and back-pointer) for the current sentence.
                    pi[sentence_idx, sentence_label_idx, :] = top_k_elements_sorted
                    bp[sentence_idx, sentence_label_idx, :, 0] = top_k_flattened_indexes_sorted // top_k
                    bp[sentence_idx, sentence_label_idx, :, 1] = top_k_flattened_indexes_sorted % top_k

        return bp, pi

    def document_predict(self, corpus: Corpus):
        best_document_score = None
        for document in corpus.documents:
            for document_label in DOCUMENT_LABELS:
                sum_document_score = 0
                for i, sentence in enumerate(document.sentences):
                    sum_document_score = sum_document_score + \
                                         self.sentence_score(document, i, document_label=document_label,
                                                             sentence_label=None, pre_sentence_label=None)
                if best_document_score is None or sum_document_score > best_document_score:
                    best_document_score = sum_document_score
                    document.label = document_label

    def sentence_predict(self, document: Document):
        for i, sentence in enumerate(document.sentences):
            best_sentence_score = None
            for sentence_label in SENTENCE_LABELS:
                temp_sentence_score = self.sentence_score(document, i,
                                                          document_label=None,
                                                          sentence_label=sentence_label,
                                                          pre_sentence_label=None)
                if best_sentence_score is None or temp_sentence_score > best_sentence_score:
                    best_sentence_score = temp_sentence_score
                    sentence.label = sentence_label

    def inference(self):
        assert self.w is not None

        start_time = time.time()

        if self.model == DOCUMENT_CLASSIFIER:
            self.document_predict(self.corpus)

        elif self.model == SENTENCE_CLASSIFIER:
            for document in self.corpus.documents:
                self.sentence_predict(document)

        elif self.model in {SENTENCE_STRUCTURED, STRUCTURED_JOINT}:
            pb = ProgressBar(len(self.corpus.documents))
            for i, document in enumerate(self.corpus.documents):
                pb.start_next_task()
                self.viterbi_inference(document, infer_document_label=(self.model == STRUCTURED_JOINT))
            pb.finish()

        print("Inference done: {0:.3f} seconds".format(time.time() - start_time))

    def load_model(self, model_name):
        path = MODELS_PATH
        self.w = np.loadtxt(path + model_name)

    def evaluate_model(self, ground_truth: Corpus):
        assert self.w is not None

        if self.model == SENTENCE_CLASSIFIER:
            sentences_results = {
                "correct": 0,
                "errors": 0
            }
            for d1, d2 in zip(self.corpus.documents, ground_truth.documents):
                for s1, s2 in zip(d1.sentences, d2.sentences):
                    if s1.label == s2.label:
                        sentences_results["correct"] += 1
                    else:
                        sentences_results["errors"] += 1

            return sentences_results, sentences_results["correct"] / sum(sentences_results.values())
        else:
            document_results = {
                "correct": 0,
                "errors": 0
            }
            sentences_results = {
                "correct": 0,
                "errors": 0
            }
            for d1, d2 in zip(self.corpus.documents, ground_truth.documents):

                if d1.label == d2.label:
                    document_results["correct"] += 1
                else:
                    document_results["errors"] += 1

                for s1, s2 in zip(d1.sentences, d2.sentences):
                    if s1.label == s2.label:
                        sentences_results["correct"] += 1
                    else:
                        sentences_results["errors"] += 1

            return document_results, document_results["correct"] / sum(document_results.values()), \
                sentences_results, sentences_results["correct"] / sum(sentences_results.values())

    # def print_results_to_file(self, tagged_file, model_name):
    #     path = TEST_PATH
    #
    #     f = open(path + model_name, 'w')
    #     for s_truth, s_eval in zip(tagged_file.sentences, self.corpus.sentences):
    #         f.write(" ".join(["{}_{}".format(t_truth.word, t_eval.tag) for t_truth, t_eval
    #                           in zip(s_truth.tokens, s_eval.tokens)]) + "\n")

    def confusion_matrix(self, ground_truth: Corpus, model_name):
        path = TEST_PATH

        for d1, d2 in zip(self.corpus.documents, ground_truth.documents):
            for s1, s2 in zip(d1.sentences, d2.sentences):
                self.matrix[SENTENCE_LABELS.index(s1.label)][SENTENCE_LABELS.index(s2.label)] += 1
        np.savetxt(path + "{}__confusion_matrix.txt".format(model_name), self.matrix)

        file = open(path + "{}__confusion_matrix_lines.txt".format(model_name), "w")
        for i in range(len(SENTENCE_LABELS)):
            for j in range(len(SENTENCE_LABELS)):
                value = self.matrix[i][j]
                file.write("Truth label: {}, Predicted label: {}, number of errors: {}\n".format(SENTENCE_LABELS[j],
                                                                                                 SENTENCE_LABELS[i],
                                                                                                 value))
        file.close()

    # def confusion_matrix_zeros_diagonal(self):
    #     for i in range(len(TAGS)):
    #         self.matrix[i][i] = 0

    # def confusion_matrix_ten_max_errors(self, model_name, is_test):
    #     if is_test:
    #         path = BASIC_TEST_PATH if self.is_basic else ADVANCED_TEST_PATH
    #     else:
    #         path = BASIC_COMP_PATH if self.is_basic else ADVANCED_COMP_PATH
    #
    #     self.confusion_matrix_zeros_diagonal()
    #     ten_max_values = {}
    #     file_name = "{}_confusion_matrix_ten_max_errors".format(model_name)
    #     file = open(path + file_name, "w")
    #     for k in range(10):
    #         i, j = np.unravel_index(self.matrix.argmax(), self.matrix.shape)
    #         value = self.matrix[i][j]
    #         ten_max_values[(i, j)] = value
    #         self.matrix[i][j] = 0
    #         file.write("Truth tag: {}, Predicted tag: {}, number of errors: {}\n".format(TAGS[j], TAGS[i], value))
    #     file.close()
    #     return ten_max_values

