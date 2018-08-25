
import numpy as np
import time
from itertools import chain

from constants import DEBUG, DATA_PATH, STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH, DOCUMENT_LABELS, SENTENCE_LABELS, NR_SENTENCE_LABELS, MODELS, TEST_PATH
from corpus import Corpus
from document import Document
from sentence import Sentence
from feature_vector import FeatureVector
from utils import get_sorted_highest_k_elements_in_matrix, ProgressBar, print_title


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

    # Viterbi is a dynamic-programming algorithm. It finds the top_k sentence labelings yields the highest scores.
    # If `infer_document_label` is False, `document.label` will be used.
    # Otherwise, all possible document labels will be tested, and the one with
    # the maximal total document score will be assigned to `document.label`.
    def viterbi_inference(self, document: Document, infer_document_label: bool=True, top_k: int=1, assign_best_labeling=True):
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

        # Backward pass: Assign the top_k sentence labelings (with highest scores) using calculated scores
        # array `pi` and back-pointers array `bp`.

        # Initialize top_k labelings, each containing the document label and `None`s for sentences labels.
        labelings = [[document.label] + [None for _ in document.sentences] for _ in range(top_k)]

        # For each rank, trace back (using the back-pointers) the labeling that yields the k-th highest score.
        for rank in range(top_k):
            sentence_rank = rank
            following_sentence_label_idx = 0
            for sentence_idx in range(nr_sentences - 1, -1, -1):
                # We use the back-pointers of following sentence #(sentence_idx+1) for tracing back the label&rank of
                # the sentence #sentence_idx.
                # Note that it is ok to access the cells at #(last_sentence_idx+1), because we made sure to calculate
                # it during the forward pass. It helps us finding the starting of the tracing of the k-th best labeling.
                following_sentence_idx = sentence_idx + 1
                sentence_label_idx = bp[following_sentence_idx, following_sentence_label_idx, sentence_rank, 0]
                sentence_rank = bp[following_sentence_idx, following_sentence_label_idx, sentence_rank, 1]
                labelings[rank][sentence_idx+1] = SENTENCE_LABELS[sentence_label_idx]
                if rank == 0 and assign_best_labeling:
                    document.sentences[sentence_idx].label = SENTENCE_LABELS[sentence_label_idx]
                following_sentence_label_idx = sentence_label_idx

            # The first sentence is assigned (by the forward pass) with -inf scores for each rank>0 (explained in
            #   the implementation of the forward pass method).
            # Hence any labeling path is expected to track back to rank=0 for the first sentence.
            assert(sentence_rank == 0)

        return labelings

    def viterbi_forward_pass(self, document: Document, document_label=None, top_k: int=1):
        assert document_label is None or document_label in DOCUMENT_LABELS

        # Viterbi is a dynamic-programming algorithm. It uses two arrays in order to store intermediate
        # information. One array is called `pi` and used for storing partial sums of sentences scores. The
        # second array is called `bp` (back-pointers) and stores for each sentence and label the label and
        # rank of the previous sentence that is used to achieve the score stored in `pi` in the same index.
        # For each sentence_idx>1, sentence_label_idx, and k in {0, ..., top_k-1}:
        #   bp[sentence_idx, sentence_label_idx, k, 0] is the pre_sentence_label_idx
        #   bp[sentence_idx, sentence_label_idx, k, 1] is the pre_sentence_rank (which is in {0, ..., top_k-1})
        # So that the k'th best rank for `sentence_idx` and `sentence_label_idx` is achieved via the path
        #   that tracks back to bp[sentence_idx-1, pre_sentence_label_idx, pre_sentence_rank].
        nr_sentences = document.count_sentences()
        pi = np.zeros((nr_sentences+1, NR_SENTENCE_LABELS, top_k))
        bp = np.zeros((nr_sentences+1, NR_SENTENCE_LABELS, top_k, 2), dtype=int)

        # Note:
        # We perform one more iteration, with a None sentence, only in order to calculate `bp` for last_sentence_idx+1.
        # This is going to help us finding the top_k best labelings paths in the backward pass.
        for sentence_idx, sentence in enumerate(chain(document.sentences, [None])):
            for sentence_label_idx, sentence_label in enumerate(SENTENCE_LABELS):
                if sentence_idx == 0:
                    # In order to avoid duplicating each final labeling vector for top_k times, we enforce that the
                    # first sentence will have positive scores only for rank=0 (for each labeling).
                    pi[sentence_idx, sentence_label_idx, :] = -np.inf
                    pi[sentence_idx, sentence_label_idx, 0] = self.sentence_score(
                        document, sentence_idx, document_label, sentence_label, pre_sentence_label=None)
                    continue

                if sentence is not None:
                    # Remember that the label of the previous sentence may affect the score of the current sentence.
                    # For each possible label (for the previous sentence), evaluate the score of the current
                    # sentence, when using `sentence_label` as the label for the current sentence.
                    sentence_scores_per_pre_sentence_label = [(pre_sentence_label_idx, self.sentence_score(
                        document, sentence_idx, document_label, sentence_label, pre_sentence_label))
                            for pre_sentence_label_idx, pre_sentence_label in enumerate(SENTENCE_LABELS)]
                else:
                    # It is the last iteration of the outer loop (for sentence_idx = last_sentence_idx+1).
                    # We perform this iteration just in order to calculate `bp` for that index, so that the backward
                    # pass could easily track back the top_k paths, starting at the last sentence.
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

                # Find indeces (and values) of top_k elements with highest score in the above created matrix.
                # Notice: Out of (NR_SENTENCE_LABELS * top_k) elements we find and sort the best top_k elements
                #   using time complexity (NR_SENTENCE_LABELS * top_k) + (top_k * log(top_k)), which is optimal.
                top_k_row_indexes_sorted, top_k_col_indexes_sorted, top_k_elements_sorted = \
                    get_sorted_highest_k_elements_in_matrix(total_scores_per_pre_sentence_label_and_rank, top_k)

                # Update the dynamic-programming intermediate results arrays (scores and back-pointer).
                pi[sentence_idx, sentence_label_idx, :] = top_k_elements_sorted
                bp[sentence_idx, sentence_label_idx, :, 0] = top_k_row_indexes_sorted
                bp[sentence_idx, sentence_label_idx, :, 1] = top_k_col_indexes_sorted

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

