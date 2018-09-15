
import numpy as np
import time
from itertools import chain

from constants import STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH, DOCUMENT_LABELS, SENTENCE_LABELS, NR_SENTENCE_LABELS, \
    NR_DOCUMENT_LABELS, TEST_PATH
from corpus import Corpus
from document import Document
from sentence import Sentence
from corpus_features_extractor import CorpusFeaturesExtractor
from sentiment_model_configuration import SentimentModelConfiguration
from utils import get_sorted_highest_k_elements_in_matrix, ProgressBar


class SentimentModel:
    def __init__(self, features_extractor: CorpusFeaturesExtractor,
                 model_config: SentimentModelConfiguration, w: np.ndarray=None):

        if not isinstance(features_extractor, CorpusFeaturesExtractor):
            raise ValueError('The `features_extractor` argument is not a `CorpusFeaturesExtractor` object')
        if not isinstance(model_config, SentimentModelConfiguration):
            raise ValueError('The `model_config` argument is not a `SentimentModelConfiguration` object')
        model_config.verify()

        self.features_extractor = features_extractor
        self.model_config = model_config
        self.w = w

    def sentence_score(self, document: Document, sentence: Sentence,
                       document_label=None, sentence_label=None, pre_sentence_label=None):
        assert sentence_label is None or sentence_label in SENTENCE_LABELS
        assert document_label is None or document_label in DOCUMENT_LABELS
        assert pre_sentence_label is None or pre_sentence_label in SENTENCE_LABELS
        fv = self.features_extractor.evaluate_clique_feature_vector(
            document, sentence, document_label, sentence_label, pre_sentence_label)
        return np.sum(np.take(self.w, fv))

    # Viterbi is a dynamic-programming algorithm. It finds the top_k sentence labelings yields the highest scores.
    # If `infer_document_label` is False, `document.label` will be used.
    # Otherwise, all possible document labels will be tested, and the one with
    # the maximal total document score will be assigned to `document.label`.
    def viterbi_inference(self, document: Document, infer_document_label: bool = True, top_k: int = 1,
                          assign_best_labeling: bool = True, infer_top_k_per_each_document_label: bool = False):
        assert(not infer_top_k_per_each_document_label or infer_document_label)
        nr_sentences = document.count_sentences()
        assert(NR_SENTENCE_LABELS ** nr_sentences >= top_k)

        # Forward pass: calculate `pi` and `bp`.
        bp, pi = None, None
        best_document_label = None
        viterbi_forward_pass_results_per_document_label = \
            {document_label: {'bp': None, 'pi': None} for document_label in DOCUMENT_LABELS}
        bp_for_multi_document_labels = {}
        if infer_document_label:
            for document_label in DOCUMENT_LABELS:
                cur_bp, cur_pi = self.viterbi_forward_pass(document, document_label=document_label, top_k=top_k)
                viterbi_forward_pass_results_per_document_label[document_label]['bp'] = cur_bp
                viterbi_forward_pass_results_per_document_label[document_label]['pi'] = cur_pi
                if (bp is None and pi is None) or (cur_pi[-1].max() > pi[-1].max()):
                    bp, pi = cur_bp, cur_pi
                    best_document_label = document_label

            # Now we have the viterbi results (pi+bp) for each document label independently.
            # We want to find the k labelings with the highest scores.
            # The problem is we now have the best-k for each document label.
            # The final best-k labelings may contain labelings with different document labels.
            # Hence, we have to merge the two viterbi results into a single top-k array.
            # For each one of these final best-k labelings, the back-pointer (for that merge step)
            #   stores the document label of that labeling. It indicates in which result continue
            #   to track back in order to find the rest of that labeling (out of the 2 results).

            # Create a matrix of size (NR_DOCUMENT_LABELS * top_k), such that the cell M[document_lbl_idx, rank_idx]
            # contains the total score of the (rank_idx+1)-th best result from the viterbi execution with document_lbl_idx.
            total_scores_per_document_label_and_rank = np.zeros((NR_DOCUMENT_LABELS, top_k))
            for document_label_idx, document_label in enumerate(DOCUMENT_LABELS):
                total_scores_per_document_label_and_rank[document_label_idx, :] = \
                    viterbi_forward_pass_results_per_document_label[document_label]['pi'][-1, 0, :]

            # Find indeces (and values) of top_k elements with highest score in the above created matrix.
            # Notice: Out of (NR_DOCUMENT_LABELS * top_k) elements we find and sort the best top_k elements
            #   using time complexity (NR_DOCUMENT_LABELS * top_k) + (top_k * log(top_k)), which is optimal.
            bp_for_multi_document_labels['document_label_idxs'], bp_for_multi_document_labels['rank_idxs'], _ = \
                get_sorted_highest_k_elements_in_matrix(total_scores_per_document_label_and_rank, top_k)

        else:
            # Here we do not have to infer the document label.
            # `document.label` will be used as a fixed document label.
            bp, pi = self.viterbi_forward_pass(document, document_label=None, top_k=top_k)

        # Backward pass: Assign the top_k sentence labelings (with highest scores) using calculated scores
        # array `pi` and back-pointers array `bp`.
        all_labelings = []

        # If we have to infer document label, infer top_k labelings for each possible document label;
        # otherwise, run the inference just once for `document.label`.
        document_labels_to_iterate_over = DOCUMENT_LABELS if infer_document_label and infer_top_k_per_each_document_label else [document.label]
        for cur_document_label in document_labels_to_iterate_over:
            # For each rank, trace back (using the back-pointers) the labeling that yields the k-th highest score.
            for rank in range(top_k):
                rank_to_use_in_bp = rank
                if infer_document_label:
                    if infer_top_k_per_each_document_label:
                        document_label = cur_document_label
                    else:
                        # Get the `rank`-th highest labeling out of both viterbi-forward-pass results.
                        rank_to_use_in_bp = bp_for_multi_document_labels['rank_idxs'][rank]
                        document_label = DOCUMENT_LABELS[bp_for_multi_document_labels['document_label_idxs'][rank]]
                    bp = viterbi_forward_pass_results_per_document_label[document_label]['bp']
                else:
                    document_label = document.label

                assign_labeling = (rank == 0 and assign_best_labeling) and \
                                  (not infer_top_k_per_each_document_label or best_document_label == document_label)
                labeling = self.extract_labeling_from_viterbi_forward_pass_results(
                    document, document_label, bp, rank_to_use_in_bp, assign_labeling)
                all_labelings.append(labeling)

        return all_labelings

    def extract_labeling_from_viterbi_forward_pass_results(self, document: Document,
                                                           document_label,
                                                           bp: np.ndarray,
                                                           rank_idx: int = 0,
                                                           assign_labeling_to_document: bool = False):
        assert(rank_idx <= bp.shape[2])

        # The first cell (`1+`) is for the document label.
        labeling = [None] * (1 + document.count_sentences())
        labeling[0] = document_label

        sentence_rank_idx = rank_idx
        following_sentence_label_idx = 0
        for sentence_idx in range(document.count_sentences() - 1, -1, -1):
            # We use the back-pointers of following sentence #(sentence_idx+1) for tracing back the label&rank
            # of the sentence #sentence_idx.
            # Note that it is ok to access the cells at #(last_sentence_idx+1), because we made sure to
            # calculate it during the forward pass. It helps us finding the starting of the tracing of the k-th
            # best labeling.
            following_sentence_idx = sentence_idx + 1
            sentence_label_idx = bp[following_sentence_idx, following_sentence_label_idx, sentence_rank_idx, 0]
            sentence_rank_idx = bp[following_sentence_idx, following_sentence_label_idx, sentence_rank_idx, 1]
            labeling[sentence_idx + 1] = SENTENCE_LABELS[sentence_label_idx]
            following_sentence_label_idx = sentence_label_idx

        # The first sentence is assigned (by the forward pass) with -inf scores for each rank>0 (explained in
        #   the implementation of the forward pass method).
        # Hence any labeling path is expected to track back to rank=0 for the first sentence.
        assert (sentence_rank_idx == 0)

        if assign_labeling_to_document:
            document.assign_labeling(labeling)

        return labeling

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
                        document, sentence, document_label, sentence_label)
                    continue

                if sentence is not None:
                    # Remember that the label of the previous sentence may affect the score of the current sentence.
                    # For each possible label (for the previous sentence), evaluate the score of the current
                    # sentence, when using `sentence_label` as the label for the current sentence.
                    sentence_scores_per_pre_sentence_label = [(pre_sentence_label_idx, self.sentence_score(
                        document, sentence, document_label, sentence_label, pre_sentence_label))
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
                top_k_label_indeces, top_k_ranks, top_k_elements_sorted = \
                    get_sorted_highest_k_elements_in_matrix(total_scores_per_pre_sentence_label_and_rank, top_k)

                # Update the dynamic-programming intermediate results arrays (scores and back-pointer).
                pi[sentence_idx, sentence_label_idx, :] = top_k_elements_sorted
                bp[sentence_idx, sentence_label_idx, :, 0] = top_k_label_indeces
                bp[sentence_idx, sentence_label_idx, :, 1] = top_k_ranks

        return bp, pi

    def document_predict(self, corpus: Corpus):
        for document in corpus.documents:
            best_document_score = None
            for document_label in DOCUMENT_LABELS:
                sum_document_score = 0
                for sentence in document.sentences:
                    sum_document_score = sum_document_score + \
                                         self.sentence_score(document, sentence, document_label=document_label,
                                                             sentence_label=None, pre_sentence_label=None)
                if best_document_score is None or sum_document_score > best_document_score:
                    best_document_score = sum_document_score
                    document.label = document_label

    def sentence_predict(self, document: Document):
        for sentence in document.sentences:
            best_sentence_score = None
            for sentence_label in SENTENCE_LABELS:
                temp_sentence_score = self.sentence_score(document, sentence,
                                                          document_label=None,
                                                          sentence_label=sentence_label,
                                                          pre_sentence_label=None)
                if best_sentence_score is None or temp_sentence_score > best_sentence_score:
                    best_sentence_score = temp_sentence_score
                    sentence.label = sentence_label

    def inference(self, corpus: Corpus):
        assert self.w is not None

        start_time = time.time()

        if self.model_config.model_type == DOCUMENT_CLASSIFIER:
            self.document_predict(corpus)

        elif self.model_config.model_type == SENTENCE_CLASSIFIER:
            for document in corpus.documents:
                self.sentence_predict(document)

        elif self.model_config.model_type in {SENTENCE_STRUCTURED, STRUCTURED_JOINT}:
            pb = ProgressBar(len(corpus.documents))
            for i, document in enumerate(corpus.documents):
                pb.start_next_task()
                self.viterbi_inference(
                    document,
                    infer_document_label=(self.model_config.model_type == STRUCTURED_JOINT),
                    infer_top_k_per_each_document_label=False)
            pb.finish()

        print("Inference done: {0:.3f} seconds".format(time.time() - start_time))

    def evaluate_model(self, inferred_corpus: Corpus, ground_truth_corpus: Corpus):
        assert self.w is not None

        results = {}
        if self.model_config.infer_sentences_labels:
            sentences_results = {
                "correct": 0,
                "errors": 0
            }
            for (_, sentence_inferred), (_, sentence_ground_truth) in zip(inferred_corpus, ground_truth_corpus):
                assert (sentence_inferred.label in SENTENCE_LABELS and sentence_ground_truth.label in SENTENCE_LABELS)
                if sentence_inferred.label == sentence_ground_truth.label:
                    sentences_results["correct"] += 1
                else:
                    sentences_results["errors"] += 1
            sentences_results["accuracy"] = sentences_results["correct"] / sum(sentences_results.values())
            results['sentences'] = sentences_results
        if self.model_config.infer_document_label:
            documents_results = {
                "correct": 0,
                "errors": 0
            }
            for document_inferred, document_ground_truth in zip(inferred_corpus.documents, ground_truth_corpus.documents):
                assert (document_inferred.label in DOCUMENT_LABELS and document_ground_truth.label in DOCUMENT_LABELS)
                if document_inferred.label == document_ground_truth.label:
                    documents_results["correct"] += 1
                else:
                    documents_results["errors"] += 1
            documents_results["accuracy"] = documents_results["correct"] / sum(documents_results.values())
            results["documents"] = documents_results
        return results

    # def print_results_to_file(self, tagged_file, model_name):
    #     path = TEST_PATH
    #
    #     f = open(path + model_name, 'w')
    #     for s_truth, s_eval in zip(tagged_file.sentences, self.corpus.sentences):
    #         f.write(" ".join(["{}_{}".format(t_truth.word, t_eval.tag) for t_truth, t_eval
    #                           in zip(s_truth.tokens, s_eval.tokens)]) + "\n")

    def confusion_matrix(self, inferred_corpus: Corpus, ground_truth_corpus: Corpus):
        confusion_matrix = np.zeros((len(SENTENCE_LABELS), len(SENTENCE_LABELS)))
        for d1, d2 in zip(inferred_corpus.documents, ground_truth_corpus.documents):
            for s1, s2 in zip(d1.sentences, d2.sentences):
                confusion_matrix[SENTENCE_LABELS.index(s1.label)][SENTENCE_LABELS.index(s2.label)] += 1
        np.savetxt(TEST_PATH + "confusion_matrix__{}.txt".format(self.model_config.model_name), confusion_matrix)

        file = open(TEST_PATH + "confusion_matrix_lines__{}.txt".format(self.model_config.model_name), "w")
        for i in range(len(SENTENCE_LABELS)):
            for j in range(len(SENTENCE_LABELS)):
                value = confusion_matrix[i][j]
                file.write("Truth label: {}, Predicted label: {}, number of errors: {}\n".format(SENTENCE_LABELS[j],
                                                                                                 SENTENCE_LABELS[i],
                                                                                                 value))
        file.close()
        return confusion_matrix

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

    def save(self, model_weights_filename: str = None):
        if model_weights_filename is None:
            model_weights_filename = self.model_config.model_weights_filename
        np.savetxt(MODELS_PATH + model_weights_filename, self.w)

    @staticmethod
    def load(model_config: SentimentModelConfiguration, features_extractor: CorpusFeaturesExtractor = None):
        if features_extractor is None:
            # features_extractor = CorpusFeaturesExtractor.load_or_create(model_config, dataset.train)
            pass  # TODO: load it!
        model_weights = np.loadtxt(MODELS_PATH + model_config.model_weights_filename)
        return SentimentModel(features_extractor, model_config, model_weights)
