
import numpy as np
import time

from constants import DEBUG, DATA_PATH, STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH, DOCUMENT_LABELS, SENTENCE_LABELS, MODELS, TEST_PATH
from corpus import Corpus
from document import Document
from sentence import Sentence
from feature_vector import FeatureVector
from utils import ProgressBar, print_title


class SentimentModelTester:
    def __init__(self, corpus: Corpus, feature_vector: FeatureVector, model, w=None):
        assert model in MODELS
        if DEBUG:
            if not isinstance(corpus, Corpus):
                raise Exception('The corpus argument is not a Corpus object')
        self.matrix = np.zeros((len(SENTENCE_LABELS), len(SENTENCE_LABELS)))
        self.corpus = corpus
        self.feature_vector = feature_vector
        self.w = w
        self.model = model

    def sentence_score(self, document: Document, sen_index: int, model, document_label, sentence_label, pre_sentence_label):
        assert model in MODELS
        assert sentence_label in SENTENCE_LABELS
        assert document_label is None or document_label in DOCUMENT_LABELS
        assert pre_sentence_label is None or pre_sentence_label in SENTENCE_LABELS
        fv = self.feature_vector.evaluate_clique_feature_vector(document, sen_index, model,
                                                                document_label=document_label,
                                                                sentence_label=sentence_label,
                                                                pre_sentence_label=pre_sentence_label)
        return np.sum(np.take(self.w, fv))

    def viterbi_on_document(self, document: Document, document_index: int, model, verbose: bool=True):
        assert model in MODELS

        start_time = time.time()

        n = document.count_sentences()
        bp_best, pi_best = None, None

        for document_label in DOCUMENT_LABELS:
            bp, pi = self.viterbi_on_document_label(document, document_label, model)
            if (bp_best is None and pi_best is None) or (pi[n - 1].max() > pi_best[n - 1].max()):
                bp_best, pi_best = bp, pi
                document.label = document_label

        sentence_label = np.where(pi_best[n - 1] == pi_best[n - 1].max())
        document.sentences[n - 1].label = SENTENCE_LABELS[sentence_label[0][0]]

        for k in range(n - 2, -1, -1):
            document.sentences[k].label = int(bp_best[k + 1, SENTENCE_LABELS.index(document.sentences[k + 1].label)])

        if verbose:
            print("Viterbi on document #{doc_num} done. {time:.3f} seconds.".format(
                doc_num=document_index, time=(time.time() - start_time)), end='\r')
        return document, document_index

    def viterbi_on_sentences(self, document: Document, document_index: int, model, verbose: bool=True):
        assert model in MODELS

        start_time = time.time()

        n = document.count_sentences()
        bp, pi = self.viterbi_on_document_label(document, document_label=None, model=model)

        last_sentence_label_index = np.where(pi[n - 1] == pi[n - 1].max())
        document.sentences[n - 1].label = SENTENCE_LABELS[last_sentence_label_index[0][0]]

        for k in range(n - 2, -1, -1):
            document.sentences[k].label = int(bp[k + 1, SENTENCE_LABELS.index(document.sentences[k + 1].label)])

        if verbose:
            print("Viterbi on sentences over document #{doc_num} done. {time:.3f} seconds.".format(
                doc_num=document_index, time=(time.time() - start_time)), end='\r')
        return document, document_index

    def viterbi_on_document_label(self, document: Document, document_label, model):
        assert model in MODELS
        assert document_label is None or document_label in DOCUMENT_LABELS

        n = document.count_sentences()
        count_labels = len(SENTENCE_LABELS)
        pi = np.zeros((n, count_labels))
        bp = np.zeros((n, count_labels))
        for k, sentence in enumerate(document.sentences):
            #print("Sentence: {}".format(k))
            for v_index, v in enumerate(SENTENCE_LABELS):
                if k == 0:
                    pi[k, v_index] = self.sentence_score(document, k, model, document_label, v, pre_sentence_label=None)
                else:
                    pi_q = [self.sentence_score(document, k, model, document_label, v, t) + pi[k - 1, t_index]
                            for t_index, t in enumerate(SENTENCE_LABELS)]
                    pi[k, v_index] = np.amax(pi_q)
                    bp[k, v_index] = SENTENCE_LABELS[np.argmax(pi_q)]
        return bp, pi

    def document_predict(self, corpus: Corpus, model):
        best_document_score = None
        for document in corpus.documents:
            for document_label in DOCUMENT_LABELS:
                sum_document_score = 0
                for i, sentence in enumerate(document.sentences):
                    sum_document_score = sum_document_score + \
                                         self.sentence_score(document, i, model, document_label=document_label,
                                                             sentence_label=None, pre_sentence_label=None)
                if best_document_score is None or sum_document_score > best_document_score:
                    best_document_score = sum_document_score
                    document.label = document_label

    def sentence_predict(self, document: Document, model):
        for i, sentence in enumerate(document.sentences):
            best_sentence_score = None
            for sentence_label in SENTENCE_LABELS:
                temp_sentence_score = self.sentence_score(document, i, model, document_label=None, sentence_label=sentence_label, pre_sentence_label=None)
                if best_sentence_score is None or temp_sentence_score > best_sentence_score:
                    best_sentence_score = temp_sentence_score
                    sentence.label = sentence_label

    def inference(self):
        start_time = time.time()

        if self.model == DOCUMENT_CLASSIFIER:
            self.document_predict(self.corpus, self.model)

        elif self.model == SENTENCE_CLASSIFIER:
            for document in self.corpus.documents:
                self.sentence_predict(document, self.model)

        elif self.model == SENTENCE_STRUCTURED:
            pb = ProgressBar(len(self.corpus.documents))
            for i, document in enumerate(self.corpus.documents):
                pb.start_next_task()
                self.viterbi_on_sentences(document, i, self.model)
            pb.finish()

        elif self.model == STRUCTURED_JOINT:
            pb = ProgressBar(len(self.corpus.documents))
            for i, document in enumerate(self.corpus.documents):
                pb.start_next_task()
                self.viterbi_on_document(document, i, self.model)
            pb.finish()

        print("inference done: {0:.3f} seconds".format(time.time() - start_time))

    def load_model(self, model_name):
        path = MODELS_PATH
        self.w = np.loadtxt(path + model_name)

    def evaluate_model(self, ground_truth: Corpus, model):
        if model == SENTENCE_CLASSIFIER:
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

