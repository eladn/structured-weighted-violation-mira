import re

import random
from scipy import sparse

import numpy as np
import time
from itertools import islice

from qpsolvers import solve_qp
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, zero_one_loss
from warnings import warn
import sys

from constants import DEBUG, DATA_PATH, STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH, DOCUMENT_LABELS, SENTENCE_LABELS, MODELS, TEST_PATH

from utils import ProgressBar, print_title


class Corpus:
    def __init__(self):
        self.documents = []

    def clone(self):
        new_corpus = Corpus()
        new_corpus.documents = [doc.clone() for doc in self.documents]
        return new_corpus

    def load_file(self, file_name, documents_label: int, insert_sentence_labels: bool):
        assert documents_label in DOCUMENT_LABELS
        pattern = re.compile("^\d \d{7}$")
        with open(DATA_PATH + file_name) as f:
            document = Document(documents_label)
            for i, line in enumerate(f):
                if line == "\n":
                    self.documents.append(document)
                    document = Document(documents_label)
                elif pattern.match(line):
                    continue
                else:
                    document.load_sentence(line, insert_sentence_labels)

    def count_documents(self):
        return np.size(self.documents)

    def count_sentences(self):
        return sum([doc.count_sentences() for doc in self.documents])

    def count_tokens(self):
        return sum([doc.count_tokens() for doc in self.documents])

    def __str__(self):
        return "\n".join([str(document) for document in self.documents])


class Document:
    def __init__(self, label=None):
        assert label is None or label in DOCUMENT_LABELS
        self.sentences = []
        self.label = label

    def clone(self):
        new_doc = Document(self.label)
        new_doc.sentences = [sentence.clone() for sentence in self.sentences]
        return new_doc

    def load_sentence(self, line, insert_sec_labels):
        self.sentences.append(Sentence(line, insert_sec_labels))

    def count_sentences(self):
        return len(self.sentences)

    def count_tokens(self):
        return sum([sen.count_tokens() for sen in self.sentences])

    def y(self):
        return [self.label] + [s.label for s in self.sentences]

    def __str__(self):
        return "\n".join([str(sentence) for sentence in self.sentences])


class Sentence:
    def __init__(self, sentence: str, insert_sec_labels: bool):
        self.tokens = []
        splitted_sec = sentence.split("\t")
        if insert_sec_labels:
            self.label = int(splitted_sec[0])
        else:
            self.label = None
        words = "\t".join(splitted_sec[1:])
        self.tokens = [TaggedWord(word_tag=token) for token in Sentence.split_cleaned_line(words)]

    def clone(self):
        new_sentence = Sentence('-1\ttv_NN', False)
        new_sentence.label = self.label
        new_sentence.tokens = [tagged_word.clone() for tagged_word in self.tokens]
        return new_sentence

    @staticmethod
    def split_cleaned_line(line):  # TODO check if fix with data?
        return line.strip("\n ").split(" ")

    def count_tokens(self):
        return len(self.tokens)

    def xgram(self, index, x):
        beginning = ["*" for _ in range(max(x - index - 1, 0))]
        end = [token.tag for token in self.tokens[max(index - x + 1, 0):index + 1]]
        return tuple(beginning + end)

    def trigram(self, index):
        return self.xgram(index, 2)  # only two tags before current

    def bigram(self, index):
        return self.xgram(index, 1)

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])


class TaggedWord:
    def __init__(self, word=None, tag=None, word_tag=None):
        """
        This class supports:
        - word_tag (for example: word_tag = "the_DT")
        - word, tag (for example: word = "the", tag = "DT")
        - word (for example: word = "the") - untagged
        """

        if word_tag is not None:
            word, tag = TaggedWord.split_word_tag(word_tag)

        self.word = word
        self.tag = tag

    def clone(self):
        return TaggedWord(self.word, self.tag)

    @staticmethod
    def split_word_tag(word_tag):
        return word_tag.split("_")  # TODO check if fix with the data?

    @staticmethod
    def split_word_from_word_tag(word_tag):
        return TaggedWord.split_word_tag(word_tag)[0]

    @staticmethod
    def split_tag_from_word_tag(word_tag):
        return TaggedWord.split_word_tag(word_tag)[1]

    def split_tag_from_taggedword(self):
        return self.tag

    def is_tag(self):
        return bool(self.tag)

    def print_to_file(self):
        return "{}_{}".format(self.word, self.tag)

    def __eq__(self, tagged_word):
        return self.word == tagged_word.word and self.tag == tagged_word.tag

    def __str__(self):
        return "{word}:{tag}".format(word=self.word, tag=self.tag)


class Train:
    def __init__(self, corpus, feature_vector, model):
        assert model in MODELS
        if DEBUG:
            if not isinstance(corpus, Corpus):
                raise Exception('The corpus argument is not a Corpus object')
        self.corpus = corpus
        self.feature_vector = feature_vector
        self.empirical_counts = None
        self.evaluated_feature_vectors = []
        self.w = None
        self.model = model

    def generate_features(self):
        start_time = time.time()
        self.feature_vector.initialize_features(self.model)
        self.w = np.zeros(self.feature_vector.count_features())
        print("generate_feature done: {0:.3f} seconds".format(time.time() - start_time))
        print("Features count: {}".format(self.feature_vector.count_features()))

    def evaluate_feature_vectors(self):
        start_time = time.time()
        print("Evaluating features")
        for doc_index, document in enumerate(self.corpus.documents, start=1):
            self.evaluated_feature_vectors.append(self.evaluate_document_feature_vector(document))
            print("Document #{} evaluated features".format(doc_index), end='\r')
        print("evaluate_feature_vectors done: {0:.3f} seconds".format(time.time() - start_time))

    def evaluate_document_feature_vector(self, document, y_tag=None):
        # TODO if not structured - should not count from 1
        y_document = y_tag[0] if y_tag is not None else None
        row_ind, col_ind = [], []
        for sentence_index, sentence in enumerate(islice(document.sentences, 1, None), start=1):  # TODO
            y_sentence = y_tag[sentence_index + 1] if y_tag is not None else None
            y_pre_sentence = y_tag[sentence_index] if y_tag is not None else None
            feature_indices = self.feature_vector.evaluate_clique_feature_vector(
                document, sentence_index, self.model, y_document, y_sentence, y_pre_sentence)
            col_ind += feature_indices
            row_ind += [sentence_index - 1 for _ in feature_indices]

        return csr_matrix(([1 for _ in col_ind], (row_ind, col_ind)),
                          shape=(document.count_sentences() - 1, self.feature_vector.count_features()))  # TODO

    def loss(self, w, w_i):
        return np.sum((w - w_i) ** 2)

    def gradient(self, w, w_i):
        return 2 * np.sum(w - w_i)

    def mira_algorithm(self, iterations=5, k=10, best_k_labeling_method='rand'):
        assert best_k_labeling_method in {'rand', 'viterbi', 'rand-and-viterbi'}

        optimization_time = time.time()
        print_title("Training model: {}, k = {}, iterations = {}".format(self.model, k, iterations))
        pb = ProgressBar(iterations * len(self.corpus.documents))
        test = Test(self.corpus.clone(), self.feature_vector, self.model)
        for cur_iter in range(1, iterations+1):
            for doc_idx, (document, test_document, feature_vectors) in enumerate(zip(self.corpus.documents, test.corpus.documents, self.evaluated_feature_vectors)):
                task_str = 'iter: {cur_iter}/{nr_iters} -- document: {cur_doc}/{nr_docs}'.format(
                    cur_iter=cur_iter, nr_iters=iterations, cur_doc=doc_idx+1, nr_docs=len(self.corpus.documents)
                )
                pb.start_next_task(task_str)

                if best_k_labeling_method == 'rand':
                    c = self.extract_random_labeling_subset(document, k, use_document_tag=False)
                else:  # best_k_labeling_method == 'viterbi'
                    test.w = self.w
                    # TODO: impl & use k-best viterbi
                    test.viterbi_on_sentences(test_document, doc_idx, test.model, verbose=False)
                    labels = [document.label] + [sentence.label for sentence in test_document.sentences]
                    c = [labels]
                # TODO: handle case best_k_labeling_method == 'rand-and-viterbi' (0.5k, 0.5k)

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

    def extract_qp_matrices(self, document, feature_vectors, y, c):
        M = sparse.eye(self.feature_vector.count_features())
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
            if self.model in {DOCUMENT_CLASSIFIER, STRUCTURED_JOINT}:
                y_tag_loss += y_tag_document_loss
            h.append(-y_tag_loss)
        G = csr_matrix(G)
        h = np.array(h).reshape(len(c), )
        return M, q.reshape((self.feature_vector.count_features()), ), G, h

    @staticmethod
    def extract_random_labeling_subset(document, k, use_document_tag: bool=False):  # TODO use viterbi to find the best c
        return [
            [document.label if use_document_tag else random.choice(DOCUMENT_LABELS)] +
            [random.choice(SENTENCE_LABELS) for _ in range(document.count_sentences())]
            for _ in range(k)]

    def save_model(self, model_name: str):
        np.savetxt(MODELS_PATH + model_name, self.w)

    def load_model(self, model_name: str):
        self.w = np.loadtxt(MODELS_PATH + model_name)


class Test:
    def __init__(self, corpus, feature_vector, model, w=None):
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

    def document_predict(self, corpus, model):
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

    def inference(self, verbose: bool=True):
        start_time = time.time()

        if self.model == DOCUMENT_CLASSIFIER:
            self.document_predict(self.corpus, self.model)

        elif self.model == SENTENCE_CLASSIFIER:
            for document in self.corpus.documents:
                self.sentence_predict(document, self.model)

        elif self.model == SENTENCE_STRUCTURED:
            for i, document in enumerate(self.corpus.documents):
                self.viterbi_on_sentences(document, i, self.model, verbose=verbose)

        elif self.model == STRUCTURED_JOINT:
            for i, document in enumerate(self.corpus.documents):
                self.viterbi_on_document(document, i, self.model)

        if verbose:
            print("inference done: {0:.3f} seconds".format(time.time() - start_time))

    def load_model(self, model_name):
        path = MODELS_PATH
        self.w = np.loadtxt(path + model_name)

    def evaluate_model(self, ground_truth, model):
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

    def confusion_matrix(self, ground_truth, model_name):
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


class FeatureVector:
    def __init__(self, corpus):
        self.corpus = corpus
        self.document = {-1: {}, 1: {}}
        self.sentence = {}
        self.sentence_document = {}
        self.pre_sentence_sentence = {}
        self.pre_sentence_sentence_document = {}
        # for i in range(1, 4):
        #     self.document[1][i] = {}
        #     self.document[-1][i] = {}
        for i in range(1, 15):
            self.document[1][i] = {}
            self.document[-1][i] = {}
        for i in [-1, 1]:
            self.sentence[i] = {}
            # for j in range(1, 4):
            #     self.sentence[i][j] = {}
            for j in range(1, 15):
                self.sentence[i][j] = {}
        for i in [-1, 1]:
            for j in [-1, 1]:
                self.sentence_document[(i, j)] = {}
                # for k in range(1, 4):
                #     self.sentence_document[(i, j)][k] = {}
                for k in range(1, 15):
                    self.sentence_document[(i, j)][k] = {}
        for i in [-1, 1]:
            for j in [-1, 1]:
                self.pre_sentence_sentence[(i, j)] = {}
                # for k in range(1, 4):
                #     self.pre_sentence_sentence[(i, j)][k] = {}
                for k in range(1, 15):
                    self.pre_sentence_sentence[(i, j)][k] = {}
        for i in [-1, 1]:
            for j in [-1, 1]:
                for d in [-1, 1]:
                    self.pre_sentence_sentence_document[(i, j, d)] = {}
                    # for k in range(1, 4):
                    #     self.pre_sentence_sentence_document[(i, j, d)][k] = {}
                    for k in range(1, 15):
                        self.pre_sentence_sentence_document[(i, j, d)][k] = {}

        self.index = 0

    def increment_index(self):
        self.index += 1

    def pre_tags(self, sentence, index):
        pre_pre_tag = sentence[index - 2].tag if index >= 2 else "*"
        pre_tag = sentence[index - 1].tag if index >= 1 else "*"
        return pre_pre_tag, pre_tag

    def pre_word(self, sentence, index):
        return sentence[index - 1].word if index >= 1 else "*"

    def pre_pre_word(self, sentence, index):
        return sentence[index - 2].word if index >= 2 else "*"

    def next_word(self, sentence, index):
        return sentence[index + 1].word if index != len(sentence) - 1 else "*"

    def f_1_word_tag(self, sentence, index):
        return sentence[index].word, sentence[index].tag

    def f_2_tag(self, sentence, index):
        return sentence[index].tag

    def f_3_bigram(self, sentence, index):
        _, pre_tag = self.pre_tags(sentence, index)
        pre_word = self.pre_word(sentence, index)
        return pre_word, pre_tag, sentence[index].word, sentence[index].tag

    def f_4_bigram_none(self, sentence, index):
        _, pre_tag = self.pre_tags(sentence, index)
        return pre_tag, sentence[index].word, sentence[index].tag

    def f_5_bigram_none(self, sentence, index):
        _, pre_tag = self.pre_tags(sentence, index)
        pre_word = self.pre_word(sentence, index)
        return pre_word, pre_tag, sentence[index].tag

    def f_6_bigram_none_none(self, sentence, index):
        _, pre_tag = self.pre_tags(sentence, index)
        return pre_tag, sentence[index].tag

    def f_7_trigram(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        pre_word = self.pre_word(sentence, index)
        pre_pre_word = self.pre_pre_word(sentence, index)
        return pre_pre_word, pre_pre_tag, pre_word, pre_tag, sentence[index].word, sentence[index].tag

    def f_8_trigram_pre_pre_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        pre_word = self.pre_word(sentence, index)
        return pre_pre_tag, pre_word, pre_tag, sentence[index].word, sentence[index].tag

    def f_9_trigram_pre_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        pre_pre_word = self.pre_pre_word(sentence, index)
        return pre_pre_word, pre_pre_tag, pre_tag, sentence[index].word, sentence[index].tag

    def f_10_trigram_pre_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        pre_word = self.pre_word(sentence, index)
        pre_pre_word = self.pre_pre_word(sentence, index)
        return pre_pre_word, pre_pre_tag, pre_word, pre_tag, sentence[index].tag

    def f_11_trigram_pre_pre_none_pre_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        return pre_pre_tag, pre_tag, sentence[index].word, sentence[index].tag

    def f_12_trigram_pre_pre_none_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        pre_word = self.pre_word(sentence, index)
        return pre_pre_tag, pre_word, pre_tag, sentence[index].word, sentence[index].tag

    def f_13_trigram_pre_none_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        pre_pre_word = self.pre_pre_word(sentence, index)
        return pre_pre_word, pre_pre_tag, pre_tag, sentence[index].tag

    def f_14_trigram_none_none_none(self, sentence, index):
        pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
        return pre_pre_tag, pre_tag, sentence[index].tag

    def initialize_feature_based_on_label(self, sentence, index, pre_sentence_label=None, sentence_label=None,
                                          document_label=None):
        feature_types = ["f_1_word_tag", "f_2_tag", "f_3_bigram", "f_4_bigram_none", "f_5_bigram_none",
                         "f_6_bigram_none_none",
                         "f_7_trigram", "f_8_trigram_pre_pre_none", "f_9_trigram_pre_none", "f_10_trigram_pre_none",
                         "f_11_trigram_pre_pre_none_pre_none", "f_12_trigram_pre_pre_none_none",
                         "f_13_trigram_pre_none_none",
                         "f_14_trigram_none_none_none"]
        if pre_sentence_label and sentence_label and document_label:
            for feature_index, feature_type in enumerate(feature_types, start=1):
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                if predicate in self.pre_sentence_sentence_document[(pre_sentence_label, sentence_label, document_label)][feature_index]:
                    break
                self.pre_sentence_sentence_document[(pre_sentence_label, sentence_label, document_label)][
                    feature_index][predicate] = self.index
                self.increment_index()

        if not pre_sentence_label and not sentence_label and document_label:
            for feature_index, feature_type in enumerate(feature_types, start=1):
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                if predicate in self.document[document_label][feature_index]:
                    break
                self.document[document_label][feature_index][predicate] = self.index
                self.increment_index()

        if not pre_sentence_label and sentence_label and not document_label:
            for feature_index, feature_type in enumerate(feature_types, start=1):
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                if predicate in self.sentence[sentence_label][feature_index]:
                    break
                self.sentence[sentence_label][feature_index][predicate] = self.index
                self.increment_index()

        if not pre_sentence_label and sentence_label and document_label:
            for feature_index, feature_type in enumerate(feature_types, start=1):
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                if predicate in self.sentence_document[(sentence_label, document_label)][feature_index]:
                    break
                self.sentence_document[(sentence_label, document_label)][feature_index][predicate] = self.index
                self.increment_index()

        if pre_sentence_label and sentence_label and not document_label:
            for feature_index, feature_type in enumerate(feature_types, start=1):
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                if predicate in self.pre_sentence_sentence[(pre_sentence_label, sentence_label)][feature_index]:
                    break
                self.pre_sentence_sentence[(pre_sentence_label, sentence_label)][feature_index][
                    predicate] = self.index
                self.increment_index()

    def initialize_features(self, model):
        print("Initializing features")
        for doc_index, document in enumerate(self.corpus.documents, start=1):
            for sen_index, sentence in enumerate(document.sentences):
                for index, token in enumerate(sentence.tokens):
                    if sen_index >= 1:
                        if model == STRUCTURED_JOINT:
                            self.initialize_feature_based_on_label(
                                sentence, index, document.sentences[sen_index - 1].label, sentence.label,
                                document.label)

                        if model in (STRUCTURED_JOINT, SENTENCE_STRUCTURED):
                            self.initialize_feature_based_on_label(
                                sentence, index, pre_sentence_label=document.sentences[sen_index - 1].label,
                                sentence_label=sentence.label)
                    if model in (STRUCTURED_JOINT, DOCUMENT_CLASSIFIER):
                        self.initialize_feature_based_on_label(sentence, index, document_label=document.label)

                    if model in (STRUCTURED_JOINT, SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED):
                        self.initialize_feature_based_on_label(sentence, index, sentence_label=sentence.label)

                    if model == STRUCTURED_JOINT:
                        self.initialize_feature_based_on_label(sentence, index, sentence_label=sentence.label,
                                                               document_label=document.label)
            print("Document #{} initialized features".format(doc_index), end='\r')
        print("Finished initializing features.")

    def evaluate_clique_feature_vector(self, document: Document, sen_index: int, model,
                                       document_label=None, sentence_label=None,
                                       pre_sentence_label=None):
        assert model in MODELS
        assert document_label is None or document_label in DOCUMENT_LABELS
        assert sentence_label is None or sentence_label in SENTENCE_LABELS
        assert pre_sentence_label is None or pre_sentence_label in SENTENCE_LABELS

        sentence = document.sentences[sen_index]
        document_label = document.label if document_label is None else document_label
        sentence_label = sentence.label if sentence_label is None else sentence_label

        if sen_index == 0:
            pre_sentence_label = None
        else:
            pre_sentence_label = document.sentences[
                sen_index - 1].label if pre_sentence_label is None else pre_sentence_label

        evaluated_feature = []
        for index, token in enumerate(sentence.tokens):
            pre_word = self.pre_word(sentence.tokens, index)
            pre_pre_word = self.pre_pre_word(sentence.tokens, index)
            pre_pre_tag, pre_tag = self.pre_tags(sentence.tokens, index)
            tag = token.tag
            # feature_arguments = [tag, (pre_tag, tag), (pre_pre_tag, pre_tag, tag)]
            #
            feature_arguments = [(token.word, tag), tag, (pre_word, pre_tag, token.word, tag),
                                 (pre_tag, token.word, tag), (pre_word, pre_tag, tag), (pre_tag, tag),
                                 (pre_pre_word, pre_pre_tag, pre_word, pre_tag, token.word, tag),
                                 (pre_pre_tag, pre_word, pre_tag, token.word, tag),
                                 (pre_pre_word, pre_pre_tag, pre_tag, token.word, tag),
                                 (pre_pre_word, pre_pre_tag, pre_word, pre_tag, tag),
                                 (pre_pre_tag, pre_tag, token.word, tag),
                                 (pre_pre_tag, pre_word, pre_tag, token.word, tag),
                                 (pre_pre_word, pre_pre_tag, pre_tag, tag), (pre_pre_tag, pre_tag, tag)]

            if model in (STRUCTURED_JOINT, DOCUMENT_CLASSIFIER):
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.document[document_label][f_index + 1].get(arguments)
                    if value is not None:
                        evaluated_feature.append(value)

            if model in (STRUCTURED_JOINT, SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED):
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.sentence[sentence_label][f_index + 1].get(arguments)
                    if value is not None:
                        evaluated_feature.append(value)

            if model in (STRUCTURED_JOINT, SENTENCE_STRUCTURED):
                for f_index, arguments in enumerate(feature_arguments):
                    if pre_sentence_label:
                        value = self.pre_sentence_sentence[(pre_sentence_label, sentence_label)][f_index + 1].get(arguments)
                        if value is not None:
                            evaluated_feature.append(value)

            if model == STRUCTURED_JOINT:
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.sentence_document[(sentence_label, document_label)][f_index + 1].get(arguments)
                    if value is not None:
                        evaluated_feature.append(value)
                for f_index, arguments in enumerate(feature_arguments):
                    if pre_sentence_label:
                        value = self.pre_sentence_sentence_document[(pre_sentence_label, sentence_label, document_label)][
                            f_index + 1].get(arguments)
                        if value is not None:
                            evaluated_feature.append(value)

        return evaluated_feature

    def extended_feature_vector(self, evaluated_feature_vector):
        vector = np.zeros(self.count_features())
        if evaluated_feature_vector is not None and evaluated_feature_vector.size > 0:
            vector.put(evaluated_feature_vector, 1)
        return vector

    # def print_num_of_features(self):
    #     for feature_type, features in self.features.items():
    #         if feature_type in [101, 102]:
    #             count_features = 0
    #             for num_chars, _features in features.items():
    #                 print("f_{} {}: {}".format(feature_type, num_chars, len(_features)))
    #                 count_features += len(_features)
    #             print("total f_{}: {}".format(feature_type, count_features))
    #         else:
    #             print("f_{}: {}".format(feature_type, len(features)))

    def count_features(self):
        return self.index
