import re
from multiprocessing.pool import Pool

import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b

# from constants import TAGS, DEBUG, DATA_PATH, TEST_TAGS, PROCESSES, BASIC_MODELS_PATH, ADVANCED_MODELS_PATH, \
#     BASIC_COMP_PATH, BASIC_TEST_PATH, ADVANCED_TEST_PATH, ADVANCED_COMP_PATH

from constants import TAGS, DEBUG, DATA_PATH, STRUCTURED_JOINT, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, \
    SENTENCE_STRUCTURED, MODELS_PATH


class Corpus:
    def __init__(self):
        self.documents = []

    def load_file(self, file_name, insert_document_labels, insert_sentence_labels):
        pattern = re.compile("^\d \d{7}$")
        with open(DATA_PATH + file_name) as f:
            document = Document(insert_document_labels)
            for i, line in enumerate(f):
                if line == "\n":
                    self.documents.append(document)
                    document = Document(insert_document_labels)  # TODO check if insert_doc_labels need to be here?
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
    def __init__(self, insert_document_labels):
        self.sentences = []
        if insert_document_labels:
            self.label = "None"  # TODO check how to do this with specific data

    def load_sentence(self, line, insert_sec_labels):
        self.sentences.append(Sentence(line, insert_sec_labels))

    def count_sentences(self):
        return np.size(self.sentences)

    def count_tokens(self):
        return sum([sen.count_tokens() for sen in self.sentences])

    def __str__(self):
        return "\n".join([str(sentence) for sentence in self.sentences])


class Sentence:
    def __init__(self, sentence, insert_sec_labels):
        self.tokens = []
        splitted_sec = sentence.split("\t")
        if insert_sec_labels:
            self.label = splitted_sec[0]
        else:
            self.label = "None"
        words = "\t".join(splitted_sec[1:])
        self.tokens = [TaggedWord(word_tag=token) for token in Sentence.split_cleaned_line(words)]

    @staticmethod
    def split_cleaned_line(line):  # TODO check if fix with data?
        return line.strip("\n ").split(" ")

    def count_tokens(self):
        return self.tokens.size

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

        if DEBUG:
            if not word:
                raise Exception('Invalid arguments')
            if not isinstance(word, str):
                raise Exception('The word argument is not a string')
            if tag and tag not in TAGS:
                raise Exception('The tag argument is not in the tags list')

        self.word = word
        self.tag = tag

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
        if DEBUG:
            if not isinstance(corpus, Corpus):
                raise Exception('The corpus argument is not a Corpus object')
        self.corpus = corpus
        self.feature_vector = feature_vector
        self.empirical_counts = None
        self.evaluated_feature_vectors = {}
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
        # for sentence in self.corpus.sentences:
        #     for index, token in enumerate(sentence.tokens):
        #         for tag in TAGS:
        #             pre_pre_tag, pre_tag = self.feature_vector.pre_tags(sentence.tokens, index)
        #             if self.is_basic:
        #                 feature_vector = self.feature_vector.evaluate_basic_feature_vector(sentence.tokens, index,
        #                                                                                    tag=tag)
        #                 history = (pre_pre_tag, pre_tag, token.word)
        #             else:
        #                 pre_word = self.feature_vector.pre_word(sentence.tokens, index)
        #                 next_word = self.feature_vector.next_word(sentence.tokens, index)
        #                 feature_vector = self.feature_vector.evaluate_advanced_feature_vector(sentence.tokens, index,
        #                                                                                       tag=tag)
        #                 history = (pre_pre_tag, pre_tag, token.word, pre_word, next_word)
        #
        #             if history not in self.evaluated_feature_vectors:
        #                 self.evaluated_feature_vectors[history] = {}
        #             self.evaluated_feature_vectors[history][tag] = feature_vector if feature_vector.size > 0 else None
        print("evaluate_feature_vectors done: {0:.3f} seconds".format(time.time() - start_time))

    def train_model(self, lambda_param):
        optimization_time = time.time()
        print("Training model with lambda = {}, model = {}".format(lambda_param, self.model))
        print("------------------------------------")
        # TODO train model and update self.w
        print("Total execution time: {0:.3f} seconds".format(time.time() - optimization_time))
        return None  # TODO

    def save_model(self, model_name):
        np.savetxt(MODELS_PATH + model_name, self.w)

    def load_model(self, model_name):
        self.w = np.loadtxt(MODELS_PATH + model_name)


class Test:
    def __init__(self, corpus, feature_vector, is_basic, v=None):
        if DEBUG:
            if not isinstance(corpus, Corpus):
                raise Exception('The corpus argument is not a Corpus object')
        self.corpus = corpus
        self.feature_vector = feature_vector
        self.v = v
        self.is_basic = is_basic
        self.evaluated_exp_v_f = {}
        self.matrix = np.zeros((len(TAGS), len(TAGS)))

    def evaluate_exp_v_f_on_sentence(self, sentences, b_index):
        evaluated_exp_v_f = {}
        for s_index, sentence in enumerate(sentences):
            for index, token in enumerate(sentence.tokens):
                if self.is_basic:
                    history = token.word
                else:
                    pre_word = self.feature_vector.pre_word(sentence.tokens, index)
                    next_word = self.feature_vector.next_word(sentence.tokens, index)
                    history = (token.word, pre_word, next_word)

                count_test_tags = len(TEST_TAGS)
                evaluated_exp_v_f[history] = np.ones((count_test_tags, count_test_tags, count_test_tags))

                for pre_pre_index, pre_pre_tag in enumerate(TEST_TAGS):
                    for pre_index, pre_tag in enumerate(TEST_TAGS):
                        for t_index, tag in enumerate(TEST_TAGS):
                            evaluated_exp_v_f[history][pre_pre_index][pre_index][t_index] = \
                                self.exp_v_f(sentence, index, pre_pre_tag, pre_tag, tag)
            print("test evaluate_exp_v_f_on_sentence {} (batch {}) done".format(s_index, b_index))

        print("test evaluate_exp_v_f_on_sentence batch {} done".format(b_index))
        return evaluated_exp_v_f

    def evaluate_exp_v_f(self):
        start_time = time.time()
        splits = PROCESSES * 3
        with Pool(processes=PROCESSES) as pool:
            results = pool.starmap(self.evaluate_exp_v_f_on_sentence,
                                   [(s, i) for i, s in enumerate(np.array_split(self.corpus.sentences, splits))])
        for r in results:
            for history, exp_v_f in r.items():
                if history not in self.evaluated_exp_v_f:
                    self.evaluated_exp_v_f[history] = exp_v_f

        print("evaluate_exp_v_f done: {0:.3f} seconds".format(time.time() - start_time))

    # def predefined_exp_v_f(self, sentence, index, pre_pre_tag, pre_tag, tag):
    #     token = sentence.tokens[index]
    #     pre_pre_index = TEST_TAGS.index(pre_pre_tag)
    #     pre_index = TEST_TAGS.index(pre_tag)
    #     t_index = TEST_TAGS.index(tag)
    #     if self.is_basic:
    #         return self.evaluated_exp_v_f[token.word][pre_pre_index][pre_index][t_index]
    #     pre_word = self.feature_vector.pre_word(sentence.tokens, index)
    #     next_word = self.feature_vector.next_word(sentence.tokens, index)
    #     return self.evaluated_exp_v_f[(token.word, pre_word, next_word)][pre_pre_index][pre_index][t_index]

    def v_f(self, sentence, index, pre_pre_tag=None, pre_tag=None, tag=None):
        if self.is_basic:
            f = self.feature_vector.evaluate_basic_feature_vector(sentence.tokens, index,
                                                                  pre_pre_tag=pre_pre_tag, pre_tag=pre_tag, tag=tag)
        else:
            f = self.feature_vector.evaluate_advanced_feature_vector(sentence.tokens, index,
                                                                     pre_pre_tag=pre_pre_tag, pre_tag=pre_tag, tag=tag)
        return np.take(self.v, f).sum() if f.size > 0 else 0

    def exp_v_f(self, sentence, index, pre_pre_tag=None, pre_tag=None, tag=None):
        return np.exp(self.v_f(sentence, index, pre_pre_tag=pre_pre_tag, pre_tag=pre_tag, tag=tag))

    def sum_exp_v_f(self, sentence, index, pre_pre_tag=None, pre_tag=None):
        return np.array(
            [self.exp_v_f(sentence, index, pre_pre_tag=pre_pre_tag, pre_tag=pre_tag, tag=tag) for tag in TAGS]).sum()

    def q(self, sentence, index, pre_pre_tag=None, pre_tag=None, tag=None):
        return self.exp_v_f(sentence, index, pre_pre_tag, pre_tag, tag) / \
               self.sum_exp_v_f(sentence, index, pre_pre_tag, pre_tag)

    def viterbi_on_sentence(self, sentence, s_index):
        start_time = time.time()
        n = sentence.count_tokens()
        count_tags = len(TAGS)
        pi = np.zeros((n, count_tags, count_tags))
        bp = np.ndarray(shape=(n, count_tags, count_tags), dtype=object)
        bp[:] = None
        pi[0, 0, :] = np.array([self.q(sentence, 0, "*", "*", tag) for tag in TAGS])
        bp[0, 0, :] = "*"
        for k, token in enumerate(sentence.tokens[1:], 1):
            print("Token: {}".format(k))
            for u_index, u in enumerate(TAGS):
                tags = TAGS if k != 1 else ["*"]
                denominators = np.array([self.sum_exp_v_f(sentence, k, t, u) for t in tags])
                for v_index, v in enumerate(TAGS):
                    pi_q = [self.exp_v_f(sentence, k, t, u, v) / denominators[t_index] * pi[
                        k - 1, t_index, u_index] for t_index, t in enumerate(tags)]
                    pi[k, u_index, v_index] = np.amax(pi_q)
                    bp[k, u_index, v_index] = TAGS[np.argmax(pi_q)] if k != 1 else "*"

        pre_tag, tag = np.where(pi[n - 1] == pi[n - 1].max())

        sentence.tokens[n - 2].tag = TAGS[pre_tag[0]]
        sentence.tokens[n - 1].tag = TAGS[tag[0]]

        for k in range(n - 3, -1, -1):
            sentence.tokens[k].tag = bp[k + 2,
                                        TAGS.index(sentence.tokens[k + 1].tag),
                                        TAGS.index(sentence.tokens[k + 2].tag)]

        print("Sentence {} viterbi done".format(s_index))
        print("{0:.3f} seconds".format(time.time() - start_time))
        return sentence, s_index

    def viterbi(self):
        start_time = time.time()

        with Pool(processes=PROCESSES) as pool:
            results = pool.starmap(self.viterbi_on_sentence, [(s, i) for i, s in enumerate(self.corpus.sentences)])
        for sentence, s_index in results:
            self.corpus.sentences[s_index] = sentence

        print("viterbi done: {0:.3f} seconds".format(time.time() - start_time))

    def load_model(self, model_name):
        path = BASIC_MODELS_PATH if self.is_basic else ADVANCED_MODELS_PATH
        self.v = np.loadtxt(path + model_name)

    def evaluate_model(self, ground_truth):
        results = {
            "correct": 0,
            "errors": 0
        }
        for s1, s2 in zip(self.corpus.sentences, ground_truth.sentences):
            for t1, t2 in zip(s1.tokens, s2.tokens):
                if t1.tag == t2.tag:
                    results["correct"] += 1
                else:
                    results["errors"] += 1

        return results, results["correct"] / sum(results.values())

    def print_results_to_file(self, tagged_file, model_name, is_test):
        if is_test:
            path = BASIC_TEST_PATH if self.is_basic else ADVANCED_TEST_PATH
        else:
            path = BASIC_COMP_PATH if self.is_basic else ADVANCED_COMP_PATH

        f = open(path + model_name, 'w')
        for s_truth, s_eval in zip(tagged_file.sentences, self.corpus.sentences):
            f.write(" ".join(["{}_{}".format(t_truth.word, t_eval.tag) for t_truth, t_eval
                              in zip(s_truth.tokens, s_eval.tokens)]) + "\n")

    def confusion_matrix(self, ground_truth, model_name, is_test):
        if is_test:
            path = BASIC_TEST_PATH if self.is_basic else ADVANCED_TEST_PATH
        else:
            path = BASIC_COMP_PATH if self.is_basic else ADVANCED_COMP_PATH

        for s1, s2 in zip(self.corpus.sentences, ground_truth.sentences):
            for t1, t2 in zip(s1.tokens, s2.tokens):
                self.matrix[TAGS.index(t1.tag)][TAGS.index(t2.tag)] += 1
        np.savetxt(path + "{}_confusion_matrix".format(model_name), self.matrix)

        file = open(path + "{}_confusion_matrix_lines".format(model_name), "w")
        for i in range(len(TAGS)):
            for j in range(len(TAGS)):
                value = self.matrix[i][j]
                file.write("Truth tag: {}, Predicted tag: {}, number of errors: {}\n".format(TAGS[j], TAGS[i], value))
        file.close()

    def confusion_matrix_zeros_diagonal(self):
        for i in range(len(TAGS)):
            self.matrix[i][i] = 0

    def confusion_matrix_ten_max_errors(self, model_name, is_test):
        if is_test:
            path = BASIC_TEST_PATH if self.is_basic else ADVANCED_TEST_PATH
        else:
            path = BASIC_COMP_PATH if self.is_basic else ADVANCED_COMP_PATH

        self.confusion_matrix_zeros_diagonal()
        ten_max_values = {}
        file_name = "{}_confusion_matrix_ten_max_errors".format(model_name)
        file = open(path + file_name, "w")
        for k in range(10):
            i, j = np.unravel_index(self.matrix.argmax(), self.matrix.shape)
            value = self.matrix[i][j]
            ten_max_values[(i, j)] = value
            self.matrix[i][j] = 0
            file.write("Truth tag: {}, Predicted tag: {}, number of errors: {}\n".format(TAGS[j], TAGS[i], value))
        file.close()
        return ten_max_values


class FeatureVector:
    def __init__(self, corpus):
        self.corpus = corpus
        self.document = {-1: {}, 1: {}}
        self.sentence = {}
        self.sentence_document = {}
        self.pre_sentence_sentence = {}
        self.pre_sentence_sentence_document = {}
        for i in range(1, 15):
            self.document[1][i] = {}
            self.document[0][i] = {}
        for i in range(-1, 2):
            for j in range(1, 15):
                self.sentence[i][j] = {}
        for i in range(-1, 2):
            for j in [-1, 1]:
                for k in range(1, 15):
                    self.sentence_document[(i, j)][k] = {}
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(1, 15):
                    self.pre_sentence_sentence[(i, j)][k] = {}
        for i in range(-1, 2):
            for j in range(-1, 2):
                for d in [-1, 1]:
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
        feature_types = ["f_1_word_tag", "f_2_tag, f_3_bigram", "f_4_bigram_none", "f_5_bigram_none",
                         "f_6_bigram_none_none",
                         "f_7_trigram", "f_8_trigram_pre_pre_none", "f_9_trigram_pre_none", "f_10_trigram_pre_none",
                         "f_11_trigram_pre_pre_none_pre_none", "f_12_trigram_pre_pre_none_none",
                         "f_13_trigram_pre_none_none",
                         "f_14_trigram_none_none_none"]
        if pre_sentence_label and sentence_label and document_label:
            for feature_type in feature_types:
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                self.pre_sentence_sentence_document[(pre_sentence_label, sentence_label, document_label)][
                    predicate] = self.index
                self.increment_index()

        if not pre_sentence_label and not sentence_label and document_label:
            for feature_type in feature_types:
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                self.sentence_document[document_label][predicate] = self.index
                self.increment_index()

        if not pre_sentence_label and sentence_label and not document_label:
            for feature_type in feature_types:
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                self.sentence[sentence_label][predicate] = self.index
                self.increment_index()

        if not pre_sentence_label and sentence_label and document_label:
            for feature_type in feature_types:
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                self.sentence_document[(sentence_label, document_label)][
                    predicate] = self.index
                self.increment_index()

        if pre_sentence_label and sentence_label and not document_label:
            for feature_type in feature_types:
                predicate = getattr(self, feature_type)(sentence.tokens, index)
                self.pre_sentence_sentence[(pre_sentence_label, sentence_label)][
                    predicate] = self.index
                self.increment_index()

    def initialize_features(self, model):
        for document in self.corpus:
            for sen_index, sentence in enumerate(document.sentences):
                for index, token in enumerate(sentence.tokens):
                    if sen_index >= 1:
                        if model == STRUCTURED_JOINT:
                            self.initialize_feature_based_on_label(sentence, index, document[sen_index - 1].label,
                                                                   sentence.label, document.label)

                        if model in (STRUCTURED_JOINT, SENTENCE_STRUCTURED):
                                self.initialize_feature_based_on_label(sentence, index,
                                                                       pre_sentence_label=document[sen_index - 1].label,
                                                                       sentence_label=sentence.label)
                    if model in (STRUCTURED_JOINT, DOCUMENT_CLASSIFIER):
                        self.initialize_feature_based_on_label(sentence, index, document_label=document.label)

                    if model in (STRUCTURED_JOINT, SENTENCE_CLASSIFIER):
                        self.initialize_feature_based_on_label(sentence, index, sentence_label=sentence.label)

                    if model == STRUCTURED_JOINT:
                        self.initialize_feature_based_on_label(sentence, index, sentence_label=sentence.label,
                                                               document_label=document.label)

    def evaluate_feature_vector(self, document, sen_index, model, document_label=None, sentence_label=None,
                                pre_sentence_label=None):
        sentence = document[sen_index]
        document_label = document.label if document_label is None else document_label
        sentence_label = sentence.label if sentence_label is None else sentence_label
        pre_sentence_label = document[
            sen_index - 1].label if pre_sentence_label is None and sen_index >= 1 else pre_sentence_label  # TODO check what to do when sen_index=0 and pre_sentence_label is None?

        evaluated_feature = []
        for index, token in enumerate(sentence.tokens):
            pre_word = self.pre_word(sentence, index)
            pre_pre_word = self.pre_pre_word(sentence, index)
            pre_pre_tag, pre_tag = self.pre_tags(sentence, index)
            tag = token.tag
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
                    if value:
                        evaluated_feature.append(value)

            if model in (STRUCTURED_JOINT, SENTENCE_CLASSIFIER):
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.sentence[sentence_label][f_index + 1].get(arguments)
                    if value:
                        evaluated_feature.append(value)

            if model in (STRUCTURED_JOINT, SENTENCE_STRUCTURED):
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.pre_sentence_sentence[(pre_sentence_label, sentence_label)][f_index + 1].get(arguments)
                    if value:
                        evaluated_feature.append(value)

            if model == STRUCTURED_JOINT:
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.sentence_document[(sentence_label, document_label)][f_index + 1].get(arguments)
                    if value:
                        evaluated_feature.append(value)
                for f_index, arguments in enumerate(feature_arguments):
                    value = self.pre_sentence_sentence_document[(pre_sentence_label, sentence_label, document_label)][f_index + 1].get(arguments)
                    if value:
                        evaluated_feature.append(value)

        return np.array(evaluated_feature)

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
