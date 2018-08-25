from constants import SENTENCE_CLASSIFIER, DOCUMENT_CLASSIFIER, SENTENCE_STRUCTURED, STRUCTURED_JOINT, \
    DOCUMENT_LABELS, SENTENCE_LABELS, MODELS
from document import Document
from utils import ProgressBar

import numpy as np


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
        pb = ProgressBar(len(self.corpus.documents))
        for doc_index, document in enumerate(self.corpus.documents, start=1):
            pb.start_next_task()
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
        pb.finish()
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
