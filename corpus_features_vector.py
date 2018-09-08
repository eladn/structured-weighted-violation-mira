from corpus import Corpus
from document import Document
from sentence import Sentence
from config import Config
from utils import ProgressBar, Singleton
from constants import MODELS, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED, STRUCTURED_JOINT, \
    DOCUMENT_LABELS, SENTENCE_LABELS

import numpy as np
from abc import ABC, abstractmethod


def sanitize_labels(document: Document, sentence: Sentence,
                    document_label=None, sentence_label=None, pre_sentence_label=None):
    assert document_label is None or document_label in DOCUMENT_LABELS
    assert sentence_label is None or sentence_label in SENTENCE_LABELS
    assert pre_sentence_label is None or pre_sentence_label in SENTENCE_LABELS

    document_label = document.label if document_label is None else document_label
    sentence_label = sentence.label if sentence_label is None else sentence_label

    if sentence.index == 0:
        pre_sentence_label = None
    else:
        pre_sentence_label = document.sentences[
            sentence.index - 1].label if pre_sentence_label is None else pre_sentence_label

    return document_label, sentence_label, pre_sentence_label


# TODO: doc!
class SentenceFeatureAttributes(Singleton):

    # feature_attribute_types = [
    #     "f_1_word_tag", "f_2_tag", "f_3_bigram", "f_4_bigram_none", "f_5_bigram_none",
    #     "f_6_bigram_none_none", "f_7_trigram", "f_8_trigram_pre_pre_none",
    #     "f_9_trigram_pre_none", "f_10_trigram_pre_none", "f_11_trigram_pre_pre_none_pre_none",
    #     "f_12_trigram_pre_pre_none_none", "f_13_trigram_pre_none_none", "f_14_trigram_none_none_none"]

    feature_attribute_types = {1: "f_1_word_tag", 3: "f_3_bigram"}

    feature_attribute_types_name2idx_mapping = {name: idx for idx, name in feature_attribute_types.items()}

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

    def iterate_over_sentence_attributes(self, sentence: Sentence):
        for token_idx in range(len(sentence.tokens)):
            for fa_type_idx, fa_type_name in self.feature_attribute_types.items():
                fa_value = getattr(self, fa_type_name)(sentence.tokens, token_idx)
                yield fa_type_idx, fa_value


# TODO: doc!
class FeatureGroup(ABC):
    def __init__(self, use_in_models):
        assert(isinstance(use_in_models, set) and all(model in MODELS for model in use_in_models))
        self.use_in_models = use_in_models

    @abstractmethod
    def iterate_over_possible_values(self):
        ...

    @abstractmethod
    def get_value_for_sentence(self, document: Document, sentence: Sentence,
                               document_label, sentence_label, pre_sentence_label):
        ...


class DocumentFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__({STRUCTURED_JOINT, DOCUMENT_CLASSIFIER})

    def iterate_over_possible_values(self):
        for document_label in DOCUMENT_LABELS:
            yield document_label

    def get_value_for_sentence(self, document: Document, sentence: Sentence,
                               document_label, sentence_label, pre_sentence_label):
        return document_label
        # return document.label


class SentenceFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__({STRUCTURED_JOINT, SENTENCE_STRUCTURED, SENTENCE_CLASSIFIER})

    def iterate_over_possible_values(self):
        for sentence_label in SENTENCE_LABELS:
            yield sentence_label

    def get_value_for_sentence(self, document: Document, sentence: Sentence,
                               document_label, sentence_label, pre_sentence_label):
        return sentence_label
        # return document.sentences[sentence_idx].label


class PreSentenceSentenceFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__({STRUCTURED_JOINT, SENTENCE_STRUCTURED})

    def iterate_over_possible_values(self):
        for pre_sentence_label in SENTENCE_LABELS:
            for sentence_label in SENTENCE_LABELS:
                yield pre_sentence_label, sentence_label

    def get_value_for_sentence(self, document: Document, sentence: Sentence,
                               document_label, sentence_label, pre_sentence_label):
        if sentence.index < 1:
            return None
        return pre_sentence_label, sentence_label
        # return document.sentences[sentence_idx-1].label, document.sentences[sentence_idx].label


class SentenceDocumentFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__({STRUCTURED_JOINT})

    def iterate_over_possible_values(self):
        for sentence_label in SENTENCE_LABELS:
            for document_label in DOCUMENT_LABELS:
                yield sentence_label, document_label

    def get_value_for_sentence(self, document: Document, sentence: Sentence,
                               document_label, sentence_label, pre_sentence_label):
        return sentence_label, document_label
        # return document.sentences[sentence_idx].label, document.label


class PreSentenceSentenceDocumentFeatureGroup(FeatureGroup):
    def __init__(self):
        super().__init__({STRUCTURED_JOINT})

    def iterate_over_possible_values(self):
        for pre_sentence_label in SENTENCE_LABELS:
            for sentence_label in SENTENCE_LABELS:
                for document_label in DOCUMENT_LABELS:
                    yield pre_sentence_label, sentence_label, document_label

    def get_value_for_sentence(self, document: Document, sentence: Sentence,
                               document_label, sentence_label, pre_sentence_label):
        if sentence.index < 1:
            return None
        return pre_sentence_label, sentence_label, document_label
        # return document.sentences[sentence_idx - 1].label, document.sentences[sentence_idx].label, document.label


# TODO: doc!
# feature groups are dependent of the labels and the model
# feature attributes are dependent only of the corpus
class CorpusFeaturesVector:
    all_feature_groups_types = [
        DocumentFeatureGroup(),
        SentenceFeatureGroup(),
        PreSentenceSentenceFeatureGroup(),
        SentenceDocumentFeatureGroup(),
        PreSentenceSentenceDocumentFeatureGroup()
    ]

    def __init__(self, corpus: Corpus, config: Config):
        self.corpus = corpus
        self.config = config
        self.fa_to_fa_idx_mapping = {}  # A mapping from feature attribute to its index
        self.nr_feature_attrs = 0
        self.nr_feature_groups = 0
        self.fa_in_fg_mask = None
        self.fg_to_fg_idx_mapping = {}  # A mapping from feature group to its index
        self.nr_features = 0
        self.feature_groups_types = []
        self.features_idxs = None

    @property
    def size(self):
        return self.nr_features

    def iterate_over_feature_groups_indeces_of_sentence(self, document: Document, sentence: Sentence,
                                                        document_label=None, sentence_label=None,
                                                        pre_sentence_label=None):
        document_label, sentence_label, pre_sentence_label = sanitize_labels(
            document, sentence, document_label, sentence_label, pre_sentence_label)
        for feature_group_type_idx, feature_group_type in enumerate(self.feature_groups_types):
            fg_value = feature_group_type.get_value_for_sentence(
                document, sentence, document_label, sentence_label, pre_sentence_label)
            if fg_value is None:
                continue
            yield self.fg_to_fg_idx_mapping[feature_group_type_idx][fg_value]

    def initialize_features(self):
        self._initialize_feature_groups()
        self._initialize_feature_attributes()
        self._initialize_features_idxs()
        self.initialize_sentence_features()

    def _initialize_feature_groups(self):
        self.feature_groups_types = [
            feature_group
            for feature_group in self.all_feature_groups_types
            if self.config.model_type in feature_group.use_in_models]

        next_fg_idx = 0
        self.fg_to_fg_idx_mapping = {}
        for feature_group_type_idx, feature_group_type in enumerate(self.feature_groups_types):
            self.fg_to_fg_idx_mapping[feature_group_type_idx] = {}
            for fg_value in feature_group_type.iterate_over_possible_values():
                self.fg_to_fg_idx_mapping[feature_group_type_idx][fg_value] = next_fg_idx
                next_fg_idx += 1
        self.nr_feature_groups = next_fg_idx

    def _initialize_feature_attributes(self):
        # Here we go over the corpus and assign a unique index for each feature attribute in the corpus.
        # We also store for each sentence a numpy 1D array of the indeces of the feature attributes in
        #   that sentence, sorted by the idx.
        print("Initializing feature attributes")
        pb = ProgressBar(len(self.corpus.documents))
        # A mapping from feature attribute to its index
        self.fa_to_fa_idx_mapping = {fa_type_idx: dict()
                                     for fa_type_idx in SentenceFeatureAttributes().feature_attribute_types}
        next_feature_attribute_index = 0
        for document, sentence in self.corpus:
            if sentence.index == 0:
                pb.start_next_task()
            sentence_fa_idxs = set()
            for fa_type_idx, fa_value in SentenceFeatureAttributes().iterate_over_sentence_attributes(sentence):
                if fa_value not in self.fa_to_fa_idx_mapping[fa_type_idx]:
                    self.fa_to_fa_idx_mapping[fa_type_idx][fa_value] = next_feature_attribute_index
                    next_feature_attribute_index += 1
                    sentence_fa_idxs.add(self.fa_to_fa_idx_mapping[fa_type_idx][fa_value])
            sentence_fa_idxs = np.array(list(sentence_fa_idxs), dtype=int)
            sentence_fa_idxs.sort()
            sentence.feature_attributes_idxs = sentence_fa_idxs
        pb.finish()

        self.nr_feature_attrs = next_feature_attribute_index

    def _initialize_features_idxs(self):
        p = np.zeros((self.nr_feature_groups, self.nr_feature_attrs))
        self.fa_in_fg_mask = np.zeros((self.nr_feature_groups, self.nr_feature_attrs), dtype=bool)
        for document, sentence in self.corpus:
            for fg_idx in self.iterate_over_feature_groups_indeces_of_sentence(document, sentence):
                fa_idxs = sentence.feature_attributes_idxs
                self.fa_in_fg_mask[fg_idx, fa_idxs] = True
                p[fg_idx, fa_idxs] += 1

        self.nr_all_features = self.fa_in_fg_mask.sum()

        # DILUL
        # TODO: DOC!
        # rnd = np.random.rand(*self.fa_in_fg_mask.shape) < self.config.features_dilution_ratio
        # self.fa_in_fg_mask &= rnd
        if False and self.nr_all_features > self.config.nr_features:
            diluted_fa_in_fg_mask = np.zeros(self.fa_in_fg_mask.shape, dtype=bool)

            p = p[self.fa_in_fg_mask]
            p /= p.sum()
            indeces_of_features_to_keep = np.random.choice(
                self.nr_all_features, self.config.nr_features, replace=False, p=p)
            features_to_keep_flattened_mask = np.zeros((self.nr_all_features, ), dtype=bool)
            features_to_keep_flattened_mask[indeces_of_features_to_keep] = True
            diluted_fa_in_fg_mask[self.fa_in_fg_mask] = features_to_keep_flattened_mask

            # turn on important features
            feature_attributes_idxs_to_always_leave_on = set()
            FEATURE_ATTRIBUTES_TYPES_IDXS_TO_ALWAYS_LEAVE_ON = {1, 3}
            for fa_type_idx in FEATURE_ATTRIBUTES_TYPES_IDXS_TO_ALWAYS_LEAVE_ON:
                for _, fa_idx in self.fa_to_fa_idx_mapping[fa_type_idx].items():
                    feature_attributes_idxs_to_always_leave_on.add(fa_idx)
            feature_attributes_idxs_to_always_leave_on = np.array(list(feature_attributes_idxs_to_always_leave_on))
            first_feature_mask = np.zeros(self.fa_in_fg_mask.shape, dtype=bool)
            first_feature_mask[:, feature_attributes_idxs_to_always_leave_on] = True
            diluted_fa_in_fg_mask[self.fa_in_fg_mask & first_feature_mask] = True

            self.fa_in_fg_mask = diluted_fa_in_fg_mask

        self.nr_features = self.fa_in_fg_mask.sum()
        # assert(self.nr_features >= self.config.nr_features)

        self.features_idxs = np.zeros(self.fa_in_fg_mask.shape, dtype=int)
        self.features_idxs[self.fa_in_fg_mask] = np.arange(0, self.nr_features)

    def initialize_sentence_features(self, corpus: Corpus=None):
        corpus = self.corpus if corpus is None else corpus
        print("Initializing sentences features")
        pb = ProgressBar(len(corpus.documents))
        for document, sentence in corpus:
            if sentence.index == 0:
                pb.start_next_task()
            sentence.features = dict()
            fa_idxs = sentence.feature_attributes_idxs
            for fg_idx in range(self.nr_feature_groups):
                fa_mask = self.fa_in_fg_mask[fg_idx, fa_idxs]
                features = self.features_idxs[fg_idx, fa_idxs[fa_mask]]
                sentence.features[fg_idx] = features

                # mrk = '<--' if fa_idxs.shape[0] == features.shape[0] else ''
                # print(fg_idx, fa_idxs.shape, fa_mask.shape, features.shape, mrk)
            # print()
        pb.finish()

    def initialize_corpus_features(self, corpus: Corpus):
        print("Initializing corpus features")
        pb = ProgressBar(len(corpus.documents))
        for document, sentence in corpus:
            if sentence.index == 0:
                pb.start_next_task()

            # sentence_fa_idxs = set()
            # for fa_type_idx, fa_value in SentenceFeatureAttributes().iterate_over_sentence_attributes(sentence):
            #     if fa_value not in self.fa_to_fa_idx_mapping[fa_type_idx]:
            #         continue
            #
            #     fa_idx = None
            #     nr_occurances = len(self.fa_to_fa_idx_mapping[fa_type_idx][fa_value])
            #     for occurance in range(nr_occurances):
            #         fa_idx = self.fa_to_fa_idx_mapping[fa_type_idx][fa_value][occurance]
            #         if fa_idx not in sentence_fa_idxs:
            #             break
            #     assert fa_idx is not None
            #
            #     sentence_fa_idxs.add(fa_idx)

            sentence_fa_idxs = {
                self.fa_to_fa_idx_mapping[fa_type_idx][fa_value]
                for fa_type_idx, fa_value in SentenceFeatureAttributes().iterate_over_sentence_attributes(sentence)
                if fa_value in self.fa_to_fa_idx_mapping[fa_type_idx]
            }

            sentence_fa_idxs = np.array(list(sentence_fa_idxs), dtype=int)
            sentence_fa_idxs.sort()
            sentence.feature_attributes_idxs = sentence_fa_idxs
        pb.finish()
        self.initialize_sentence_features(corpus)

    def evaluate_clique_feature_vector(self, document: Document, sentence: Sentence,
                                       document_label=None, sentence_label=None, pre_sentence_label=None):
        document_label, sentence_label, pre_sentence_label = sanitize_labels(
            document, sentence, document_label, sentence_label, pre_sentence_label)

        fv = [sentence.features[fg_idx]
              for fg_idx in self.iterate_over_feature_groups_indeces_of_sentence(
                document, sentence, document_label, sentence_label, pre_sentence_label)]
        return np.concatenate(fv)
