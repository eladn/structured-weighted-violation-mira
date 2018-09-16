from corpus import Corpus
from document import Document
from sentence import Sentence
from sentiment_model_configuration import SentimentModelConfiguration
from utils import ProgressBar, Singleton
from constants import MODELS, DOCUMENT_CLASSIFIER, SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED, STRUCTURED_JOINT, \
    DOCUMENT_LABELS, SENTENCE_LABELS, FEATURES_EXTRACTORS_PATH

import numpy as np
from abc import ABC, abstractmethod
from itertools import islice
from scipy.sparse import csr_matrix
import pickle
import os


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
class SentenceFeatureAttributesExtractor(Singleton):

    # Not used. just for documenting and for reference to `feature_attribute_types`.
    all_feature_attribute_types = {
        1: "f_1_word_tag", 2: "f_2_tag", 3: "f_3_bigram", 4: "f_4_bigram_none", 5: "f_5_bigram_none",
        6: "f_6_bigram_none_none", 7: "f_7_trigram", 8: "f_8_trigram_pre_pre_none",
        9: "f_9_trigram_pre_none", 10: "f_10_trigram_pre_none", 11: "f_11_trigram_pre_pre_none_pre_none",
        12: "f_12_trigram_pre_pre_none_none", 13: "f_13_trigram_pre_none_none", 14: "f_14_trigram_none_none_none"}

    # feature_attribute_types = {
    #     1: "f_1_word_tag", 3: "f_3_bigram", 4: "f_4_bigram_none", 5: "f_5_bigram_none",
    #     7: "f_7_trigram", 8: "f_8_trigram_pre_pre_none",
    #     9: "f_9_trigram_pre_none", 10: "f_10_trigram_pre_none", 11: "f_11_trigram_pre_pre_none_pre_none",
    #     12: "f_12_trigram_pre_pre_none_none", 13: "f_13_trigram_pre_none_none", 14: "f_14_trigram_none_none_none"}

    # It turns out that these features benefit the most to the accuracy.
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
class CorpusFeaturesExtractor:
    MAX_NR_FEATURE_ATTRIBUTE_OCCURRENCES = 3

    all_feature_groups_types = [
        DocumentFeatureGroup(),
        SentenceFeatureGroup(),
        PreSentenceSentenceFeatureGroup(),
        SentenceDocumentFeatureGroup(),
        PreSentenceSentenceDocumentFeatureGroup()
    ]

    def __init__(self, model_config: SentimentModelConfiguration):
        self._model_config = model_config
        # A mapping from feature attribute to its index
        self._fa_to_fa_idx_mapping = {fa_type_idx: dict()
                                      for fa_type_idx in SentenceFeatureAttributesExtractor().feature_attribute_types}
        self._nr_feature_attrs = 0
        self._nr_feature_groups = 0
        self._fa_in_fg_mask = None
        self._fg_to_fg_idx_mapping = {}  # A mapping from feature group to its index
        self._nr_features = 0
        self._feature_groups_types = []
        self._features_idxs = None
        self.random_state = np.random.RandomState(seed=self._model_config.feature_extractor_random_state_seed)

    @staticmethod
    def load_or_create(model_config: SentimentModelConfiguration, train_corpus: Corpus,
                       save_to_file_if_create: bool = False):
        if False and os.path.isfile(FEATURES_EXTRACTORS_PATH + model_config.train_corpus_features_extractor_filename):
            with open(FEATURES_EXTRACTORS_PATH + model_config.train_corpus_features_extractor_filename, 'rb') as f:
                features_extractor = pickle.load(f)
            features_extractor.initialize_corpus_features(train_corpus)
        else:
            features_extractor = CorpusFeaturesExtractor(model_config)
            features_extractor.generate_features_from_corpus(train_corpus)
            if save_to_file_if_create:
                with open(FEATURES_EXTRACTORS_PATH + model_config.train_corpus_features_extractor_filename, 'wb') as f:
                    pickle.dump(features_extractor, f)
        return features_extractor

    @property
    def nr_features(self):
        return self._nr_features

    def iterate_over_feature_groups_indeces_of_sentence(self, document: Document, sentence: Sentence,
                                                        document_label=None, sentence_label=None,
                                                        pre_sentence_label=None):
        document_label, sentence_label, pre_sentence_label = sanitize_labels(
            document, sentence, document_label, sentence_label, pre_sentence_label)
        for feature_group_type_idx, feature_group_type in enumerate(self._feature_groups_types):
            fg_value = feature_group_type.get_value_for_sentence(
                document, sentence, document_label, sentence_label, pre_sentence_label)
            if fg_value is None:
                continue
            yield self._fg_to_fg_idx_mapping[feature_group_type_idx][fg_value]

    def generate_features_from_corpus(self, corpus: Corpus):
        self._initialize_feature_groups()
        self._initialize_sentences_feature_attributes_for_corpus(corpus, generate_new_fa_if_not_exists=True)
        self._initialize_features_mask(corpus)
        self._features_dilution()
        self._initialize_features_idxs()
        self._initialize_sentences_features_for_corpus(corpus)
        print('Number of features: {}'.format(self.nr_features))

    def _initialize_feature_groups(self):
        # TODO: doc!
        self._feature_groups_types = [
            feature_group
            for feature_group in self.all_feature_groups_types
            if self._model_config.model_type in feature_group.use_in_models]

        next_fg_idx = 0
        self._fg_to_fg_idx_mapping = {}
        for feature_group_type_idx, feature_group_type in enumerate(self._feature_groups_types):
            self._fg_to_fg_idx_mapping[feature_group_type_idx] = {}
            for fg_value in feature_group_type.iterate_over_possible_values():
                self._fg_to_fg_idx_mapping[feature_group_type_idx][fg_value] = next_fg_idx
                next_fg_idx += 1
        self._nr_feature_groups = next_fg_idx

    def _initialize_features_mask(self, corpus: Corpus):
        # TODO: doc!
        self.fa_in_fg_count = np.zeros((self._nr_feature_groups, self._nr_feature_attrs))
        self._fa_in_fg_mask = np.zeros((self._nr_feature_groups, self._nr_feature_attrs), dtype=bool)
        for document, sentence in corpus:
            for fg_idx in self.iterate_over_feature_groups_indeces_of_sentence(document, sentence):
                fa_idxs = sentence.feature_attributes_idxs
                self._fa_in_fg_mask[fg_idx, fa_idxs] = True
                self.fa_in_fg_count[fg_idx, fa_idxs] += 1

        self.nr_all_features = self._fa_in_fg_mask.sum()
        self._nr_features = self.nr_all_features

    def _features_dilution(self):
        # Feature dilution: limiting the model in order to avoid over-fitting.

        # dilution - method 1: remove features that has occurred a negligible number of times in the corpus.
        if self._model_config.min_nr_feature_occurrences is not None:
            self._fa_in_fg_mask = self._fa_in_fg_mask & (self.fa_in_fg_count >= self._model_config.min_nr_feature_occurrences)

        # dilution - method 2: randomly choosing a ratio of the features.
        # rnd = self.random_state.rand(*self.fa_in_fg_mask.shape) < self.config.features_dilution_ratio
        # self.fa_in_fg_mask &= rnd

        # dilution - method 3: sampling `nr_features` features from the discrete distribution
        #                      of number of occurrences per each feature.
        # if self.nr_all_features > self.config.nr_features:
        #     diluted_fa_in_fg_mask = np.zeros(self.fa_in_fg_mask.shape, dtype=bool)
        #
        #     p = self.fa_in_fg_count[self.fa_in_fg_mask]
        #     p /= p.sum()
        #     indeces_of_features_to_keep = self.random_state.choice(
        #         self.nr_all_features, self.config.nr_features, replace=False, p=p)
        #     features_to_keep_flattened_mask = np.zeros((self.nr_all_features, ), dtype=bool)
        #     features_to_keep_flattened_mask[indeces_of_features_to_keep] = True
        #     diluted_fa_in_fg_mask[self.fa_in_fg_mask] = features_to_keep_flattened_mask
        #
        #     # turn on important features
        #     feature_attributes_idxs_to_always_leave_on = set()
        #     FEATURE_ATTRIBUTES_TYPES_IDXS_TO_ALWAYS_LEAVE_ON = {1, 3}
        #     for fa_type_idx in FEATURE_ATTRIBUTES_TYPES_IDXS_TO_ALWAYS_LEAVE_ON:
        #         for _, fa_idxs_list in self.fa_to_fa_idx_mapping[fa_type_idx].items():
        #             for fa_idx in fa_idxs_list:
        #                 feature_attributes_idxs_to_always_leave_on.add(fa_idx)
        #     feature_attributes_idxs_to_always_leave_on = np.array(list(feature_attributes_idxs_to_always_leave_on))
        #     first_feature_mask = np.zeros(self.fa_in_fg_mask.shape, dtype=bool)
        #     first_feature_mask[:, feature_attributes_idxs_to_always_leave_on] = True
        #     diluted_fa_in_fg_mask[self.fa_in_fg_mask & first_feature_mask] = True
        #
        #     self.fa_in_fg_mask = diluted_fa_in_fg_mask

        # The number of features has now been changed, hence re-calculate `nr_features`.
        self._nr_features = self._fa_in_fg_mask.sum()

    def _initialize_features_idxs(self):
        # TODO: doc!
        self._features_idxs = np.zeros(self._fa_in_fg_mask.shape, dtype=int)
        self._features_idxs[self._fa_in_fg_mask] = np.arange(0, self._nr_features)

    def _initialize_sentences_features_for_corpus(self, corpus: Corpus=None):
        # TODO: doc!
        print("Initializing sentences features for corpus `{corpus_name}`.".format(corpus_name=corpus.name))
        pb = ProgressBar(len(corpus.documents))
        for document, sentence in corpus:
            if sentence.index == 0:
                pb.start_next_task()
            sentence.features = dict()
            fa_idxs = sentence.feature_attributes_idxs
            for fg_idx in range(self._nr_feature_groups):
                fa_mask = self._fa_in_fg_mask[fg_idx, fa_idxs]
                features = self._features_idxs[fg_idx, fa_idxs[fa_mask]]
                sentence.features[fg_idx] = features
        pb.finish()

    def _initialize_sentences_feature_attributes_for_corpus(self, corpus: Corpus, generate_new_fa_if_not_exists=False):
        """
            Goes over the corpus and assign a unique index for each feature attribute in the corpus.
            Store for each sentence a numpy 1D array of the indeces of the feature attributes in
              that sentence, sorted by the idx.
        """
        title = "Initializing feature attributes for corpus `{corpus_name}`.".format(corpus_name=corpus.name)
        if generate_new_fa_if_not_exists:
            title += " Generate new feature-attributes when needed (and add them to the feature vector)."
        else:
            title += " Use only feature-attributes already in the feature vector (do not create new)."
        print(title)
        pb = ProgressBar(len(corpus.documents))
        next_feature_attribute_index_to_generate = self._nr_feature_attrs
        for document, sentence in corpus:
            if sentence.index == 0:
                pb.start_next_task()

            sentence_fa_idxs = set()
            for fa_type_idx, fa_value in SentenceFeatureAttributesExtractor().iterate_over_sentence_attributes(sentence):
                if fa_value not in self._fa_to_fa_idx_mapping[fa_type_idx]:
                    if not generate_new_fa_if_not_exists:
                        continue
                    self._fa_to_fa_idx_mapping[fa_type_idx][fa_value] = list()

                fa_idx = None
                nr_fa_occurrences = len(self._fa_to_fa_idx_mapping[fa_type_idx][fa_value])
                for current_fa_occurrence in range(nr_fa_occurrences):
                    fa_idx = self._fa_to_fa_idx_mapping[fa_type_idx][fa_value][current_fa_occurrence]
                    if fa_idx not in sentence_fa_idxs:
                        break

                if generate_new_fa_if_not_exists and \
                        (fa_idx is None or fa_idx in sentence_fa_idxs) and \
                        nr_fa_occurrences < self.MAX_NR_FEATURE_ATTRIBUTE_OCCURRENCES:
                    self._fa_to_fa_idx_mapping[fa_type_idx][fa_value].append(next_feature_attribute_index_to_generate)
                    next_feature_attribute_index_to_generate += 1
                    fa_idx = self._fa_to_fa_idx_mapping[fa_type_idx][fa_value][-1]
                    assert fa_idx not in sentence_fa_idxs
                assert fa_idx is not None

                sentence_fa_idxs.add(fa_idx)

            sentence_fa_idxs = np.array(list(sentence_fa_idxs), dtype=int)
            sentence_fa_idxs.sort()
            sentence.feature_attributes_idxs = sentence_fa_idxs
        pb.finish()
        if generate_new_fa_if_not_exists:
            self._nr_feature_attrs += next_feature_attribute_index_to_generate

    def initialize_corpus_features(self, corpus: Corpus):
        self._initialize_sentences_feature_attributes_for_corpus(corpus, generate_new_fa_if_not_exists=False)
        self._initialize_sentences_features_for_corpus(corpus)

    def evaluate_clique_feature_vector(self, document: Document, sentence: Sentence,
                                       document_label=None, sentence_label=None, pre_sentence_label=None):
        document_label, sentence_label, pre_sentence_label = sanitize_labels(
            document, sentence, document_label, sentence_label, pre_sentence_label)

        fv = [sentence.features[fg_idx]
              for fg_idx in self.iterate_over_feature_groups_indeces_of_sentence(
                document, sentence, document_label, sentence_label, pre_sentence_label)]
        return np.concatenate(fv)

    def evaluate_document_feature_vector(self, document: Document, y_tag=None):
        start_from_sentence_idx = 1 if self._model_config.use_pre_sentence else 0
        nr_sentences = document.count_sentences() - start_from_sentence_idx
        y_document = y_tag[0] if y_tag is not None else None
        row_ind, col_ind = [], []
        for sentence in islice(document.sentences, start_from_sentence_idx, None):
            y_sentence = y_tag[sentence.index + 1] if y_tag is not None else None
            y_pre_sentence = y_tag[sentence.index] if y_tag is not None else None
            feature_indices = self.evaluate_clique_feature_vector(
                document, sentence, y_document, y_sentence, y_pre_sentence)
            col_ind += list(feature_indices)  # TODO: consider stacking it all in array (size can be known)
            row_ind += [sentence.index - start_from_sentence_idx for _ in feature_indices]
        return csr_matrix(([1 for _ in col_ind], (row_ind, col_ind)), shape=(nr_sentences, self.nr_features))
