from constants import *
from utils import hash_file, AttributePrinter


class SentimentModelConfiguration:
    optimizer = 'qp'  # {'qp', 'cvx'}
    trainer_alg = 'mira'  # {'mira', 'SWVM'}
    training_k_random_labelings = 0
    training_k_best_viterbi_labelings = 15
    training_iterations = 5
    training_batch_size = 8

    # The loss type controls how does the document loss is combined with the sentences loss in the joint model.
    loss_type = 'plus'  # {'mult', 'plus', 'max'}
    doc_loss_factor = 0.2

    min_nr_feature_occurrences = 3

    docs_train_filename_base_wo_ext = "train-0.6p"
    docs_test_filename_base_wo_ext = "test-0.2p"

    # model_type = DOCUMENT_CLASSIFIER
    # model_type = SENTENCE_CLASSIFIER
    # model_type = STRUCTURED_JOINT
    model_type = SENTENCE_STRUCTURED

    feature_extractor_random_state_seed = 0

    def is_param_used(self, param_name: str):
        if param_name == 'loss_type':
            return self.model_type == STRUCTURED_JOINT
        if param_name == 'doc_loss_factor':
            return self.model_type == STRUCTURED_JOINT and self.loss_type == 'plus'
        return True

    @property
    def pos_docs_train_filename(self):
        return 'pos-' + self.docs_train_filename_base_wo_ext + '.txt'

    @property
    def neg_docs_train_filename(self):
        return 'neg-' + self.docs_train_filename_base_wo_ext + '.txt'

    @property
    def pos_docs_test_filename(self):
        return 'pos-' + self.docs_test_filename_base_wo_ext + '.txt'

    @property
    def neg_docs_test_filename(self):
        return 'neg-' + self.docs_test_filename_base_wo_ext + '.txt'

    @property
    def train_data_hash(self):
        if hasattr(self, '__train_data_hash'):
            return self.__train_data_hash
        self.__train_data_hash = hash_file((
            DATA_PATH + self.pos_docs_train_filename,
            DATA_PATH + self.neg_docs_train_filename), hash_type='md5')
        return self.__train_data_hash

    corpus_configurations = (
        AttributePrinter(attribute='docs_train_filename_base_wo_ext', name='corpus'),
        AttributePrinter(attribute='model_type', name='model'),
        AttributePrinter(attribute='min_nr_feature_occurrences', name='min-feature-occ',
                         print_iff_true=True),
        AttributePrinter(attribute='train_data_hash')
    )

    @property
    def train_corpus_features_extractor_filename(self):
        if not hasattr(self, '__train_corpus_feature_vector_filename') or not self.__train_corpus_feature_vector_filename:
            self.__train_corpus_feature_vector_filename = 'corpus-feature-vector__' + '__'.join(
                filter(bool, (conf.print(self) for conf in self.corpus_configurations)))
        return self.__train_corpus_feature_vector_filename

    @property
    def loss_type_str(self):
        loss_type_str = str(self.loss_type)
        if self.loss_type == 'plus':
            loss_type_str += str(self.doc_loss_factor)
        return loss_type_str

    @property
    def infer_document_label(self):
        return self.model_type in INFER_DOCUMENT_MODEL_TYPES

    @property
    def infer_sentences_labels(self):
        return self.model_type in INFER_SENTENCES_MODEL_TYPES

    @property
    def use_pre_sentence(self):
        return self.model_type in {SENTENCE_STRUCTURED, STRUCTURED_JOINT}

    model_configurations = (
        AttributePrinter(attribute='model_type', name='model'),
        AttributePrinter(attribute='trainer_alg', name='alg',
                         print_condition=lambda val, _: val != 'mira'),
        AttributePrinter(attribute='training_k_random_labelings', name='k-rnd', print_iff_true=True),
        AttributePrinter(attribute='training_k_best_viterbi_labelings', name='k-viterbi', print_iff_true=True),
        AttributePrinter(attribute='training_iterations', name='iter'),
        AttributePrinter(attribute='training_batch_size', name='batch',
                         print_condition=lambda value, _: value and int(value) > 1),
        AttributePrinter(attribute='loss_type_str', name='loss',
                         print_condition=lambda _, config: config.model_type == STRUCTURED_JOINT),
        AttributePrinter(attribute='min_nr_feature_occurrences', name='min-feature-occ',
                         print_iff_true=True),
        AttributePrinter(attribute='docs_train_filename_base_wo_ext', name='train-set'),
        AttributePrinter(attribute='train_data_hash')
    )

    def to_string(self, separator='__'):
        if not hasattr(self, '__model_properties') or not self.__model_properties:
            self.__model_properties = list(filter(bool, (conf.print(self) for conf in self.model_configurations)))
            assert(all(isinstance(s, str) for s in self.__model_properties))
        if separator is None:
            return list(self.__model_properties)
        return separator.join(self.__model_properties)

    @property
    def model_name(self):
        return self.to_string(separator='__')

    def __str__(self):
        return self.to_string(', ')

    @property
    def model_weights_filename(self):
        return self.model_name + '.txt'

    @property
    def model_confusion_matrix_filename(self):
        return self.model_name + '__confusion_matrix.txt'

    def verify(self):
        assert(self.model_type in MODEL_TYPES)

    def clone(self):
        new_config = SentimentModelConfiguration()
        for param_name, _, default in self.get_all_settable_params():
            setattr(new_config, param_name, getattr(self, param_name, default))
        return new_config

    def from_dict(self, params_dict: dict):
        for param_name, param_value in params_dict.items():
            setattr(self, param_name, param_value)

    def to_dict(self):
        ret_dct = {
            param_name: getattr(self, param_name, default)
            for param_name, _, default in self.get_all_settable_params(flg_only_if_used=True)
        }
        return ret_dct

    @classmethod
    def get_all_params(cls):
        cnf = cls()
        return [(attr, type(getattr(cnf, attr)), getattr(cnf, attr)) for attr in dir(cnf)
                if not callable(getattr(cnf, attr)) and not attr.startswith("__")]

    def get_all_settable_params(self, flg_only_if_used: bool = False):
        return [(param_name, _type, default)
                for param_name, _type, default in self.get_all_params()
                if _type in {str, int, float, bool}
                and (not isinstance(getattr(type(self), param_name, None), property))
                and (not flg_only_if_used or self.is_param_used(param_name))]

    def iterate_over_configurations(self, params_dicts):
        for params_dict in params_dicts:
            config = self.clone()
            config.from_dict(params_dict)
            yield config
