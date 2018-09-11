from constants import *
from utils import hash_file, multi_dicts_product_iterator


class ConfigurationOptionPrinter:
    def __init__(self,
                 attribute: str=None,
                 name: str=None,
                 description: str=None,
                 value_type=None,
                 print_condition=None,
                 print_iff_not_none: bool=False,
                 print_iff_true: bool=False,
                 print_value_format_str=None,
                 print_value_function=None,
                 name_value_separator: str=None):

        if print_value_format_str is not None and print_value_function is not None:
            raise RuntimeError(
                'ConfigurationOption constructor: Cannot specify both `print_value_format_str` and `print_value_function`.')
        if bool(print_condition is not None) + bool(print_iff_not_none) + bool(print_iff_true) > 1:
            raise RuntimeError(
                'ConfigurationOption constructor: Cannot specify more than one of: `print_condition`, `print_iff_not_none` and `print_iff_true`.')
        if attribute is None and print_value_function is None:
            raise RuntimeError(
                'ConfigurationOption constructor: Cannot specify neither `attribute` nor `print_value_function`.')

        self.name_value_separator = str(name_value_separator) if name_value_separator is not None else '='
        self.attribute = attribute
        self.name = name
        self.description = description
        self.value_type = value_type
        self.print_condition = print_condition
        self.print_iff_not_none = print_iff_not_none
        self.print_iff_true = print_iff_true
        self.print_value_format_str = print_value_format_str
        self.print_value_function = print_value_function

    def print(self, config):
        if self.attribute and not hasattr(config, self.attribute):
            return ''
        value = getattr(config, self.attribute) if isinstance(self.attribute, str) else None
        if self.print_iff_not_none and value is None:
            return ''
        if self.print_iff_true and not value:
            return ''
        if self.print_condition is not None and not self.print_condition(value, config):
            return ''
        name_and_separator = (str(self.name) + self.name_value_separator) if self.name else ''
        if self.print_value_function is not None:
            return name_and_separator + self.print_value_function(value, config)
        print_fmt = '{}'
        if self.print_value_format_str is not None:
            print_fmt = '{' + str(self.print_value_format_str) + '}'
        printed_value = print_fmt.format(value)
        return name_and_separator + printed_value


class SentimentModelConfiguration:
    mira_k_random_labelings = 0
    mira_k_best_viterbi_labelings = 15
    mira_iterations = 5
    mira_batch_size = 8

    loss_type = 'max'  # {'mult', 'plus', 'max'}
    doc_loss_factor = 1

    min_nr_feature_occurrences = 3

    docs_train_filename_base_wo_ext = "train-0.6p"
    docs_test_filename_base_wo_ext = "test-0.2p"

    # model_type = DOCUMENT_CLASSIFIER
    # model_type = SENTENCE_CLASSIFIER
    # model_type = STRUCTURED_JOINT
    model_type = SENTENCE_STRUCTURED

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
        ConfigurationOptionPrinter(attribute='docs_train_filename_base_wo_ext', name='corpus'),
        ConfigurationOptionPrinter(attribute='model_type', name='model'),
        ConfigurationOptionPrinter(attribute='min_nr_feature_occurrences', name='min-feature-occ',
                                   print_iff_true=True),
        ConfigurationOptionPrinter(attribute='train_data_hash')
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
        return self.model_type in {DOCUMENT_CLASSIFIER, STRUCTURED_JOINT}

    @property
    def infer_sentences_labels(self):
        return self.model_type != DOCUMENT_CLASSIFIER

    model_configurations = (
        ConfigurationOptionPrinter(attribute='model_type', name='model'),
        ConfigurationOptionPrinter(attribute='mira_k_random_labelings', name='k-rnd', print_iff_true=True),
        ConfigurationOptionPrinter(attribute='mira_k_best_viterbi_labelings', name='k-viterbi', print_iff_true=True),
        ConfigurationOptionPrinter(attribute='mira_iterations', name='iter'),
        ConfigurationOptionPrinter(attribute='mira_batch_size', name='batch',
                                   print_condition=lambda value, _: value and int(value) > 1),
        ConfigurationOptionPrinter(attribute='loss_type_str', name='loss',
                                   print_condition=lambda _, config: config.model_type == STRUCTURED_JOINT),
        ConfigurationOptionPrinter(attribute='min_nr_feature_occurrences', name='min-feature-occ',
                                   print_iff_true=True),
        ConfigurationOptionPrinter(attribute='docs_train_filename_base_wo_ext', name='train-set'),
        ConfigurationOptionPrinter(attribute='train_data_hash')
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
        assert(self.model_type in MODELS)

    def clone(self):
        new_config = SentimentModelConfiguration()
        for param_name, _, default in self.get_all_settable_params():
            setattr(new_config, param_name, getattr(self, param_name, default))
        return new_config

    @classmethod
    def get_all_params(cls):
        cnf = cls()
        return [(attr, type(getattr(cnf, attr)), getattr(cnf, attr)) for attr in dir(cnf)
                if not callable(getattr(cnf, attr)) and not attr.startswith("__")]

    @classmethod
    def get_all_settable_params(cls):
        return [(param_name, _type, default)
                for param_name, _type, default in cls.get_all_params()
                if _type in {str, int, float, bool} and not isinstance(getattr(cls, param_name, None), property)]

    def iterate_over_configurations(self, *args, **kwargs):
        """
            Input example:
            iterate_over_configurations(
                [ {'mira_k_random_labelings': 0, 'mira_k_best_viterbi_labelings': 10},
                  {'mira_k_random_labelings': 10, 'mira_k_best_viterbi_labelings': 0} ],
                [ {'model_type': [SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED], 'loss_type': 'plus'},
                  {'model_type': [DOCUMENT_CLASSIFIER, STRUCTURED_JOINT], 'loss_type': ['plus', 'mult']} ],
                mira_iterations = [3, 4, 5, 6],
                min_nr_feature_occurrences = [2, 3, 4]
            )
        """
        for values_dict in multi_dicts_product_iterator(*args, **kwargs):
            config = self.clone()
            for key, value in values_dict.items():
                setattr(config, key, value)
            yield config
