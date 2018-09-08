from constants import *
from utils import hash_file


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


class Config:
    perform_train = True
    evaluate_over_train_set = True
    evaluate_over_test_set = True

    mira_k_random_labelings = 0
    mira_k_best_viterbi_labelings = 7
    mira_iterations = 5
    mira_batch_size = 8

    loss_type = 'plus'  # {'mult', 'plus'}
    doc_loss_factor = 1.2

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
    def train_corpus_feature_vector_filename(self):
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

    model_configurations = (
        ConfigurationOptionPrinter(attribute='model_type', name='model'),
        ConfigurationOptionPrinter(attribute='mira_k_random_labelings', name='k-rnd', print_iff_true=True),
        ConfigurationOptionPrinter(attribute='mira_k_best_viterbi_labelings', name='k-viterbi', print_iff_true=True),
        ConfigurationOptionPrinter(attribute='mira_iterations', name='iter'),
        ConfigurationOptionPrinter(attribute='mira_batch_size', name='batch',
                                   print_condition=lambda value, _: value and int(value) > 1),
        ConfigurationOptionPrinter(attribute='loss_type_str', name='loss',
                                   print_condition=lambda _, config: config.model_type in {DOCUMENT_CLASSIFIER, STRUCTURED_JOINT}),
        ConfigurationOptionPrinter(attribute='min_nr_feature_occurrences', name='min-feature-occ',
                                   print_iff_true=True),
        ConfigurationOptionPrinter(attribute='docs_train_filename_base_wo_ext', name='train-set'),
        ConfigurationOptionPrinter(attribute='train_data_hash')
    )

    @property
    def model_name(self):
        if not hasattr(self, '__model_name') or not self.__model_name:
            self.__model_name = '__'.join(filter(bool, (conf.print(self) for conf in self.model_configurations)))
        return self.__model_name

    @property
    def model_weights_filename(self):
        return self.model_name + '.txt'

    @property
    def model_confusion_matrix_filename(self):
        return self.model_name + '__confusion_matrix.txt'

    def verify(self):
        assert(self.model_type in MODELS)
