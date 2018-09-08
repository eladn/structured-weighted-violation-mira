from constants import *
from utils import hash_file


class Config:
    perform_train = True
    evaluate_over_train_set = True
    evaluate_over_test_set = True

    mira_k_random_labelings = 0
    mira_k_best_viterbi_labelings = 5
    mira_iterations = 5

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

    @property
    def train_corpus_feature_vector_filename(self):
        min_nr_feature_occurrences = 'min-feature-occ={}__'.format(self.min_nr_feature_occurrences) \
            if hasattr(self, 'min_nr_feature_occurrences') and self.min_nr_feature_occurrences is not None else ''
        return 'corpus-feature-vector__{train}__model={model_type}__{min_nr_feature_occurrences}{hash}.pkl'.format(
            train=self.docs_train_filename_base_wo_ext,
            model_type=self.model_type,
            min_nr_feature_occurrences=min_nr_feature_occurrences,
            hash=self.train_data_hash
        )

    @property
    def loss_type_str(self):
        loss_type_str = str(self.loss_type)
        if self.loss_type == 'plus':
            loss_type_str += str(self.doc_loss_factor)
        return loss_type_str

    @property
    def model_name(self):
        if hasattr(self, '__model_name'):
            return self.__model_name
        min_nr_feature_occurrences = 'min-feature-occ={}__'.format(self.min_nr_feature_occurrences) \
            if hasattr(self, 'min_nr_feature_occurrences') and self.min_nr_feature_occurrences is not None else ''
        self.__model_name = "{model_type}__k-rnd={k_rnd}__k-viterbi={k_viterbi}__iter={iter}__loss={loss}__train-set={data_filename}__{min_nr_feature_occurrences}{data_hash}".format(
            model_type=self.model_type,
            k_rnd=self.mira_k_random_labelings,
            k_viterbi=self.mira_k_best_viterbi_labelings,
            iter=self.mira_iterations,
            loss=self.loss_type_str,
            min_nr_feature_occurrences=min_nr_feature_occurrences,
            data_filename=self.docs_train_filename_base_wo_ext,
            data_hash=self.train_data_hash[:8]
        )
        return self.__model_name

    @property
    def model_weights_filename(self):
        return self.model_name + '.txt'

    @property
    def model_confusion_matrix_filename(self):
        return self.model_name + '__confusion_matrix.txt'

    def verify(self):
        assert(self.model_type in MODELS)
