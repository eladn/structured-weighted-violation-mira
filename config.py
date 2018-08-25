from constants import *
from utils import hash_file


class Config:
    perform_train = True
    evaluate_over_train_set = True
    evaluate_over_test_set = True

    mira_k_random_labelings = 1
    mira_k_best_viterbi_labelings = 2
    mira_iterations = 5

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
    def model_name(self):
        if hasattr(self, '__model_name'):
            return self.__model_name
        self.__model_name = "{model_type}__k-rnd={k_rnd}__k-viterbi={k_viterbi}__iter={iter}__train-set={data_filename}__{data_hash}".format(
            model_type=self.model_type,
            k_rnd=self.mira_k_random_labelings,
            k_viterbi=self.mira_k_best_viterbi_labelings,
            iter=self.mira_iterations,
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
