from constants import *
from utils import hash_file


class Config:
    perform_train = False
    perform_test = True

    mira_k = 10
    mira_iterations = 5

    docs_train_filename_base_wo_ext = "test-0.2p"
    docs_test_filename_base_wo_ext = "test-0.2p"

    # model_type = DOCUMENT_CLASSIFIER
    # model_type = SENTENCE_CLASSIFIER
    model_type = STRUCTURED_JOINT
    # model_type = SENTENCE_STRUCTURED

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
        self.__model_name = "{model_type}__k{k}__iter{iter}__{data_filename}__{data_hash}".format(
            model_type=self.model_type, k=self.mira_k, iter=self.mira_iterations,
            data_filename=self.docs_train_filename_base_wo_ext, data_hash=self.train_data_hash[:8]
        )
        return self.__model_name

    @property
    def model_weights_filename(self):
        return self.model_name + '.txt'

    @property
    def model_confusion_matrix_filename(self):
        return self.model_name + '__confusion_matrix.txt'
