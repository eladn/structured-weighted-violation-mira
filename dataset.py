from sentiment_model_configuration import SentimentModelConfiguration
from corpus import Corpus

from collections import namedtuple


Dataset = namedtuple("Dataset", ["train", "test"])


def load_dataset(config: SentimentModelConfiguration):
    train_set = Corpus('train')
    train_set.load_file(config.pos_docs_train_filename, documents_label=1, insert_sentence_labels=True)
    train_set.load_file(config.neg_docs_train_filename, documents_label=-1, insert_sentence_labels=True)
    test_set = Corpus('test')
    test_set.load_file(config.pos_docs_test_filename, documents_label=1, insert_sentence_labels=True)
    test_set.load_file(config.neg_docs_test_filename, documents_label=-1, insert_sentence_labels=True)
    return Dataset(train=train_set, test=test_set)
