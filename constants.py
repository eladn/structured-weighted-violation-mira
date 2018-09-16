import os

DEBUG = True

DOCUMENT_LABELS = [-1, 1]
SENTENCE_LABELS = [-1, 1]
NR_SENTENCE_LABELS = len(SENTENCE_LABELS)
NR_DOCUMENT_LABELS = len(DOCUMENT_LABELS)

STRUCTURED_JOINT = "structured-joint"
DOCUMENT_CLASSIFIER = "document-classifier"
SENTENCE_CLASSIFIER = "sentence-classifier"
SENTENCE_STRUCTURED = "sentence-structured"
MODELS = (STRUCTURED_JOINT, DOCUMENT_CLASSIFIER,
          SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED)

PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = PATH + "/data/"
MODELS_PATH = PATH + "/models/"
FEATURES_EXTRACTORS_PATH = PATH + "/features_extractors/"
# BASIC_MODELS_PATH = MODELS_PATH + "basic/"
# ADVANCED_MODELS_PATH = MODELS_PATH + "advanced/"
TEST_PATH = PATH + "/test/"
# BASIC_TEST_PATH = TEST_PATH + "basic/"
# ADVANCED_TEST_PATH = TEST_PATH + "advanced/"
# COMP_PATH = PATH + "/comp/"
# BASIC_COMP_PATH = COMP_PATH + "basic/"
# ADVANCED_COMP_PATH = COMP_PATH + "advanced/"
