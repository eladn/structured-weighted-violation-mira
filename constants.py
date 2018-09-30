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
MODEL_TYPES = (STRUCTURED_JOINT, DOCUMENT_CLASSIFIER,
               SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED)
INFER_DOCUMENT_MODEL_TYPES = (STRUCTURED_JOINT, DOCUMENT_CLASSIFIER)
INFER_SENTENCES_MODEL_TYPES = (STRUCTURED_JOINT, SENTENCE_STRUCTURED, SENTENCE_CLASSIFIER)

PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data/")
MODELS_PATH = os.path.join(PROJECT_PATH, "models/")
FEATURES_EXTRACTORS_PATH = os.path.join(PROJECT_PATH, "features_extractors/")
# BASIC_MODELS_PATH = os.path.join(MODELS_PATH, "basic/")
# ADVANCED_MODELS_PATH = os.path.join(MODELS_PATH, "advanced/")
TEST_PATH = os.path.join(PROJECT_PATH, "test/")
# BASIC_TEST_PATH = os.path.join(TEST_PATH, "basic/")
# ADVANCED_TEST_PATH = os.path.join(TEST_PATH, "advanced/")
# COMP_PATH = os.path.join(PROJECT_PATH, "comp/")
# BASIC_COMP_PATH = os.path.join(COMP_PATH, "basic/")
# ADVANCED_COMP_PATH = os.path.join(COMP_PATH, "advanced/")
EVALUATION_RESULTS_PATH = os.path.join(PROJECT_PATH, "evaluation_results/")


def make_dirs_if_not_exist(dirpaths):
    for dirpath in dirpaths:
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)


make_dirs_if_not_exist([DATA_PATH, MODELS_PATH, FEATURES_EXTRACTORS_PATH, TEST_PATH, EVALUATION_RESULTS_PATH])
