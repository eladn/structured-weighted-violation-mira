DEBUG = True

# PROCESSES = 8

CC = 'CC'
VBD = 'VBD'
RP = 'RP'
MD = 'MD'
IN = 'IN'
SYM = 'SYM'
RBR = 'RBR'
VB = 'VB'
EX = 'EX'
FW = 'FW'
CD = 'CD'
VBG = 'VBG'
WRB = 'WRB'
NNPS = 'NNPS'
PDT = 'PDT'
VBZ = 'VBZ'
TO = 'TO'
WDT = 'WDT'
PRP = 'PRP'
JJS = 'JJS'
VBP = 'VBP'
JJR = 'JJR'
VBN = 'VBN'
UH = 'UH'
NNS = 'NNS'
NNP = 'NNP'
NN = 'NN'
JJ = 'JJ'
WP = 'WP'
RBS = 'RBS'
RB = 'RB'
DT = 'DT'
POS = 'POS'
DOT = '.'
COMMA = ','
RRB = '-RRB-'
TILDE = '``'
HASH = '#'
PRP_DOLLAR = 'PRP$'
LRB = '-LRB-'
COLON = ':'
QUOTE = "''"
DOLLAR = '$'
WP_DOLLAR = 'WP$'

TAGS = [CC, VBD, RP, MD, IN, SYM, RBR, VB, EX, FW, CD, VBG, WRB, NNPS, PDT, VBZ, TO, WDT, PRP,
        JJS, VBP, JJR, VBN, UH, NNS, NNP, NN, JJ, WP, RBS, RB, DT, POS, DOT, COMMA, RRB, TILDE, HASH,
        PRP_DOLLAR, LRB, COLON, QUOTE, DOLLAR, WP_DOLLAR]

TEST_TAGS = ['*'] + TAGS

DATA_PATH = "./data/"
# MODELS_PATH = "./models/"
# BASIC_MODELS_PATH = MODELS_PATH + "basic/"
# ADVANCED_MODELS_PATH = MODELS_PATH + "advanced/"
# TEST_PATH = "./test/"
# BASIC_TEST_PATH = TEST_PATH + "basic/"
# ADVANCED_TEST_PATH = TEST_PATH + "advanced/"
# COMP_PATH = "./comp/"
# BASIC_COMP_PATH = COMP_PATH + "basic/"
# ADVANCED_COMP_PATH = COMP_PATH + "advanced/"