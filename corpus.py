from document import Document
from constants import DOCUMENT_LABELS, DATA_PATH

import re
import numpy as np


class Corpus:
    def __init__(self, name: str=None):
        self.documents = []
        self.name = '' if name is None else str(name)

    def clone(self, copy_document_labels: bool = True, copy_sentence_labels: bool = True):
        print('Cloning corpus ...', end='')
        new_corpus_name = self.name + ' [Clone]' if self.name else ''
        new_corpus = Corpus(new_corpus_name)
        new_corpus.documents = [doc.clone(copy_document_labels, copy_sentence_labels) for doc in self.documents]
        print(' Done.')
        return new_corpus

    def load_file(self, file_name, documents_label: int, insert_sentence_labels: bool):
        assert documents_label in DOCUMENT_LABELS
        pattern = re.compile("^\d \d{7}$")
        with open(DATA_PATH + file_name) as f:
            document = Document(documents_label)
            for i, line in enumerate(f):
                if line == "\n":
                    self.documents.append(document)
                    document = Document(documents_label)
                elif pattern.match(line):
                    continue
                else:
                    document.load_sentence(line, insert_sentence_labels)
        for document_idx, document in enumerate(self.documents):
            document.index = document_idx

    def count_documents(self):
        return np.size(self.documents)

    def count_sentences(self):
        return sum([doc.count_sentences() for doc in self.documents])

    def count_tokens(self):
        return sum([doc.count_tokens() for doc in self.documents])

    def __str__(self):
        return "\n".join([str(document) for document in self.documents])

    def __iter__(self):
        for document_idx, document in enumerate(self.documents):
            for sentence_idx, sentence in enumerate(document.sentences):
                yield document, sentence

