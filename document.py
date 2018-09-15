from sentence import Sentence
from constants import DOCUMENT_LABELS


class Document:
    def __init__(self, label=None, index=None):
        assert label is None or label in DOCUMENT_LABELS
        self.sentences = []
        self.label = label
        self.index = index

    def clone(self, copy_document_label: bool = True, copy_sentence_labels: bool = True):
        label = self.label if copy_document_label else None
        new_doc = Document(label, self.index)
        new_doc.sentences = [sentence.clone(copy_sentence_labels) for sentence in self.sentences]
        return new_doc

    def load_sentence(self, line, insert_sec_labels):
        self.sentences.append(Sentence(line, insert_sec_labels, len(self.sentences)))

    def count_sentences(self):
        return len(self.sentences)

    def count_tokens(self):
        return sum([sen.count_tokens() for sen in self.sentences])

    def y(self):
        return [self.label] + [s.label for s in self.sentences]

    def __str__(self):
        return "\n".join([str(sentence) for sentence in self.sentences])

    def assign_labeling(self, labeling: list):
        from itertools import islice
        assert(len(labeling) == 1 + self.count_sentences())
        self.label = labeling[0]
        for lbl, sentence in zip(islice(labeling, 1, None), self.sentences):
            sentence.label = lbl
