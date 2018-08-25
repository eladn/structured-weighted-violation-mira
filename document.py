from sentence import Sentence
from constants import DOCUMENT_LABELS


class Document:
    def __init__(self, label=None):
        assert label is None or label in DOCUMENT_LABELS
        self.sentences = []
        self.label = label

    def clone(self):
        new_doc = Document(self.label)
        new_doc.sentences = [sentence.clone() for sentence in self.sentences]
        return new_doc

    def load_sentence(self, line, insert_sec_labels):
        self.sentences.append(Sentence(line, insert_sec_labels))

    def count_sentences(self):
        return len(self.sentences)

    def count_tokens(self):
        return sum([sen.count_tokens() for sen in self.sentences])

    def y(self):
        return [self.label] + [s.label for s in self.sentences]

    def __str__(self):
        return "\n".join([str(sentence) for sentence in self.sentences])
