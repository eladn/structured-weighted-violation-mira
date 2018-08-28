from tagged_word import TaggedWord


class Sentence:
    def __init__(self, sentence: str, insert_sec_labels: bool, index=None):
        self.tokens = []
        splitted_sec = sentence.split("\t")
        if insert_sec_labels:
            self.label = int(splitted_sec[0])
        else:
            self.label = None
        words = "\t".join(splitted_sec[1:])
        self.tokens = [TaggedWord(word_tag=token) for token in Sentence.split_cleaned_line(words)]
        self.index = index

        # TODO: doc!
        self.feature_attributes_idxs = None
        self.features = None

    def clone(self):
        new_sentence = Sentence('-1\ttv_NN', False, self.index)
        new_sentence.label = self.label
        new_sentence.tokens = [tagged_word.clone() for tagged_word in self.tokens]
        new_sentence.feature_attributes_idxs = self.feature_attributes_idxs
        new_sentence.features = self.features
        return new_sentence

    @staticmethod
    def split_cleaned_line(line):  # TODO check if fix with data?
        return line.strip("\n ").split(" ")

    def count_tokens(self):
        return len(self.tokens)

    def xgram(self, index, x):
        beginning = ["*" for _ in range(max(x - index - 1, 0))]
        end = [token.tag for token in self.tokens[max(index - x + 1, 0):index + 1]]
        return tuple(beginning + end)

    def trigram(self, index):
        return self.xgram(index, 2)  # only two tags before current

    def bigram(self, index):
        return self.xgram(index, 1)

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])
