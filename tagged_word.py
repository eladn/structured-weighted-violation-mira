
class TaggedWord:
    def __init__(self, word=None, tag=None, word_tag=None):
        """
        This class supports:
        - word_tag (for example: word_tag = "the_DT")
        - word, tag (for example: word = "the", tag = "DT")
        - word (for example: word = "the") - untagged
        """

        if word_tag is not None:
            word, tag = TaggedWord.split_word_tag(word_tag)

        self.word = word
        self.tag = tag

    def clone(self, copy_tag: bool = True):
        tag = self.tag if copy_tag else None
        return TaggedWord(self.word, tag)

    @staticmethod
    def split_word_tag(word_tag):
        return word_tag.split("_")  # TODO check if fix with the data?

    @staticmethod
    def split_word_from_word_tag(word_tag):
        return TaggedWord.split_word_tag(word_tag)[0]

    @staticmethod
    def split_tag_from_word_tag(word_tag):
        return TaggedWord.split_word_tag(word_tag)[1]

    def split_tag_from_taggedword(self):
        return self.tag

    def is_tag(self):
        return bool(self.tag)

    def print_to_file(self):
        return "{}_{}".format(self.word, self.tag)

    def __eq__(self, tagged_word):
        return self.word == tagged_word.word and self.tag == tagged_word.tag

    def __str__(self):
        return "{word}:{tag}".format(word=self.word, tag=self.tag)
