from sentiment_model_configuration import SentimentModelConfiguration
from corpus import Corpus
from dataset import load_dataset
from utils import print_title


def data_exploration(corpus: Corpus):
    nr_sentences = corpus.count_sentences()
    nr_tokens = corpus.count_tokens()
    average_tokens_per_sentence = nr_tokens / nr_sentences

    print_title("Data exploration: corpus `{}`".format(corpus.name))
    print("    {} sentences".format(nr_sentences))
    print("    {} tokens".format(nr_tokens))
    print("    {} average tokens per sentence".format(average_tokens_per_sentence))
    print()


if __name__ == "__main__":
    model_config = SentimentModelConfiguration()
    dataset = load_dataset(model_config)
    data_exploration(dataset.train)
    data_exploration(dataset.test)
