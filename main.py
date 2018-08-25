from corpus import Corpus
from feature_vector import FeatureVector
from sentiment_model_trainer import SentimentModelTrainer
from sentiment_model_tester import SentimentModelTester
from config import Config
from collections import namedtuple
from utils import print_title


def data_exploration(train_set, test_set, comp_set):
    train_set_count_sentences = train_set.count_sentences()
    train_set_count_tokens = train_set.count_tokens()
    train_set_average_tokens_per_sentence = train_set_count_tokens / train_set_count_sentences

    test_set_count_sentences = test_set.count_sentences()
    test_set_count_tokens = test_set.count_tokens()
    test_set_average_tokens_per_sentence = test_set_count_tokens / test_set_count_sentences

    comp_set_count_sentences = comp_set.count_sentences()
    comp_set_count_tokens = comp_set.count_tokens()
    comp_set_average_tokens_per_sentence = comp_set_count_tokens / comp_set_count_sentences

    print("Train")
    print("---------------")
    print("{} sentences".format(train_set_count_sentences))
    print("{} tokens".format(train_set_count_tokens))
    print("{} average tokens per sentence".format(train_set_average_tokens_per_sentence))
    print()
    print("Test")
    print("---------------")
    print("{} sentences".format(test_set_count_sentences))
    print("{} tokens".format(test_set_count_tokens))
    print("{} average tokens per sentence".format(test_set_average_tokens_per_sentence))
    print()
    print("Comp")
    print("---------------")
    print("{} sentences".format(comp_set_count_sentences))
    print("{} tokens".format(comp_set_count_tokens))
    print("{} average tokens per sentence".format(comp_set_average_tokens_per_sentence))
    print()


Dataset = namedtuple("Dataset", ["train", "test"])


def load_dataset(config: Config):
    train_set = Corpus()
    train_set.load_file(config.pos_docs_train_filename, documents_label=1, insert_sentence_labels=True)
    train_set.load_file(config.neg_docs_train_filename, documents_label=-1, insert_sentence_labels=True)
    test_set = Corpus()
    test_set.load_file(config.pos_docs_test_filename, documents_label=1, insert_sentence_labels=True)
    test_set.load_file(config.neg_docs_test_filename, documents_label=-1, insert_sentence_labels=True)
    return Dataset(train=train_set, test=test_set)


def main():
    config = Config()
    dataset = load_dataset(config)
    feature_vector = FeatureVector(dataset.train)
    trainer = SentimentModelTrainer(dataset.train.clone(), feature_vector, model=config.model_type)
    trainer.generate_features()

    print('Model name: ' + config.model_name)

    if config.perform_train:
        from random import shuffle
        shuffle(trainer.corpus.documents)

        trainer.evaluate_feature_vectors()
        trainer.mira_algorithm(iterations=config.mira_iterations, k=config.mira_k,
                               best_k_labeling_method=config.mira_best_k_labeling_method)
        trainer.save_model(config.model_weights_filename)

    evaluation_datasets = []
    if config.evaluate_over_train_set:
        evaluation_datasets.append(('train', dataset.train))
    if config.evaluate_over_test_set:
        evaluation_datasets.append(('test', dataset.test))

    for evaluation_dataset_name, evaluation_dataset in evaluation_datasets:
        print_title("Model evaluation over {} set:".format(evaluation_dataset_name))
        if config.perform_train:
            tester = SentimentModelTester(evaluation_dataset.clone(), feature_vector, config.model_type, w=trainer.w)
        else:
            tester = SentimentModelTester(evaluation_dataset.clone(), feature_vector, config.model_type)
            tester.load_model(config.model_weights_filename)

        tester.inference()

        evaluation_set_ground_truth = evaluation_dataset.clone()
        print(tester.evaluate_model(evaluation_set_ground_truth))
        # tester.print_results_to_file(tagged_test_set, model_name, is_test=True)
        tester.confusion_matrix(evaluation_set_ground_truth, config.model_name)
        # tester.confusion_matrix_ten_max_errors(model_name, is_test=True)


if __name__ == "__main__":
    main()
