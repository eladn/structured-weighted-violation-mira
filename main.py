from classes import Corpus, FeatureVector, Train, Test
from config import Config
from collections import namedtuple


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
    trainer = Train(dataset.train.clone(), feature_vector, model=config.model_type)
    trainer.generate_features()

    print('Model name: ' + config.model_name)

    if config.perform_train:
        from random import shuffle
        shuffle(trainer.corpus.documents)

        trainer.evaluate_feature_vectors()
        trainer.mira_algorithm(iterations=config.mira_iterations, k=config.mira_k)
        trainer.save_model(config.model_weights_filename)

    if config.perform_test:
        if config.perform_train:
            tester = Test(dataset.test.clone(), feature_vector, config.model_type, w=trainer.w)
        else:
            tester = Test(dataset.test.clone(), feature_vector, config.model_type)
            tester.load_model(config.model_weights_filename)

        tester.inference()

        test_set_ground_truth = dataset.test.clone()
        print(tester.evaluate_model(test_set_ground_truth, config.model_type))
        # tester.print_results_to_file(tagged_test_set, model_name, is_test=True)
        tester.confusion_matrix(test_set_ground_truth, config.model_name)
        # tester.confusion_matrix_ten_max_errors(model_name, is_test=True)


if __name__ == "__main__":
    main()
