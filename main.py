from classes import Corpus, FeatureVector, Train, Test
from constants import STRUCTURED_JOINT, SENTENCE_CLASSIFIER, DOCUMENT_CLASSIFIER, SENTENCE_STRUCTURED


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


def main():
    do_train = False
    do_test = True

    k = 10
    mira_iterations = 5

    positive_docs_file = "pos-50.txt"
    negative_docs_file = "neg-50.txt"

    # model = DOCUMENT_CLASSIFIER
    # model = SENTENCE_CLASSIFIER
    # model = STRUCTURED_JOINNT
    model = SENTENCE_STRUCTURED
    model_name = "{}-k{}-iter{}.txt".format(model, k, mira_iterations)

    train_set = Corpus()
    train_set.load_file(positive_docs_file, documents_label=1, insert_sentence_labels=True)
    train_set.load_file(negative_docs_file, documents_label=-1, insert_sentence_labels=True)
    # test_set = Corpus(test_set_file, insert_labels=False)
    # data_exploration(train_set, test_set, comp_set)

    feature_vector = FeatureVector(train_set)
    train = Train(train_set, feature_vector, model=model)
    train.generate_features()
    # if initial_v is not None:
    #     train.load_model(initial_v)
    # feature_vector.print_num_of_features()
    if do_train:
        train.evaluate_feature_vectors()
        result = train.mira_algorithm(iterations=mira_iterations, k=k)
        print("Gradient average: {}".format(result))
        train.save_model(model_name)

    if do_test:
        if do_train:
            test = Test(train_set, feature_vector, model, w=train.w)
        else:
            test = Test(train_set, feature_vector, model)
            test.load_model(model_name)

        # test.evaluate_exp_v_f()
        test.inference()
        # for sentence in test_set.sentences:
        #     for token in sentence.tokens:
        #         print(token)
        #
        test_set = Corpus()
        test_set.load_file(positive_docs_file, documents_label=1, insert_sentence_labels=True)
        test_set.load_file(negative_docs_file, documents_label=-1, insert_sentence_labels=True)
        print(test.evaluate_model(test_set, model))
        # test.print_results_to_file(tagged_test_set, model_name, is_test=True)
        test.confusion_matrix(test_set, model_name)
        # test.confusion_matrix_ten_max_errors(model_name, is_test=True)


if __name__ == "__main__":
    main()