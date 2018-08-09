from pprint import pprint

from add_pos import get_text_with_pos
from classes import Corpus, FeatureVector, Train, Test


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
    is_basic = False
    # is_lower = False
    do_train = False
    do_test = True
    # do_comp = False
    lambda_param = 1
    train_set_file_imdb = "spot-imdb-sent.txt"
    train_set_file_yelp = "spot-yelp13-sent.txt"
    # test_set_file = "test.wtag"
    # comp_set_file = "comp.words"
    initial_v = "l_5"

    model_name = "l_{}".format(lambda_param)

    # TODO tern on when need to add pos to the data
    # get_text_with_pos(train_set_file_imdb)

    data_with_pos = "data_with_pos"

    train_set = Corpus()
    train_set.load_file(data_with_pos, insert_document_labels=True, insert_sentence_labels=True)
    train_set.load_file(train_set_file_yelp, insert_document_labels=True, insert_sentence_labels=True)
    # test_set = Corpus(test_set_file, insert_labels=False)
    # comp_set = Corpus(comp_set_file, is_tagged=False, insert_tags=False)
    # data_exploration(train_set, test_set, comp_set)

    feature_vector = FeatureVector(train_set)
    train = Train(train_set, feature_vector, is_basic=is_basic)
    train.generate_features()
    if initial_v is not None:
        train.load_model(initial_v)
    # feature_vector.print_num_of_features()
    if do_train:
        train.evaluate_empirical_counts()
        train.evaluate_feature_vectors()

        result = train.train_model(lambda_param)
        print("Gradient average: {}".format(result))
        train.save_model(model_name)

    if do_test:
        if do_train:
            test = Test(test_set, feature_vector, is_basic=is_basic, v=train.v)
        else:
            test = Test(test_set, feature_vector, is_basic=is_basic)
            test.load_model(model_name)

        # test.evaluate_exp_v_f()
        test.viterbi()
        # for sentence in test_set.sentences:
        #     for token in sentence.tokens:
        #         print(token)
        #
        tagged_test_set = Corpus(test_set_file, is_tagged=True, insert_tags=True, lower=is_lower)
        print(test.evaluate_model(tagged_test_set))
        test.print_results_to_file(tagged_test_set, model_name, is_test=True)
        test.confusion_matrix(tagged_test_set, model_name, is_test=True)
        test.confusion_matrix_ten_max_errors(model_name, is_test=True)

    if do_comp:
        if do_train:
            comp = Test(comp_set, feature_vector, is_basic=is_basic, v=train.v)
        else:
            comp = Test(comp_set, feature_vector, is_basic=is_basic)
            comp.load_model(model_name)

        # comp.evaluate_exp_v_f()
        comp.viterbi()
        # for sentence in comp_set.sentences:
        #     for token in sentence.tokens:
        #         print(token)

        original_comp_set = Corpus(comp_set_file, is_tagged=False, insert_tags=False, lower=is_lower)
        comp.print_results_to_file(original_comp_set, model_name, is_test=False)


if __name__ == "__main__":
    main()