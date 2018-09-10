from corpus import Corpus
from corpus_features_vector import CorpusFeaturesExtractor
from sentiment_model_trainer import SentimentModelTrainer
from sentiment_model_tester import SentimentModelTester
from config import Config
from collections import namedtuple
from utils import print_title
from constants import FEATURES_VECTORS_PATH, SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED, \
    DOCUMENT_CLASSIFIER, STRUCTURED_JOINT

import os
import pickle
from functools import partial


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
    train_set = Corpus('train')
    train_set.load_file(config.pos_docs_train_filename, documents_label=1, insert_sentence_labels=True)
    train_set.load_file(config.neg_docs_train_filename, documents_label=-1, insert_sentence_labels=True)
    test_set = Corpus('test')
    test_set.load_file(config.pos_docs_test_filename, documents_label=1, insert_sentence_labels=True)
    test_set.load_file(config.neg_docs_test_filename, documents_label=-1, insert_sentence_labels=True)
    return Dataset(train=train_set, test=test_set)


def load_feature_vector_and_dataset(config: Config):
    if os.path.isfile(FEATURES_VECTORS_PATH + config.train_corpus_feature_vector_filename):
        with open(FEATURES_VECTORS_PATH + config.train_corpus_feature_vector_filename, 'rb') as f:
            features_vector = pickle.load(f)
        dataset = load_dataset(config)
    else:
        dataset = load_dataset(config)
        # feature_vector = FeatureVector(dataset.train)
        features_vector = CorpusFeaturesExtractor(config)
        features_vector.generate_features_from_corpus(dataset.train)
        with open(FEATURES_VECTORS_PATH + config.train_corpus_feature_vector_filename, 'wb') as f:
            pickle.dump(features_vector, f)
    features_vector.initialize_corpus_features(dataset.train)
    features_vector.initialize_corpus_features(dataset.test)
    return features_vector, dataset


# def dummy_job(config: Config, job_number: int):
    # if job_number % 10 == 0:
    #     raise ValueError('asdf')
    # import time
    # time.sleep(30 if job_number < 5 else 1)
    # return


def perform_training(config: Config, job_number: int=None):
    if job_number is not None:
        fd = os.open("./run_results/training_run_results__" + config.model_name + ".txt", os.O_RDWR | os.O_CREAT)
        os.dup2(fd, 1)
        os.dup2(fd, 2)

    print('Model name: ' + config.to_string(', '))
    features_vector, dataset = load_feature_vector_and_dataset(config)
    trainer = SentimentModelTrainer(dataset.train.clone(), features_vector, config)
    trainer.evaluate_feature_vectors()
    trainer.mira_algorithm(iterations=config.mira_iterations,
                           k_best_viterbi_labelings=config.mira_k_best_viterbi_labelings,
                           k_random_labelings=config.mira_k_random_labelings,
                           save_model_after_every_iteration=True)
    # trainer.save_model()  # already done by the argument `save_model_after_every_iteration` to the mira trainer.


def train_multiple_configurations():
    """
    Creates a processes pool, spawns all training jobs into the pool, wait for all jobs executions to finish.
    """
    from multiprocessing import Pool
    config = Config()
    NR_PROCESSES = 4
    jobs_status = {'total_nr_jobs': 0, 'nr_completed_jobs': 0, 'nr_failed_jobs': 0}
    failed_configurations = []

    def print_jobs_progress():
        print(
            '{nr_finished}/{tot_nr_jobs} jobs finished. {nr_success} completed successfully. {nr_failed} failed.'.format(
                nr_finished=jobs_status['nr_completed_jobs'] + jobs_status['nr_failed_jobs'],
                tot_nr_jobs=jobs_status['total_nr_jobs'],
                nr_success=jobs_status['nr_completed_jobs'],
                nr_failed=jobs_status['nr_failed_jobs']
            ))

    def on_success(conf: Config, value):
        jobs_status['nr_completed_jobs'] += 1
        print('========   Successfully completed job over configuration: ' + conf.to_string('  ') + '   ========')
        print_jobs_progress()

    def on_error(conf: Config, value):
        failed_configurations.append(conf)
        print('XXXXXXXX   FAILED job over configuration: ' + conf.to_string('  ') + '   XXXXXXXX')
        jobs_status['nr_failed_jobs'] += 1
        print_jobs_progress()

    process_pool = Pool(NR_PROCESSES)
    for cur_config in config.iterate_over_configurations(
            [{'mira_k_random_labelings': 0, 'mira_k_best_viterbi_labelings': 10},
             {'mira_k_random_labelings': 10, 'mira_k_best_viterbi_labelings': 0}],
            [{'model_type': [SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED], 'loss_type': 'plus'},
             {'model_type': [DOCUMENT_CLASSIFIER, STRUCTURED_JOINT], 'loss_type': ['plus', 'mult']}],
            mira_iterations=7,
            min_nr_feature_occurrences=[2, 3, 4]
    ):
        print('Spawning training job for model params: ' + cur_config.to_string(separator=', '))
        jobs_status['total_nr_jobs'] += 1
        process_pool.apply_async(
            perform_training, (cur_config, jobs_status['total_nr_jobs']),
            callback=partial(on_success, cur_config),
            error_callback=partial(on_error, cur_config))

    process_pool.close()
    process_pool.join()

    print()
    print_jobs_progress()
    if len(failed_configurations) > 0:
        print_title('FAILED configurations:')
        for conf in failed_configurations:
            print(conf)


def main():
    # train_multiple_configurations()
    # exit(0)

    config = Config()
    perform_training(config)
    exit(0)

    if os.path.isfile(FEATURES_VECTORS_PATH + config.train_corpus_feature_vector_filename):
        with open(FEATURES_VECTORS_PATH + config.train_corpus_feature_vector_filename, 'rb') as f:
            features_vector = pickle.load(f)
        dataset = load_dataset(config)
    else:
        dataset = load_dataset(config)
        # feature_vector = FeatureVector(dataset.train)
        features_vector = CorpusFeaturesExtractor(config)
        features_vector.generate_features_from_corpus(dataset.train)
        with open(FEATURES_VECTORS_PATH + config.train_corpus_feature_vector_filename, 'wb') as f:
            pickle.dump(features_vector, f)

    features_vector.initialize_corpus_features(dataset.train)
    features_vector.initialize_corpus_features(dataset.test)
    trainer = SentimentModelTrainer(dataset.train.clone(), features_vector, config)

    print('Model name: ' + config.model_name)

    if config.perform_train:
        trainer.evaluate_feature_vectors()
        trainer.mira_algorithm(iterations=config.mira_iterations,
                               k_best_viterbi_labelings=config.mira_k_best_viterbi_labelings,
                               k_random_labelings=config.mira_k_random_labelings)
        trainer.save_model(config.model_weights_filename)

    evaluation_datasets = []
    if config.evaluate_over_train_set:
        evaluation_datasets.append(('train', dataset.train))
    if config.evaluate_over_test_set:
        evaluation_datasets.append(('test', dataset.test))

    for evaluation_dataset_name, evaluation_dataset in evaluation_datasets:
        print_title("Model evaluation over {} set:".format(evaluation_dataset_name))
        if config.perform_train:
            tester = SentimentModelTester(evaluation_dataset.clone(), features_vector, config.model_type, w=trainer.w)
        else:
            tester = SentimentModelTester(evaluation_dataset.clone(), features_vector, config.model_type)
            tester.load_model(config.model_weights_filename)

        tester.inference()

        evaluation_set_ground_truth = evaluation_dataset.clone()
        print(tester.evaluate_model(evaluation_set_ground_truth))
        # tester.print_results_to_file(tagged_test_set, model_name, is_test=True)
        tester.confusion_matrix(evaluation_set_ground_truth, config.model_name)
        # tester.confusion_matrix_ten_max_errors(model_name, is_test=True)


if __name__ == "__main__":
    main()
