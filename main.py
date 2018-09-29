from corpus_features_extractor import CorpusFeaturesExtractor
from sentiment_model_trainer_factory import SentimentModelTrainerFactory
from sentiment_model import SentimentModel
from sentiment_model_configuration import SentimentModelConfiguration
from utils import print_title
from constants import SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED, DOCUMENT_CLASSIFIER, STRUCTURED_JOINT
from dict_itertools import union, values_union, times
from dataset import load_dataset

import sys
import os
from functools import partial


class JobExecutionParams:
    perform_train = True
    evaluate_over_train_set = True
    evaluate_over_test_set = True
    evaluate_after_every_iteration = True


# def dummy_job(config: SentimentModelConfiguration, job_execution_params: JobExecutionParams, job_number: int):
    # if job_number % 10 == 0:
    #     raise ValueError('asdf')
    # import time
    # time.sleep(30 if job_number < 5 else 1)
    # return


def train_and_eval_single_configuration(model_config: SentimentModelConfiguration,
                                        job_execution_params: JobExecutionParams,
                                        job_number: int=None):
    if job_number is not None:
        # TODO: add current time in output log filename.
        output_log_fd = os.open(
            "./run_results/training_run_results__" + model_config.model_name + ".txt", os.O_RDWR | os.O_CREAT)
        os.dup2(output_log_fd, sys.stdout.fileno())
        os.dup2(output_log_fd, sys.stderr.fileno())

    print('Model name: ' + model_config.model_name)
    dataset = load_dataset(model_config)
    features_extractor = CorpusFeaturesExtractor.load_or_create(model_config, dataset.train)
    model = None

    evaluation_datasets = []
    if job_execution_params.evaluate_over_train_set:
        evaluation_datasets.append(('train', dataset.train))
    if job_execution_params.evaluate_over_test_set:
        evaluation_datasets.append(('test', dataset.test))
        features_extractor.initialize_corpus_features(dataset.test)
    evaluation_datasets__after_every_iteration = evaluation_datasets if job_execution_params.evaluate_after_every_iteration else None

    if job_execution_params.perform_train:
        trainer = SentimentModelTrainerFactory().create_trainer(
            dataset.train.clone(), features_extractor, model_config)
        model = trainer.fit(
            save_model_after_every_iteration=True,
            datasets_to_evaluate_after_every_iteration=evaluation_datasets__after_every_iteration)
        # model.save()  # already done by the argument `save_model_after_every_iteration` to the mira trainer.

    if model is None:
        model = SentimentModel.load(model_config, features_extractor)

    evaluation_results = {}
    for evaluation_dataset_name, evaluation_dataset in evaluation_datasets:
        print_title("Model evaluation over {} set:".format(evaluation_dataset_name))

        inferred_dataset = evaluation_dataset.clone(copy_document_labels=False, copy_sentence_labels=False)
        model.inference(inferred_dataset)

        evaluation_set_ground_truth = evaluation_dataset.clone()
        evaluation_results[evaluation_dataset_name] = model.evaluate_model(inferred_dataset, evaluation_set_ground_truth)
        print(evaluation_results)
        # model.print_results_to_file(tagged_test_set, model_name, is_test=True)
        model.confusion_matrix(inferred_dataset, evaluation_set_ground_truth)
        # model.confusion_matrix_ten_max_errors(model_name, is_test=True)

    return evaluation_results


all_configurations_params = times(
    union(
        times(
            model_type=values_union(SENTENCE_CLASSIFIER, DOCUMENT_CLASSIFIER),
            training_k_best_viterbi_labelings=0,
            training_k_random_labelings=values_union(1, 5, 10, 15)
        ),
        times(
            union(
                times(training_k_random_labelings=values_union(0, 1, 2),
                      training_k_best_viterbi_labelings=values_union(1, 5, 10, 15)),
                times(training_k_random_labelings=values_union(1, 5, 10, 15),
                      training_k_best_viterbi_labelings=0)
            ),
            union(
                times(model_type=SENTENCE_STRUCTURED, loss_type='plus'),
                times(
                      union(
                          times(loss_type='plus', doc_loss_factor=values_union(0.2, 0.5, 1, 1.3, 2)),
                          times(loss_type=values_union('mult', 'max'))
                      ), model_type=STRUCTURED_JOINT
                )
            )
        )
    ),
    training_iterations=11,
    min_nr_feature_occurrences=values_union(2, 3, 4, 5),
    training_batch_size=8,
    trainer_alg='mira'  # values_union('mira', 'SWVM')
)


def train_multiple_configurations(job_execution_params: JobExecutionParams, NR_PROCESSES: int = 4):
    """
    Creates a processes pool, spawns all training jobs into the pool, wait for all jobs executions to finish.
    """
    from multiprocessing import Pool
    config = SentimentModelConfiguration()
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

    def on_success(conf: SentimentModelConfiguration, result_value):
        jobs_status['nr_completed_jobs'] += 1
        print('========   Successfully completed job over configuration: ' + conf.to_string('  ') + '   ========')
        print_jobs_progress()
        # TODO: write `result_value` to evaluation results file.

    def on_error(conf: SentimentModelConfiguration, value):
        failed_configurations.append(conf)
        print('XXXXXXXX   FAILED job over configuration: ' + conf.to_string('  ') + '   XXXXXXXX')
        jobs_status['nr_failed_jobs'] += 1
        print_jobs_progress()

    job_execution_params = JobExecutionParams()
    job_execution_params.perform_train = True
    job_execution_params.evaluate_over_train_set = False
    job_execution_params.evaluate_over_test_set = False
    job_execution_params.evaluate_after_every_iteration = False

    process_pool = Pool(NR_PROCESSES)
    for cur_config in config.iterate_over_configurations(all_configurations_params):
        print('Spawning training job for model params: ' + cur_config.to_string(separator=', '))
        jobs_status['total_nr_jobs'] += 1
        process_pool.apply_async(
            train_and_eval_single_configuration, (cur_config, job_execution_params, jobs_status['total_nr_jobs']),
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
    # Multiple configurations
    job_execution_params = JobExecutionParams()
    job_execution_params.perform_train = True
    job_execution_params.evaluate_over_train_set = True
    job_execution_params.evaluate_over_test_set = True
    job_execution_params.evaluate_after_every_iteration = False
    # train_multiple_configurations(job_execution_params)
    # exit(0)

    # Single configuration (train + optional eval)
    job_execution_params = JobExecutionParams()
    job_execution_params.perform_train = True
    job_execution_params.evaluate_over_train_set = True
    job_execution_params.evaluate_over_test_set = True
    job_execution_params.evaluate_after_every_iteration = True
    model_config = SentimentModelConfiguration()
    train_and_eval_single_configuration(model_config, job_execution_params)
    exit(0)


if __name__ == "__main__":
    main()
