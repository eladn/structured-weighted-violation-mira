from corpus_features_extractor import CorpusFeaturesExtractor
from sentiment_model_trainer_factory import SentimentModelTrainerFactory
from sentiment_model import SentimentModel
from sentiment_model_configuration import SentimentModelConfiguration
from utils import print_title
from constants import SENTENCE_CLASSIFIER, SENTENCE_STRUCTURED, DOCUMENT_CLASSIFIER, STRUCTURED_JOINT, \
        EVALUATION_RESULTS_PATH
from dict_itertools import union, values_union, times
from dataset import load_dataset

import sys
import os
from functools import partial
import json


class JobExecutionParams:
    perform_train = True
    use_saved_models_for_training = True
    evaluate_over_train_set = True
    evaluate_over_test_set = True
    evaluate_after_every_iteration = True

    @property
    def job_type_str(self):
        qual = []
        if self.perform_train:
            qual.append('train')
        if self.evaluate_over_train_set or self.evaluate_over_test_set:
            qual.append('eval')
        return '-and-'.join(qual)


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
        output_log_dirname = "run_results_{job_type}".format(
            job_type=job_execution_params.job_type_str
        )
        output_log_dirpath = os.path.join(os.getcwd(), output_log_dirname)
        if not os.path.isdir(output_log_dirpath):
            os.mkdir(output_log_dirpath)
        output_log_filename = "{job_type}_run_results__{model_name}.log".format(
            job_type=job_execution_params.job_type_str, model_name=model_config.model_name)
        output_log_filepath = os.path.join(output_log_dirpath, output_log_filename)

        # output_log_fd = os.open(
        #     output_log_filepath, os.O_RDWR | os.O_CREAT)
        # os.dup2(output_log_fd, sys.stdout.fileno())
        # os.dup2(output_log_fd, sys.stderr.fileno())
        output_log_fd = open(output_log_filepath, 'w+')
        sys.stdout = output_log_fd
        sys.stderr = output_log_fd

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
            datasets_to_evaluate_after_every_iteration=evaluation_datasets__after_every_iteration,
            use_previous_iterations_if_exists=job_execution_params.use_saved_models_for_training)
        # model.save()  # already done by the argument `save_model_after_every_iteration` to the mira trainer.

    evaluation_for_iter_numbers = [model_config.training_iterations]
    if job_execution_params.evaluate_after_every_iteration:
        evaluation_for_iter_numbers = list(range(1, model_config.training_iterations+1))

    # TODO: if also training, use intermediate evaluation results.
    eval_model_config = model_config.clone()
    evaluation_results_per_iter = {}
    for iter_nr in evaluation_for_iter_numbers:
        eval_model_config.training_iterations = iter_nr
        model = SentimentModel.load(eval_model_config, features_extractor)
        if model is None:
            continue

        evaluation_results_for_cur_iter = {}
        for evaluation_dataset_name, evaluation_dataset in evaluation_datasets:
            print_title("Model evaluation over {} set:".format(evaluation_dataset_name))

            inferred_dataset = evaluation_dataset.clone(copy_document_labels=False, copy_sentence_labels=False)
            model.inference(inferred_dataset)

            evaluation_set_ground_truth = evaluation_dataset.clone()
            evaluation_results_for_cur_iter[evaluation_dataset_name] = model.evaluate_model(inferred_dataset, evaluation_set_ground_truth)
            print('iter #{}: {}'.format(iter_nr, evaluation_results_for_cur_iter))
            # model.print_results_to_file(tagged_test_set, model_name, is_test=True)
            model.confusion_matrix(inferred_dataset, evaluation_set_ground_truth)
            # model.confusion_matrix_ten_max_errors(model_name, is_test=True)

        evaluation_results_per_iter[iter_nr] = evaluation_results_for_cur_iter

    return evaluation_results_per_iter


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


def train_and_eval_multiple_configurations(job_execution_params: JobExecutionParams, NR_PROCESSES: int = 4):
    """
    Creates a processes pool, spawns all training jobs into the pool, wait for all jobs executions to finish.
    """
    from multiprocessing import Pool
    config = SentimentModelConfiguration()
    jobs_status = {'total_nr_jobs': 0, 'nr_completed_jobs': 0, 'nr_failed_jobs': 0}
    failed_configurations = []
    evaluation_results = []
    evaluation_results_json_filepath = os.path.join(EVALUATION_RESULTS_PATH, 'multiple_configurations_evaluation_results.json')

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
        print()
        if result_value:
            evaluation_results.append((conf.to_dict(), result_value))
            with open(evaluation_results_json_filepath, 'w') as evaluation_results_output_file:
                json.dump(evaluation_results, evaluation_results_output_file)

    def on_error(conf: SentimentModelConfiguration, value):
        failed_configurations.append(conf)
        print('XXXXXXXX   FAILED job over configuration: ' + conf.to_string('  ') + '   XXXXXXXX')
        jobs_status['nr_failed_jobs'] += 1
        print_jobs_progress()

    process_pool = Pool(NR_PROCESSES)
    for cur_config in config.iterate_over_configurations(all_configurations_params):
        print('Spawning {job_type} job for model params: {cnf}'.format(
            job_type=job_execution_params.job_type_str,
            cnf=cur_config.to_string(separator=', '))
        )
        jobs_status['total_nr_jobs'] += 1
        process_pool.apply_async(
            train_and_eval_single_configuration, (cur_config, job_execution_params, jobs_status['total_nr_jobs']),
            callback=partial(on_success, cur_config),
            error_callback=partial(on_error, cur_config))

    process_pool.close()
    process_pool.join()

    with open(evaluation_results_json_filepath, 'w') as evaluation_results_output_file:
        json.dump(evaluation_results, evaluation_results_output_file)

    print()
    print_jobs_progress()
    if len(failed_configurations) > 0:
        print_title('FAILED configurations:')
        for conf in failed_configurations:
            print(conf)


def main():
    # Multiple configurations
    job_execution_params = JobExecutionParams()
    job_execution_params.perform_train = False
    job_execution_params.evaluate_over_train_set = True
    job_execution_params.evaluate_over_test_set = True
    job_execution_params.evaluate_after_every_iteration = True
    # train_and_eval_multiple_configurations(job_execution_params)
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
