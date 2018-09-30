import json
from collections import namedtuple
from constants import MODEL_TYPES, INFER_DOCUMENT_MODEL_TYPES, INFER_SENTENCES_MODEL_TYPES, \
    STRUCTURED_JOINT, DOCUMENT_CLASSIFIER


def argmax(iterable, key=None):
    max_idx, max_val = None, None
    max_val = None
    iterate_over_idx_val_pairs = iterable.items() if isinstance(iterable, dict) else enumerate(iterable)
    for idx, item in iterate_over_idx_val_pairs:
        if key is not None:
            item = key(item)
        if max_idx is None or item > max_val:
            max_idx, max_val = idx, item
    return max_idx, max_val


ModelEvaluationResult = namedtuple('ModelEvaluationResult', ['conf_dict', 'results_per_iterno', 'max_test_acc'])


def evaluation_results_json_to_table(evaluation_results: list, param_columns_printers: list, accuracies_types) -> list:
    evaluation_results_with_best_iter = list(map(
        lambda result: ModelEvaluationResult(
            conf_dict=result[0], results_per_iterno=result[1],
            max_test_acc=argmax(result[1], key=lambda iter_res: iter_res['test'][accuracies_types[0]]['accuracy'])),
        evaluation_results
    ))
    evaluation_results_with_best_iter.sort(key=lambda result: result[2][1], reverse=True)

    number_of_iterations = len(evaluation_results_with_best_iter[0].results_per_iterno)
    table_headers = []
    table_headers += list(param_printer[0] for param_printer in param_columns_printers)
    table_headers += list('iter #{iterno} {result_over} {dataset_name}'.format(
        iterno=iterno, result_over=result_over, dataset_name=dataset_name
    )
                          for iterno in range(1, number_of_iterations+1)
                          for result_over in accuracies_types
                          for dataset_name in ('train', 'test'))

    table_lines = []

    for result_model in evaluation_results_with_best_iter:
        line = list(param_printer[1](result_model.conf_dict) for param_printer in param_columns_printers)
        for iterno, result_for_iterno in result_model.results_per_iterno.items():
            for result_over in accuracies_types:
                for dataset_name in ('train', 'test'):
                    if result_over not in result_for_iterno[dataset_name]:
                        line.append(None)
                        continue
                    is_best_mark = '**' if result_model.max_test_acc[0] == iterno and dataset_name == 'test' else ''
                    accuracy = result_for_iterno[dataset_name][result_over]['accuracy']
                    accuracy_str = '{is_best_mark}{0:.5f}{is_best_mark}'.format(float(accuracy), is_best_mark=is_best_mark)
                    line.append(accuracy_str)
        table_lines.append(line)

    all_table_lines = [table_headers] + table_lines
    return all_table_lines


def evaluation_results_table_to_str(evaluation_results_table: list, column_separator=',', row_sparator='\n',
                                    value_for_missing='') -> str:
    return row_sparator.join(
        column_separator.join(value_for_missing if cell_val is None else str(cell_val) for cell_val in line)
        for line in evaluation_results_table
    )


if __name__ == '__main__':
    evaluation_results_filepath_no_ext = './evaluation_results/multiple_configurations_evaluation_results copy 5'
    evaluation_results_json_filepath = evaluation_results_filepath_no_ext + '.json'
    with open(evaluation_results_json_filepath, 'r') as evaluation_results_json_file:
        evaluation_results = json.load(evaluation_results_json_file)

    format_types = {
        'csv': {'column_separator': ',', 'row_sparator': '\n', 'value_for_missing': ''},
        'tex': {'column_separator': ' & ', 'row_sparator': '\\\\\n', 'value_for_missing': ''}
    }
    ext = 'csv'

    for model_type in MODEL_TYPES:
        output_filename = evaluation_results_filepath_no_ext + '__model_type={model_type}.{ext}'.format(
            model_type=model_type,
            ext=ext
        )

        evaluation_results_for_model = [
            model_result for model_result in evaluation_results if model_result[0]['model_type'] == model_type]

        param_columns_printers = [
            ('min-occ', lambda conf_dict: conf_dict['min_nr_feature_occurrences']),
            ('k-rnd', lambda conf_dict: conf_dict['training_k_random_labelings']),
        ]
        if model_type != DOCUMENT_CLASSIFIER:
            param_columns_printers.append(
                ('k-vit', lambda conf_dict: conf_dict['training_k_best_viterbi_labelings'])
            )
        if model_type == STRUCTURED_JOINT:
            param_columns_printers.append(
                ('loss', lambda conf_dict: (conf_dict['loss_type'] + str(
                    conf_dict.get('doc_loss_factor', ''))) if 'loss_type' in conf_dict else None)
            )
        accuracies_types = []

        if model_type in INFER_SENTENCES_MODEL_TYPES:
            accuracies_types.append('sentences')
        if model_type in INFER_DOCUMENT_MODEL_TYPES:
            accuracies_types.append('documents')

        table_as_lists = evaluation_results_json_to_table(
            evaluation_results_for_model, param_columns_printers, accuracies_types)
        table_str = evaluation_results_table_to_str(table_as_lists, **format_types[ext])

        with open(output_filename, 'w') as output_file:
            output_file.write(table_str)
