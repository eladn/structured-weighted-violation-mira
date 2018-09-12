
"""
    Input example:
    multi_dicts_product_iterator(
        times(
            union(
                times(
                    union(
                        times(mira_k_random_labelings = 0, mira_k_best_viterbi_labelings = 10),
                        times(mira_k_random_labelings = 10, mira_k_best_viterbi_labelings = 0)
                    ),
                    union(
                        times(model_type = values_union('SENTENCE_CLASSIFIER', 'SENTENCE_STRUCTURED'), loss_type = 'plus'),
                        times(model_type = values_union('STRUCTURED_JOINT'), loss_type = values_union('plus', 'mult', 'max'))
                    )
                ),
                times(mira_k_random_labelings = 10, model_type = 'DOCUMENT_CLASSIFIER')
            ),
            mira_iterations = 7,
            min_nr_feature_occurrences = values_union(2, 3, 4)
        )
    )
"""


class values_union:
    def __init__(self, *args):
        if any(isinstance(arg, values_union) for arg in args):
            raise ValueError('Positional arguments to `values_union` mustn\'t  be of type `values_union`.')
        if any(isinstance(arg, union) for arg in args):
            raise ValueError('Positional arguments to `values_union` mustn\'t  be of type `union`.')
        if any(isinstance(arg, times) for arg in args):
            raise ValueError('Positional arguments to `values_union` mustn\'t  be of type `times`.')
        self.values = args

    def __iter__(self):
        for values in self.values:
            yield values


class union:
    def __init__(self, *args):
        self.union = args
        if not all(isinstance(arg, times) for arg in args):
            raise ValueError('All positional arguments to `union` must be of type `times`.')

    def __iter__(self):
        from itertools import chain
        for item in chain(*self.union):
            yield item


class times:
    def __init__(self, *args, **kwargs):
        self.unions = args
        if not all((isinstance(arg, union) for arg in args)):
            raise ValueError('All positional arguments to `times` must be of type `union`.')
        if not all((not isinstance(arg, times) for arg in kwargs.values())):
            raise ValueError('All keyword arguments to `times` must not be of type `times`.')
        self.dct_items_with_single_value = {key: value for key, value in kwargs.items() if not isinstance(value, values_union)}
        self.dct_items_with_multiple_values__ordered = [(key, value) for key, value in kwargs.items() if isinstance(value, values_union)]
        self.dct_items_with_multiple_values__ordered__only_lst = [vu.values for _, vu in self.dct_items_with_multiple_values__ordered]

    def __iter__(self):
        from itertools import product
        for values_sequence in product(*self.dct_items_with_multiple_values__ordered__only_lst):
            s = {key: value for value, (key, _) in zip(values_sequence, self.dct_items_with_multiple_values__ordered)}
            for dicts_from_union in product(*self.unions):
                cur_yielded_dct = {**s, **self.dct_items_with_single_value}
                for dct in dicts_from_union:
                    cur_yielded_dct.update(dct)
                yield cur_yielded_dct
