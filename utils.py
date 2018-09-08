import numpy as np
import sys
from typing import Union, Iterable, AnyStr
import hashlib
import time


class ProgressBar:
    def __init__(self, nr_tasks):
        self.nr_tasks = nr_tasks
        self.curr_task = 0
        self.curr_task_title = ''
        self.start_time = None

    @property
    def nr_completed(self):
        return self.curr_task - 1

    def start_next_task(self, task_title=None):
        assert(self.curr_task < self.nr_tasks)
        self.curr_task += 1
        self.print_status(task_title)
        if self.start_time is None:
            self.start_time = time.time()

    def finish(self):
        self.curr_task = None
        self.curr_task_title = ''
        self.print_status()

    def print_status(self, curr_task_title=None):

        # Title to print (making sure to override last printed title with spaces)
        last_task_title_len = len(self.curr_task_title)
        self.curr_task_title = curr_task_title if curr_task_title else ''
        curr_task_title_w_padding = str(self.curr_task_title).ljust(last_task_title_len)

        sys.stdout.write('\r')
        if self.curr_task:
            sys.stdout.write("[%-50s] Done: %d%%  (Current: %d / %d) %s" %
                             ('=' * int(np.ceil((self.nr_completed * 50) / self.nr_tasks)),
                             int(np.ceil((self.nr_completed * 100) / self.nr_tasks)),
                             self.curr_task, self.nr_tasks,
                             curr_task_title_w_padding))
        else:
            # last step (verify 100% is printed and print new line in the end):
            sys.stdout.write("[%-50s] Done: %d%% out of %d. " % ('=' * 50, 100, self.nr_tasks))
            sys.stdout.write("Total time: {0:.3f} seconds.\n".format(time.time() - self.start_time))
        sys.stdout.flush()


def print_title(msg, *args, **kwargs):
    print()
    print(msg, *args, **kwargs)
    print('-' * len(msg))


def hash_file(file_paths: Union[AnyStr, Iterable[AnyStr]], hash_type='sha1'):
    assert hash_type in {'md5', 'sha1'}

    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    hasher_func = getattr(hashlib, hash_type)()

    file_paths = file_paths if isinstance(file_paths, str) else file_paths
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                hasher_func.update(data)
    return hasher_func.hexdigest()


def get_sorted_highest_k_elements_in_matrix(matrix: np.ndarray, top_k: int):
    assert(matrix.size >= top_k)
    assert(len(matrix.shape) == 2)

    # Find (flattened) indeces of top_k elements with highest score in the input `matrix`,
    #   (out of its elements).
    # What is a "FLATTENED" index?
    #   For example, in the matrix [[8 3 1] [7 9 4]], the flattened index of the element `9` is 4.
    # Why do we use flattened indeces and not simply (i,j) indeces (this is a matrix)?
    #   It is just much simpler to perform the following tasks with a single value index.
    # Time complexity: linear! O(matrix.size)
    top_k_flattened_indexes = np.argpartition(matrix, -top_k, axis=None)[-top_k:]

    # Find the scores of the actual top_k elements.
    top_k_elements = matrix.flat[top_k_flattened_indexes]

    # Now lets sort these best found top_k elements by their score (descending order).
    # Time complexity: O(top_k * log(top_k))
    # Notice: Out of matrix.size elements we found and sorted the best top_k elements
    #   using time complexity (matrix.size) + (top_k * log(top_k)), which is optimal.
    top_k_elements_sorting_flattened_indexes = np.argsort(top_k_elements, kind='heapsort')[::-1]
    top_k_flattened_indexes_sorted = top_k_flattened_indexes[top_k_elements_sorting_flattened_indexes]
    top_k_elements_sorted = top_k_elements[top_k_elements_sorting_flattened_indexes]
    top_k_row_indexes_sorted = top_k_flattened_indexes_sorted // top_k
    top_k_col_indexes_sorted = top_k_flattened_indexes_sorted % top_k

    return top_k_row_indexes_sorted, top_k_col_indexes_sorted, top_k_elements_sorted


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


def shuffle_iter(*arrays):
    idxs = np.arange(0, len(arrays[0]))
    np.random.shuffle(idxs)
    for idx in idxs:
        yield tuple((arr[idx] for arr in arrays))


def shuffle_iterate_over_batches(*arrays, batch_size: int = None, shuffle: bool = True):
    assert (len(arrays) >= 1)

    nr_elements = len(arrays[0])
    assert (all(len(array) == nr_elements for array in arrays))
    if batch_size is None:
        batch_size = 1

    shuffled_indeces = np.arange(nr_elements)
    if shuffle:
        np.random.shuffle(shuffled_indeces)

    for batch_start_idx in range(0, nr_elements, batch_size):
        indeces = shuffled_indeces[batch_start_idx: min(nr_elements, batch_start_idx + batch_size)]
        batch = (array[indeces] if isinstance(array, np.ndarray) else list(array[idx] for idx in indeces) for array in
                 arrays)
        yield (batch_start_idx,) + tuple(batch)
