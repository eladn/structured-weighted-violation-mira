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
