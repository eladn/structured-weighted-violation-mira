import numpy as np
import sys


class ProgressBar:
    def __init__(self, nr_tasks):
        self.nr_tasks = nr_tasks
        self.curr_task = 0
        self.curr_task_title = ''

    @property
    def nr_completed(self):
        return self.curr_task - 1

    def start_next_task(self, task_title=None):
        assert(self.curr_task < self.nr_tasks)
        self.curr_task += 1
        self.print_status(task_title)

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
            sys.stdout.write("[%-50s] Done: %d%% out of %d\n" % ('=' * 50, 100, self.nr_tasks))
        sys.stdout.flush()


def print_title(msg):
    print(msg)
    print('-' * len(msg))
