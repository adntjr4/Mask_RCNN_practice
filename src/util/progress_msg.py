import time
import datetime
import sys

class ProgressMsg():
    def __init__(self, max_iter):
        '''
        Args:
            max_iter : (max_epoch, max_data_length, ...)
        '''
        self.max_iter = max_iter
        self.start_time = time.time()
        self.progress_time = self.start_time

    def start(self, start_iter):

        assert len(self.max_iter) == len(start_iter), 'start_iter should have same length with max variable.'

        self.start_iter = start_iter
        self.start_time = time.time()
        self.progress_time = self.start_time

    def print_prog_msg(self, current_iter):
        if time.time() - self.progress_time < 1:
            return
        self.progress_time = time.time()

        assert len(self.max_iter) == len(current_iter), 'current should have same length with max variable.'

        for i in range(len(self.max_iter)):
            assert current_iter[i] <= self.max_iter[i], 'current value should be less than max value.'

        start_per = 0
        for i in reversed(range(len(self.max_iter))):
            start_per += self.start_iter[i]
            start_per /= self.max_iter[i]
        start_per *= 100

        pg_per = 0
        for i in reversed(range(len(self.max_iter))):
            pg_per += current_iter[i]
            pg_per /= self.max_iter[i]
        pg_per *= 100

        pg_per = (pg_per-start_per) / (100-start_per) * 100

        if pg_per != 0:
            elapsed = time.time() - self.start_time
            total = 100*elapsed/pg_per
            remain = total - elapsed
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            remain_str = str(datetime.timedelta(seconds=int(remain)))
            total_str = str(datetime.timedelta(seconds=int(total)))
            txt = '>>> progress : %.2f%%, elapsed: %s, remaining: %s, total: %s \t\t\t\t\t' % (pg_per, elapsed_str, remain_str, total_str)
        else:
            elapsed = time.time() - self.start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            txt = '>>> progress : %.2f%%, elapsed: %s, remaining: %s, total: %s \t\t\t\t\t' % (pg_per, elapsed_str, 'INF', 'INF')

        print(txt, end='\r')

        return txt.replace('\t', '')

    def line_reset(self):
        sys.stdout.write("\033[K")

    def print_finish_msg(self):
        self.line_reset()
        total = time.time() - self.start_time
        total_str = str(datetime.timedelta(seconds=int(total)))
        txt = 'Finish >>> (total elapsed time : %s) \t\t\t\t\t' % total_str
        print(txt)
        return txt.replace('\t', '')

if __name__ == '__main__':
    pp = ProgressMsg((10,10))
    ss = (0, 0)

    pp.start(ss)
    
    print(pp.__class__.__name__)

    for i in range(0, 10):
        for j in range(10):
            pp.print_prog_msg((i, j))
            time.sleep(1)
        