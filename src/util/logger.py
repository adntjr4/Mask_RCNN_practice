import logging

from src.util.progress_msg import ProgressMsg


class Logger:
    def __init__(self, max_iter:tuple, log_dir:str=None, log_lvl:str='info', include_time:bool=False):
        '''
        Args:
            max_iter (tuple)
            log_dir (str)
            log_lvl (str) : 'debug', 'info'
            include_time (bool)
        '''
        assert log_lvl in ['info', 'debug']

        # init progress message class
        self.p_msg = ProgressMsg(max_iter)
        self.log_dir = log_dir
        self.logging_lvl = logging.INFO if log_lvl=='info' else logging.DEBUG
        self.include_time = include_time

        # set configs
        self.set_cfg()
        
        # init logging
        logging.basicConfig(
            format=self.logging_format,
            level=self.logging_lvl,
            handlers=self.logging_handler
            )

    def set_cfg(self):
        self.logging_mode = 'w' # 'a': add, 'w': over write

        # logging format
        if self.include_time:
            self.logging_format = '[%(asctime)s] %(message)s'
        else:
            self.logging_format = '%(message)s'

        # logging handler
        self.logging_handler = [logging.StreamHandler()]
        if self.log_dir is not None: self.logging_handler.append(logging.FileHandler(filename=self.log_dir, mode=self.logging_mode))

    def debug(self, txt):
        self.p_msg.line_reset()
        logging.debug(txt)

    def info(self, txt):
        self.p_msg.line_reset()
        logging.info(txt)

# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
# currently not using
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    import time
    lg = Logger((10, 10), 'log.log')

    lg.p_msg.start((5, 0))

    for i in range(5, 10):
        for j in range(10):
            lg.info(f'{i}, {j}')
            lg.p_msg.print_prog_msg((i, j))
            time.sleep(1)
