import logging

from src.util.progress_msg import ProgressMsg


class Logger:
    def __init__(self, max_iter, log_dir=None):
        # init progress message class
        self.p_msg = ProgressMsg(max_iter)
        self.log_dir = log_dir

        # set configs
        self.set_cfg()
        
        # init logging
        logging.basicConfig(
            format=self.logging_format,
            level=self.logging_lvl,
            handlers=self.logging_handler
            )

    def set_cfg(self):
        self.logging_lvl = logging.INFO
        self.include_time = True
        self.logging_mode = 'w' # 'a': add, 'w': over write

        # logging format
        if self.include_time:
            self.logging_format = '[%(asctime)s] %(message)s'
        else:
            self.logging_format = '%(message)s'

        # logging handler
        self.logging_handler = []
        self.logging_handler.append(logging.StreamHandler())
        if self.log_dir is not None: self.logging_handler.append(logging.FileHandler(filename=self.log_dir, mode=self.logging_mode))

    def debug(self, txt):
        logging.debug(txt)
    def info(self, txt):
        logging.info(txt)
    def warning(self, txt):
        logging.warning(txt)
    def error(self, txt):
        logging.error(txt)

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
