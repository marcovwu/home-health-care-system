import logging
import os

class Logger(object):
    Initialized = False

    @staticmethod
    def initialize(path_to_runlog_file, path_to_errorlog_file):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s')
        runlog_handler = logging.FileHandler(path_to_runlog_file)
        runlog_handler.setLevel(logging.INFO)
        errorlog_handler = logging.FileHandler(path_to_errorlog_file)
        errorlog_handler.setLevel(logging.ERROR)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.terminator = "\r"

        runlog_handler.setFormatter(formatter)
        errorlog_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(runlog_handler)
        logger.addHandler(errorlog_handler)
        logger.addHandler(console_handler)
        Logger.Initialized = True

    @staticmethod
    def exception(message):
        assert Logger.Initialized, 'Logger has not been initialized'
        logging.exception(message)

    @staticmethod
    def log(level, message):
        assert Logger.Initialized, 'Logger has not been initialized'
        logging.log(level, 'ProcessID: [{}]  {}'.format(os.getpid(), message))

    @staticmethod
    def d(message):
        Logger.log(logging.DEBUG, message)

    @staticmethod
    def i(message):
        Logger.log(logging.INFO, message)

    @staticmethod
    def w(message):
        Logger.log(logging.WARNING, message)

    @staticmethod
    def e(message):
        Logger.log(logging.ERROR, message)
