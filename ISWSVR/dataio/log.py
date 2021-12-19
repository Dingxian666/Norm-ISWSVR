# -*- coding: utf-8 -*-

import logging

from logging.handlers import RotatingFileHandler

__all__ = ['Log']

class Log():
    def __init__(self, log_file_path=None):
        self.logfile = log_file_path
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.fotmatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    
    def __console(self, level, message, filename, lineno):
        if self.logfile is not None:
            # create a file handle
            fh = logging.handlers.TimeRotatingFileHandler(self.logfile, when='MIDNIGHT', interval=1, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(self.fotmatter)
            self.logger.addHandler(fh)

        # create a stream handle for console output
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.fotmatter)
        self.logger.addHandler(ch)

        if filename != "" or lineno != "":
            message = ("[%s: %s] " %(filename, lineno)) + message
        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        
        # avoid repeated outputs
        self.logger.removeHandler(ch)
        if self.logfile is not None:
            self.logger.removeHandler(fh)
            fh.close()

        return
    
    def debug(self, message, filename="", lineno=""):
        self.__console('debug', message, filename, lineno)

    def info(self, message, filename="", lineno=""):
        self.__console('info', message, filename, lineno)

    def warning(self, message, filename="", lineno=""):
        self.__console('warning', message, filename, lineno)

    def error(self, message, filename="", lineno=""):
        self.__console('error', message, filename, lineno)