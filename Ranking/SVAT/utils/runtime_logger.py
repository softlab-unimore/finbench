import logging
import os

def get_logger(save_dir, model_name, run_name):
    """
    Obtain the logger of model

    :params:
        - save_dir: str    directory to save logs
        - model_name: str  model name of the current log
        - run_name: str    current log's filename
    :return:
        - logging.Logger
    """
    # initialize logger
    logger = logging.getLogger(model_name + "_model")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # create the logging file handler
        log_file = os.path.join(save_dir, run_name + ".log")
        fh = logging.FileHandler(log_file)

        # create the logging console handler
        ch = logging.StreamHandler()

        # format
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)

        # add handlers to logger object
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger