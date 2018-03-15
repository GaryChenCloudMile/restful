import os, yaml, logging, logging.config, codecs
from google.auth import environment_vars

# recommendation engine service bucket path

HQ_BUCKET = 'gs://recomm-job'

DATA = 'data'
MODEL = 'model'
LOG = 'log'

TRAIN_FNAME = 'data.tr'
VALID_FNAME = 'data.vl'

ERR_CDE = 'err_cde'
ERR_MSG = 'err_msg'

# For local use! Not on GCE
CREDENTIAL_NAME = environment_vars.CREDENTIALS
os.environ[CREDENTIAL_NAME] = 'D:/Python/notebook/restful/auth.json'
LOCAL_REPO = 'D:/Python/notebook/restful/repo'


PROJECT_ID = 'training-recommendation-engine'
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Logging(object):
    instance = None
    sd_handler = None

    @staticmethod
    def logger(name):
        if Logging.instance is None:
            with codecs.open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging

        logger_ = Logging.instance.getLogger(name)
        # # stack driver client
        # if Logging.sd_handler is None and os.environ.get(CREDENTIAL_NAME) is not None:
        #     from google.cloud import logging as sd_logging
        #     Logging.sd_handler = sd_logging.Client().get_default_handler()
        #
        # # see if exists stack driver handler
        # if Logging.sd_handler is not None:
        #     logger_.addHandler(Logging.sd_handler)
        return logger_

# short path of Logging.logger
def logger(name):
    return Logging.logger(name)

class APIClient:
    storage_client = None

def remove_cred_envars():
    if CREDENTIAL_NAME in os.environ:
        del os.environ[CREDENTIAL_NAME]

def bucket(bucket_name):
    if APIClient.storage_client is None:
        from google.cloud import storage
        APIClient.storage_client = storage.Client()
    return APIClient.storage_client.get_bucket(bucket_name)

