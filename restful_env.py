import yaml, logging, logging.config, codecs

class Logging(object):
    instance = None

    @staticmethod
    def logger(name):
        if Logging.instance is None:
            with codecs.open('logging.yaml', 'r', 'utf-8') as r:
                logging.config.dictConfig(yaml.load(r))
            Logging.instance = logging

        logger_ = Logging.instance.getLogger(name)
        return logger_

# short path of Logging.logger
def logger(name):
    return Logging.logger(name)

