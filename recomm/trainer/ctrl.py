import argparse, os, pandas as pd

from . import env, service
from .utils import utils, flex


class Ctrl(object):
    instance = None

    PROJECT_ID = 'project_id'
    RAW_DIR = 'raw_dir'
    OVERRIDE = 'override'
    GCS = 'gcs'
    BUCKET = 'bucket'
    PARSED_FNAME = 'parsed.yaml'
    DEPLOY_FNAME = 'deploy.yaml'
    RESPONSE = 'response'
    MODEL_ID = 'model_id'
    JOB_ID = 'job_id'

    logger = env.logger('Ctrl')

    def __init__(self):
        self.service = service.Service.instance

    def prepare(self, params):
        p = pd.Series(params)
        conf = self.service.read_user_conf(p.conf_path)
        p.at['pid'] = conf[self.PROJECT_ID]
        p.at[self.RAW_DIR] = conf[self.RAW_DIR]
        p.at[self.MODEL_ID] = conf[self.MODEL_ID]
        if p.get('runtime_version') is None:
            p.at['runtime_version'] = '1.4'

        # for cloud ml engine, del environment_vars.CREDENTIALS, or credential will invoke error
        if not p.get('is_local'):
            p = self._prepare_cloud(p)
        # local training
        else:
            p = self._prepare_local(p)
        return p

    def _check_project(self, p):
        # central repo
        p.at['repo'] = utils.join(env.HQ_BUCKET, p.pid, p.model_id)
        if 'job_dir' not in p:
            p.at['job_dir'] = utils.join(p.repo, env.MODEL)
        p.at['data_dir'] = utils.join(p.repo, env.DATA)
        p.at['deploy_path'] = utils.join(p.repo, env.DATA, self.DEPLOY_FNAME)
        p.at['parsed_conf_path'] = utils.join(p.data_dir, self.PARSED_FNAME)
        return p

    def _prepare_cloud(self, p):
        self._check_project(p)
        p.at['train_file'] = utils.join(p.repo, env.DATA, env.TRAIN_FNAME)
        p.at['valid_file'] = utils.join(p.repo, env.DATA, env.VALID_FNAME)
        p.at['export_name'] = 'export_{}'.format(p.pid)
        p.at['eval_name'] = '{}'.format(p.pid)
        return p

    def _prepare_local(self, p):
        p.at['repo'] = utils.join(os.path.abspath('../repo'), p.pid, p.model_id)
        p.at['job_dir'] = utils.join(p.repo, env.MODEL)
        p.at['data_dir'] = utils.join(p.repo, env.DATA)
        p.at['deploy_path'] = utils.join(p.repo, env.DATA, self.DEPLOY_FNAME)
        p.at['parsed_conf_path'] = utils.join(p.data_dir, self.PARSED_FNAME)
        p.at['train_file'] = utils.join(p.repo, env.DATA, env.TRAIN_FNAME)
        p.at['valid_file'] = utils.join(p.repo, env.DATA, env.VALID_FNAME)
        p.at['export_name'] = 'export_{}'.format(p.pid)
        p.at['eval_name'] = '{}'.format(p.pid)
        return p

    def train(self, params):
        """do model ml-engine traning

        :param params: dict object storing user request data
        :return: json message
        """
        ret = {}
        try:
            self.logger.info('received params: {}'.format(params))
            # if run on compute engine or any vm on GCP, remove the environment_vars.CREDENTIALS environ var
            if not params.get('is_local'):
                self.logger.info('do cloud training')
                env.remove_cred_envars()
            else:
                self.logger.info('do local training')

            p = self.prepare(params)
            schema = None
            try:
                parsed_conf = flex.io(p.parsed_conf_path)
                # parsed_conf = utils.gcs_blob(p.parsed_conf_path)
                assert parsed_conf.exists(), \
                    'parsed config [{}] not found'.format(p.parsed_conf_path)

                for trf in (p.train_file, p.valid_file):
                    blob = flex.io(trf)
                    assert blob.exists(), "training file [{}] not found".format(trf)
            except Exception as e:
                raise e
                # try to gen training data
                # self.logger.info('{}: try to generate training data...'.format(p.pid))
                # schema = self.service.gen_data(p)

            if schema is None:
                self.logger.info('{}: try to unserialize {}'.format(p.pid, p.parsed_conf_path))
                schema = self.service.unser_parsed_conf(p.parsed_conf_path)

            p.at['n_batch'] = 128
            if p.get('train_steps') is None:
                # training about 10 epochs
                tr_steps = self.count_steps(schema.tr_count_, p.n_batch)
                p.at['train_steps'] = tr_steps * 3

            if p.get('eval_steps') is None:
                # training about 10 epochs
                vl_steps = self.count_steps(schema.vl_count_, p.n_batch)
                p.at['eval_steps'] = vl_steps

            p.at['dim'] = 16
            # save once per epoch, cancel this in case of saving bad model when encounter overfitting
            p.at['save_every_steps'] = None
            # local test has no job_id attr
            if p.is_local:
                p.at['job_id'] = self.find_job_id(p)

            ret[self.RESPONSE] = self.service.train(p, schema)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
            raise e
        finally:
            pass

        return ret

    def find_job_id(self, p):
        return '{}_{}_{}'.format(p.pid, p.model_id, utils.timestamp()).replace('-', '_')

    def count_steps(self, n_total, n_batch):
        return n_total // n_batch + (1 if n_total % n_batch else 0)


# mock singleton
Ctrl.instance = Ctrl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        help='execution method',
    )
    parser.add_argument(
        '--conf-path',
        help='config path in user\'s GCS',
    )
    parser.add_argument(
        '--job-dir',
        help='where to put checkpoints',
    )
    parser.add_argument(
        '--job-id',
        help='job id for training and deploy',
    )
    parser.add_argument(
        '--is-local',
        default=False,
        type=bool,
        help='whether run on local machine instead of cloud',
    )
    parser.add_argument(
        '--train-steps',
        default=1000,
        type=int,
        help='max train steps',
    )
    parser.add_argument(
        '--runtime-version',
        default='1.4',
        help='whether run on local machine instead of cloud',
    )
    args = parser.parse_args()
    params = args.__dict__
    execution = getattr(Ctrl.instance, params.get('method'))
    execution(params)
