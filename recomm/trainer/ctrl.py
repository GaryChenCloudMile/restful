import argparse, os, pandas as pd, yaml, json

from datetime import datetime

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
    EXPORT_PATH = 'export_path'

    logger = env.logger('Ctrl')

    def __init__(self):
        self.service = service.Service.instance

    def prepare(self, params):
        p = pd.Series(params) if not isinstance(params, pd.Series) else params
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

    def gen_data(self, params):
        self.service.gen_data(self.prepare(params))

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

            return self.service.train(p, schema)
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise e
        finally:
            pass

    def find_job_id(self, p):
        return '{}_{}_{}'.format(p.pid, p.model_id, utils.timestamp()).replace('-', '_')

    def train_submit(self, params):
        p = self.prepare(params)
        job_id = self.find_job_id(p)

        commands = """
            cd {} && \
            gcloud ml-engine jobs submit training {} \
                --job-dir {} \
                --module-name trainer.ctrl \
                --package-path trainer \
                --region asia-east1 \
                --config config.yaml \
                --runtime-version 1.4 \
                -- \
                --train-steps 1000 \
                --method train \
                --conf-path {} \
                --job-id {}
        """.strip().format(env.PROJECT_PATH, job_id, p.job_dir, p.conf_path, job_id)

        # authpath = utils.join(ctx, 'auth.json')
        # svc = discovery.build('ml', 'v1', credentials=GoogleCredentials.from_stream(authpath))
        # resp = svc.projects().jobs()\
        #           .create(parent='projects/{}'.format(project),
        #                   body={
        #                       'jobId': 'recomm_movielens_16',
        #                       'trainingInput': {
        #                           'Module': 'trainer.ctrl',
        #                           'region': 'asia-east1',
        #                           'jobDir': 'gs://recomm-job/foo/model',
        #                           'packageUris': 'recomm-job/foo/model/packages/{}/package-0.0.0.tar.gz'.format(utils.timestamp()),
        #                           'runtimeVersion': '1.4',
        #                           'pythonVersion': '3.5'
        #                       }
        #                   })\
        #           .execute()
        ret = {}
        ret['job_id'] = job_id
        ret['sumbmit_response'] = utils.cmd(commands)
        return ret

    def describe(self, params):
        p = self.prepare(params)
        ml = self.service.find_ml()
        # deploy_info = self.service.deploy_info(p)
        name = 'projects/{}/jobs/{}'.format(env.PROJECT_ID, p.job_id)
        return ml.projects().jobs().get(name=name).execute()

    def model_info(self, params):
        p = self.prepare(params)
        return self.service.model_info(p)

    def get_model(self, params):
        p = self.prepare(params)
        return self.service.get_model(p), p

    def est_predict(self, params):
        import tensorflow as tf

        p = self.prepare(params)
        p.at['n_batch'] = 128
        p.at['dim'] = 16
        model = self.service.get_model(p)
        estimator = model.create_est()
        data_for_model = self.service.transform(p)

        # suprisely, I need to init datasets in a "local closure function" put into estimator's predict params
        # or I will get a different graph error
        def input_fn():
            dataset = tf.data.Dataset.from_tensors(data_for_model)
            dataset = dataset.repeat(1)
            dataset = dataset.make_one_shot_iterator().get_next()
            return dataset, None

        for e in estimator.predict(input_fn=input_fn):
            print(e)

        return self

    def deploy(self, params):
        p = self.prepare(params)
        with flex.io(p.deploy_path).as_reader('r') as f:
            deploy_conf = yaml.load(f.stream)
        return self.service.deploy(p, deploy_conf[self.EXPORT_PATH])

    def train_local_submit(self, params):
        """not working in windows envs, gcloud bind python version must be 2.7

        :param params:
        :return:
        """
        ret = {}
        p = self.prepare(params)
        try:
            self.logger.info(utils.cmd("gcloud components list"))
            commands = """
                cd {} && \
                gcloud ml-engine local train \
                    --job-dir {} \
                    --module-name trainer.ctrl \
                    --package-path trainer \
                    -- \
                    --method train \
                    --is-local true \
                    --conf-path {}
            """.strip() \
                .format(env.PROJECT_PATH, p.job_dir, p.parsed_conf_path)
            # .format(env.PROJECT_PATH, '../repo/foo/model', '../repo/foo/data/{}'.format(self.PARSED_FNAME))
            ret['response'] = utils.cmd(commands)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            pass

        return ret

    def transform(self, params):
        ret = {}
        p = self.prepare(params)
        s = datetime.now()
        try:
            ret['response'] = self.service.transform(p)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('{}: transform take time {}'.format(p.pid, datetime.now() - s))
        return ret

    def online_predict(self, params):
        p = self.prepare(params)
        parsed_conf = flex.io(p.parsed_conf_path)
        assert parsed_conf.exists(), "can't find schema cause {} not exists" \
            .format(p.parsed_conf_path)

        # restful api may pass in string json
        if isinstance(p.json_data, str):
            p.at['json_data'] = json.loads(p.json_data)
        return self.service.predict(p)

    def load_schema(self, params):
        p = self.prepare(params)
        p.at['raw_paths'] = self.service.find_raws(p)
        loader = flex.Loader(conf_path=p.conf_path,
                             parsed_conf_path=p.parsed_conf_path,
                             raw_paths=p.raw_paths)
        loader.check_schema()
        return loader

    def test(self, params):
        self.logger.info('test req: {}'.format(params))
        p = self.prepare(params)
        print(p)

        p = self.prepare(params)
        p.at['n_batch'] = 128
        p.at['dim'] = 16
        model = self.service.get_model(p)

        return model


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
