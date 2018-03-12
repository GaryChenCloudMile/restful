import os, yaml, pandas as pd, restful_env as env

from datetime import datetime
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt

from .trainer import env, service
from .trainer.utils import utils, flex

class ViewRecomm(object):

    instance = None
    logger = env.logger('ViewRecomm')

    APP_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    APP_NAME = os.path.basename(APP_DIR)

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

    def __init__(self):
        self.service = service.Service.instance

    def prepare(self, params):
        p = pd.Series(dict(params.GET)).map(lambda e: e[0] if len(e) == 1 else e)
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

    @csrf_exempt
    def gen_data(self, params):
        self.service.gen_data(self.prepare(params))

    def train(self, params):
        from .trainer.ctrl import Ctrl

        print(Ctrl.instance)

        # hack
        # Ctrl.instance.train(params)

    def find_job_id(self, p):
        return '{}_{}_{}'.format(p.pid, p.model_id, utils.timestamp()).replace('-', '_')

    @csrf_exempt
    def train_submit(self, params):
        ret = {}
        s = datetime.now()
        p = self.prepare(params)
        try:
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
            ret['job_id'] = job_id
            ret['response'] = utils.cmd(commands)
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('{}: gen_data take time {}'.format(p.pid, datetime.now() - s))
        return ret

    @csrf_exempt
    def describe(self, params):
        ret = {}
        p = self.prepare(params)
        s = datetime.now()
        try:
            ml = self.service.find_ml()
            # deploy_info = self.service.deploy_info(p)
            name = 'projects/{}/jobs/{}'.format(env.PROJECT_ID, p.job_id)
            ret[self.RESPONSE] = ml.projects().jobs().get(name=name).execute()
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('{}: describe take time {}'.format(p.pid, datetime.now() - s))
        return ret

    @csrf_exempt
    def model_info(self, params):
        ret = {}
        p = self.prepare(params)
        try:
            res = self.service.model_info(p)
            ret['response'] = res
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            pass

        return ret

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
        data_for_model = self.service.transform(p)[0]

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

    @csrf_exempt
    def deploy(self, params):
        ret = {}
        p = self.prepare(params)
        try:
            with flex.io(p.deploy_path).as_reader('r') as f:
                deploy_conf = yaml.load(f.stream)
            res = self.service.deploy(p, deploy_conf[self.EXPORT_PATH])
            ret['response'] = res
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            pass

        return ret

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

    @csrf_exempt
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

    @csrf_exempt
    def predict(self, params):
        ret = {}
        p = self.prepare(params)
        s = datetime.now()
        try:
            parsed_conf = flex.io(p.parsed_conf_path)
            assert parsed_conf.exists(), "can't find schema cause {} not exists" \
                .format(p.parsed_conf_path)

            # TODO: hack, just receive response
            ret.update( self.service.predict(p) )
            ret[env.ERR_CDE] = '00'
        except Exception as e:
            ret[env.ERR_CDE] = '99'
            ret[env.ERR_MSG] = str(e)
            self.logger.error(e, exc_info=True)
        finally:
            self.logger.info('{}: predict take time {}'.format(p.pid, datetime.now() - s))
        return ret

    @csrf_exempt
    def load_schema(self, params):
        from .trainer.utils import flex

        p = self.prepare(params)
        p.at['raw_paths'] = self.service.find_raws(p)
        loader = flex.Loader(conf_path=p.conf_path,
                             parsed_conf_path=p.parsed_conf_path,
                             raw_paths=p.raw_paths)
        loader.check_schema()
        return loader

    def test(self, params):
        self.logger.info('test req: {}'.format(params.GET))
        return self.prepare(params).to_dict()


ViewRecomm.instance = ViewRecomm()

@api_view(http_method_names=['GET', 'POST'])
@csrf_exempt
def entry(req, method):
    logger = ViewRecomm.instance.logger
    ret = {}
    s = datetime.now()
    try:
        ret[ViewRecomm.RESPONSE] = getattr(ViewRecomm.instance, method)(req)
        ret[env.ERR_CDE] = '00'
    except Exception as e:
        ret[env.ERR_CDE] = '99'
        ret[env.ERR_MSG] = str(e)
        logger.error(e, exc_info=True)
        raise e
    finally:
        logger.info('processing {} take time {}'.format(method, datetime.now() - s))

    return Response(ret)

