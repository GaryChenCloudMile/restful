import yaml, json, time, pandas as pd

from . import env
from . import reco_mf_dnn_est as est
from .utils import flex, utils

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

seed = 88
class Service(object):

    instance = None
    logger = env.logger('Service')

    def __init__(self):
        self.storage = None

    def read_user_conf(self, conf_path):
        with flex.io(conf_path) as f:
            return yaml.load(f.as_reader().stream)

    def unser_parsed_conf(self, parsed_conf_path):
        with flex.io(parsed_conf_path).as_reader() as f:
            return flex.Schema.unserialize(f.stream)

    def find_raws(self, p):
        return flex.io(p.raw_dir).list()

    def gen_data(self, p):
        p.at['raw_paths'] = self.find_raws(p)
        assert len(p.raw_paths), 'No training data to processing! found nothing in {}' \
            .format(p.raw_dir)

        loader = flex.Loader(conf_path=p.conf_path,
                              parsed_conf_path=p.parsed_conf_path,
                              raw_paths=p.raw_paths)

        loader.transform(p, reset=False, valid_size=.3)
        return loader.schema

    def get_model(self, p):
        with flex.io(p.parsed_conf_path).as_reader('r') as f:
            schema = flex.Schema.unserialize(f.stream)
        return est.ModelMfDNN(hparam=p, schema=schema, n_items=9125, n_genres=20)

    def train(self, p, schema):
        self.logger.info('received params: {}'.format(p.to_dict() if isinstance(p, pd.Series) else p))
        model = est.ModelMfDNN(hparam=p, schema=schema, n_items=9125, n_genres=20)
        train_input = model.input_fn([p.train_file], n_epoch=1, n_batch=p.n_batch)
        valid_input = model.input_fn([p.valid_file], n_epoch=1, n_batch=p.n_batch, shuffle=False)

        # try to build local export directory avoid error
        if p.is_local:
            flex.io(utils.join(p.job_dir, 'export', p.export_name)).mkdirs()
            # os.makedirs(utils.join(p.job_dir, 'export', p.export_name))

        model.fit(train_input, valid_input, reset=True)

        # export deploy info
        deploy_info = {}
        deploy_info['job_id'] = p.job_id
        deploy_info['model_name'] = '{}_{}'.format(p.pid, p.model_id).replace('-', '_')
        deploy_info['export_path'] = model.exporter.export_result.decode().replace('\\', '/')
        with flex.io(p.deploy_path).as_writer('w') as f:
            yaml.dump(deploy_info, f.stream)

        return model

    def init_model(self, p):
        self.logger.info('received params: {}'.format(p))
        with flex.io(p.parsed_conf_path).as_reader('r') as f:
            schema = flex.Schema.unserialize(f.stream)
        return est.ModelMfDNN(hparam=p, schema=schema, n_items=9125, n_genres=20)

    def find_ml(self):
        credentials = GoogleCredentials.get_application_default()
        return discovery.build('ml', 'v1', credentials=credentials)

    def deploy_info(self, p):
        with flex.io(p.deploy_path).as_reader() as f:
            return yaml.load(f.stream)

    def model_info(self, p):
        ml = self.find_ml()

        deploy_info = self.deploy_info(p)
        version = deploy_info.get('version')

        model_name = deploy_info.get('model_name')
        model_uri = 'projects/{}/models/{}/versions/{}'.format(env.PROJECT_ID, model_name, version)
        return ml.projects().models().versions().get(name=model_uri).execute()

    def deploy(self, p, export_path):
        ml = self.find_ml()
        deploy_info = self.deploy_info(p)
        model_name = deploy_info.get('model_name')

        self.logger.info('try to create model [{}] ...'.format(model_name))
        self.create_model_rsc(ml, model_name)

        self.logger.info('try to clean old version ...')
        self.clear_model_rsc(ml, p, model_name)

        model_uri = 'projects/{}/models/{}'.format(env.PROJECT_ID, model_name)
        version = 'v{}'.format(utils.timestamp())
        # ver_uri = model_uri + '/versions/{}'.format(version)
        res = ml.projects().models().versions().create(
            parent=model_uri,
            body={
                'name': version,
                'description': '[{}] recommendation model'.format(p.pid),
                # 'isDefault': True,
                'deploymentUri': export_path.decode() if isinstance(export_path, bytes) else export_path,
                'runtimeVersion': p.runtime_version
            }
        ).execute()
        # write version information
        deploy_info = self.deploy_info(p)
        deploy_info['version'] = version
        self.logger.info('[{}] write deploy version name: {}'.format(model_name, version))
        with flex.io(p.deploy_path).as_reader('w') as f:
            yaml.dump(deploy_info, f.stream)

        return res

    def create_model_rsc(self, ml, model_name):
        proj_uri = 'projects/{}'.format(env.PROJECT_ID)
        try:
            ml.projects().models().create(
                parent=proj_uri, body={'name': model_name, 'onlinePredictionLogging': True}
            ).execute()
            # wait for create
            self.wait(3)
        except Exception as e:
            self.logger.warn(e)

    def clear_model_rsc(self, ml, p, model_name):
        model_rsc = 'projects/{}/models/{}'.format(env.PROJECT_ID, model_name)
        vdict = ml.projects().models().versions().list(parent=model_rsc).execute()

        def delete(m):
            self.logger.info('delete model version [{name}]'.format(**m))
            ml.projects().models().versions().delete(name=m.get('name')).execute()

        default_model = None
        if len(vdict) and 'versions' in vdict:
            if len(vdict['versions']) == 1:
                delete(vdict['versions'][0])
            else:
                for m in vdict['versions']:
                    if m['state'] != 'READY':
                        continue
                        # raise Exception('status of {name} != READY, actual = [{state}]'.format(**m))

                    # can't delete default version
                    if m.get('isDefault'):
                        default_model = m
                        continue
                    delete(m)
                # if default_model is not None:
                #     delete(default_model)
        return self

    def wait(self, sec):
        time.sleep(sec)

    def transform(self, p):
        # self.logger.info('predict.params: {}'.format(p.to_dict()))
        loader = flex.Loader(p.conf_path, p.parsed_conf_path)
        # transform data to model recognizable
        return loader.trans_json(p.json_data)

    def predict(self, p):
        data_for_model = self.transform(p)
        deploy_info = self.deploy_info(p)

        # gcloud predict and persistent to file for debug
        # with flex.io('./data.json').as_writer('w') as f:
        #     for r in data_for_model:
        #         f.stream.write(json.dumps(r) + '\n')
        # with flex.io('./data.json').as_writer('w') as f:
        #     json.dump(data_for_model, f.stream)

        # python restful api predict
        model_uri = 'projects/{}/models/{}'.format(env.PROJECT_ID, deploy_info.get('model_name'))
        ml = self.find_ml()
        # return data type must be in records mode
        result = ml.projects().predict(name=model_uri, body={'instances': data_for_model}).execute()
        return [rec.get('predictions')[0] for rec in result.get('predictions')]

        # commands = '''
        #     gcloud ml-engine predict --model {}  \
        #            --version {} \
        #            --json-instances {}
        # '''.strip().format(deploy_info.get('model_name'),
        #                    deploy_info.get('version'),
        #                    './data.json')
        # return {'data_for_model': data_for_model, 'response': utils.cmd(commands)}

Service.instance = Service()