import numpy as np, pandas as pd, yaml, codecs, os, tensorflow as tf, json, re

from .. import env
from . import utils
from io import StringIO, BytesIO
from google.cloud.storage.blob import Blob
from datetime import datetime

from collections import OrderedDict

# pd.set_option('display.width', 100)

class Schema(object):
    # config attrs
    COLUMNS = "columns"
    USER = 'user'
    ITEM = 'item'
    LABEL = 'label'

    ID = 'id'
    DATE_FORMAT = 'date_format'
    M_DTYPE = 'm_dtype'
    # continous, datetime
    CONT = 'cont'
    DATETIME = 'datetime'
    # categorical
    CATG = 'catg'
    DEFAULT = 'default'
    IS_MULTI = 'is_multi'
    # N_UNIQUE = 'n_unique'
    VOCABS = 'vocabs'
    VOCABS_PATH = 'vocabs_path'
    SEP = 'sep'
    AUX = 'aux'
    M_DTYPE_ARY = [CONT, CATG, DATETIME]
    TYPE = 'type'
    COL_STATE = 'col_state'

    COL_ATTR = [ID, M_DTYPE, DATE_FORMAT, DEFAULT, IS_MULTI,
                SEP, VOCABS, VOCABS_PATH, AUX, TYPE,
                COL_STATE]
    logger = env.logger('Schema')

    def __init__(self, conf_path, raw_paths:list):
        """Schema configs

        :param conf_path: path for config file columns specs configurations
        :param raw_paths: multiple raw training csv files
        """
        self.conf_path = conf_path
        # TODO: wait for fetch GCS training data to parse
        self.raw_paths = raw_paths

        # self.parsed_conf_path_ = '{}.parsed'.format(conf_path)
        self.count_ = 0
        self.conf_ = None
        self.df_conf_ = None
        self.col_states_ = None
        self.tr_count_ = None
        self.vl_count_ = None

    @property
    def raw_cols(self):
        return self.df_conf_.index.tolist()

    @property
    def cols(self):
        return list(self.col_states_.keys())

    @property
    def user_cols(self):
        return self.df_conf_.query("{} == '{}'".format(Schema.TYPE, Schema.USER)).id.tolist()

    @property
    def item_cols(self):
        return self.df_conf_.query("{} == '{}'".format(Schema.TYPE, Schema.ITEM)).id.tolist()

    @property
    def features(self):
        label = self.label[0]
        return list(e for e in self.cols if e != label)

    @property
    def label(self):
        return self.df_conf_.query("{} == '{}'".format(Schema.TYPE, Schema.LABEL)).id.tolist()

    @property
    def tensor_type(self):
        dtypes = OrderedDict()
        df_conf = self.df_conf_.query("{}.notnull()".format(Schema.TYPE))
        for _, r in df_conf.iterrows():
            dtypes[r[Schema.ID]] = tf.int32 if r[Schema.M_DTYPE] == Schema.CATG else tf.float32
        return dtypes

    def init(self):
        return self.extract(self.read_conf()).check().fit()

    def read_conf(self):
        with io(self.conf_path) as f:
            return f.read()

    def extract(self, conf):
        self.conf_ = yaml.load(conf)
        for k in (Schema.USER, Schema.ITEM, Schema.LABEL, Schema.COLUMNS):
            assert k in self.conf_, 'config requires {} attrs, actual {}'\
                .format([Schema.USER, Schema.ITEM, Schema.LABEL, Schema.COLUMNS], list(self.conf_.keys()))

        cols = []
        for r in self.conf_[Schema.COLUMNS]:
            cols.append(r)

        self.df_conf_ = (pd.DataFrame(columns=Schema.COL_ATTR, data=cols)
                           .reset_index(drop=True))
        self.df_conf_ = self.df_conf_.where(self.df_conf_.notnull(), None)
        # self.df_conf_.set_index('id', drop=False)

        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['item']), 'type'] = 'item'
        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['user']), 'type'] = 'user'
        self.df_conf_.loc[self.df_conf_.id.isin(self.conf_['label']), 'type'] = 'label'

        # is_multi, aux columns default False
        for col in (Schema.IS_MULTI, Schema.AUX):
            self.df_conf_[col].where(self.df_conf_[col].notnull(), False, inplace=True)

        self.df_conf_ = self.df_conf_.set_index(Schema.ID, drop=False)
        return self

    def check(self):
        """check user input columns configs

        :return: self
        """
        # check if basic attr exists
        df_conf = self.df_conf_.query("type.notnull()")
        base = df_conf.query("{}.isnull() or {}.isnull()".format(Schema.ID, Schema.M_DTYPE))
        assert len(base) == 0, 'require {} attrs, check following settings:\n{}'\
            .format([Schema.ID, Schema.M_DTYPE], base)

        # check if user, item, label columns in self.conf_['columns']
        for k in (Schema.USER, Schema.ITEM, Schema.LABEL):
            unknowns = set(self.conf_[k]) - set(df_conf[Schema.ID])
            assert len(unknowns) == 0, '[{}]: {} not found in columns id settings'.format(k, list(unknowns))

        # check if dtype in [str, float, int, datetime]
        # err_dtypes = conf[~conf[Schema.DTYPE].isin(Schema.DTYPE_ARY)]
        # assert len(err_dtypes) == 0, 'require value of {} in {}, check following:\n{}'\
        #     .format(Schema.DTYPE, Schema.DTYPE_ARY, err_dtypes)

        # check if m_dtype in [cont, catg, datetime]
        err_m_dtypes = df_conf[~df_conf[Schema.M_DTYPE].isin(Schema.M_DTYPE_ARY)]
        assert len(err_m_dtypes) == 0, 'require value of {} in {}, check following:\n{}' \
            .format(Schema.M_DTYPE, Schema.M_DTYPE_ARY, err_m_dtypes)

        # check catg columns
        self.check_catg(df_conf)

        # datetime column requires date_format settings
        dt_no_format = df_conf.query("{} == '{}' and {}.isnull()"\
                              .format(Schema.M_DTYPE, Schema.DATETIME, Schema.DATE_FORMAT))
        assert len(dt_no_format) == 0, '{} column expect {} attr, check following:\n{}' \
            .format(Schema.DATETIME, Schema.DATE_FORMAT, dt_no_format)

        # inference whether to add auxiliary col, like sequence, weighted columns
        return self.check_aux()

    def check_aux(self):
        # need_aux = self.df_conf_.query("{}.notnull() and {} == '{}' and {} == True" \
        #                         .format(Schema.TYPE, Schema.M_DTYPE, Schema.CATG, Schema.IS_MULTI))
        # # print('need_aux', need_aux)
        # for id, r in need_aux.iterrows():
        #     # COL_ATTR = [ID, M_DTYPE, DATE_FORMAT, DEFAULT, IS_MULTI,
        #     #             SEP, VOCABS, VOCABS_PATH, AUX, TYPE,
        #     #             COL_STATE]
        #     self.df_conf_ = self.df_conf_.append({Schema.ID: '{}_len'.format(id),
        #                                           Schema.M_DTYPE: Schema.CONT,
        #                                           Schema.IS_MULTI: False,
        #                                           Schema.AUX: True,
        #                                           Schema.TYPE: Schema.AUX}, ignore_index=True)
        # self.df_conf_ = self.df_conf_.where(self.df_conf_.notnull(), None).set_index('id', drop=False)
        return self

    def check_catg(self, df_conf):
        """check catg columns

        :param df_conf: config in pandas
        :return: None
        """
        catg = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.CATG))

        # null_n_unique = catg.query("{} <= 0".format(Schema.N_UNIQUE))
        # assert len(null_n_unique) == 0, 'categorical column expect number of vocabs [{}] value > 0, ' \
        #                                 'check following:\n{}'.format(Schema.N_UNIQUE, null_n_unique)

        multi_no_sep = catg.query("{} == True and {}.isnull()".format(Schema.IS_MULTI, Schema.SEP))
        assert len(multi_no_sep) == 0, 'multivalent column expect {} attr, check following:\n{}' \
            .format(Schema.SEP, multi_no_sep)

    def raw_dtype(self, df_conf):
        # str dtype for all catg + datetime columns, float dtype for all cont columns
        catg = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.CATG))
        dt = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.DATETIME))
        cont = df_conf.query("{} == '{}'".format(Schema.M_DTYPE, Schema.CONT))
        dt_catg = pd.concat([dt, catg], ignore_index=True)

        dtype = dict(zip(dt_catg[Schema.ID], ['str'] * len(dt_catg)))
        dtype.update(dict(zip(cont[Schema.ID], ['float'] * len(cont))))
        return dtype

    def fit(self):
        """fetch columns states in training data

        :return:
        """
        from datetime import  datetime

        df_conf = self.df_conf_.query("{}.notnull()".format(Schema.TYPE))
        dtype = self.raw_dtype(df_conf)
        col_states = OrderedDict()
        # './merged_movielens.csv'
        for fpath in self.raw_paths:
            blob = io(fpath)
            if not blob.exists():
                self.logger.info("{} doesn't exists".format(fpath))
                continue

            # bio = BytesIO(blob.download_as_string())
            with blob.as_reader() as f:
                for chunk in pd.read_csv(f.stream, names=self.raw_cols, chunksize=20000, dtype=dtype):
                    chunk = chunk.where(pd.notnull(chunk), None)

                    # loop all valid columns except label
                    for _, r in df_conf.iterrows():
                        val, m_dtype, name, col_type = None, r[Schema.M_DTYPE], r[Schema.ID], r[Schema.TYPE]
                        is_aux = r[Schema.AUX]

                        if is_aux: continue

                        if col_type == Schema.LABEL:
                            if name not in col_states:
                                col_states[name] = utils.CatgMapper(name, allow_null=False).init_check()
                            # null value is not allowed in label column
                            assert not chunk[name].hasnans, 'null value detected in label column! filename {}' \
                                .format(fpath)
                        else:
                            # categorical column
                            if m_dtype == Schema.CATG:
                                is_multi, sep = r[Schema.IS_MULTI], r[Schema.SEP]
                                vocabs, vocabs_path = r[Schema.VOCABS], r[Schema.VOCABS_PATH]
                                if name not in col_states:
                                    if is_multi:
                                        col_states[name] = utils.CatgMapper(name,
                                                                            is_multi=is_multi,
                                                                            sep=sep,
                                                                            vocabs=vocabs,
                                                                            vocabs_path=vocabs_path)\
                                                                .init_check()
                                    else:
                                        col_states[name] = utils.CatgMapper(name,
                                                                            vocabs=vocabs,
                                                                            vocabs_path=vocabs_path)\
                                                                .init_check()
                            # numeric column
                            elif m_dtype == Schema.CONT:
                                if name not in col_states:
                                    col_states[name] = utils.NumericMapper(name, default=r[Schema.DEFAULT])\
                                                            .init_check()
                            # datetime column: transform to numeric
                            elif m_dtype == Schema.DATETIME:
                                dt_fmt = r[Schema.DATE_FORMAT]
                                if name not in col_states:
                                    col_states[name] = utils.DatetimeMapper(name, dt_fmt, default=r[Schema.DEFAULT])\
                                                            .init_check()
                        if m_dtype == Schema.CATG:
                            # if freeze_ == True, that means user provided vocabs informations,
                            # no need to fit anymore
                            if not col_states[name].freeze_:
                                col_states[name].partial_fit(chunk[name].values)
                        else:
                            col_states[name].partial_fit(chunk[name].values)

                    # count data size
                    self.count_ += len(chunk)

        self.col_states_ = col_states
        valid_cond = self.df_conf_[Schema.TYPE].notnull() & (self.df_conf_[Schema.AUX] == False)
        # serialize parsed column states
        def ser(id):
            sio = StringIO()
            col_states[id].serialize(sio)
            return sio.getvalue()

        self.df_conf_.loc[valid_cond, 'col_state'] = \
            self.df_conf_.loc[valid_cond, Schema.ID].map(ser)

        # output type: catg+multi to str
        return self

    def serialize(self, fp):
        """

        :param fp:
        :return:
        """
        return yaml.dump({
            'conf_path': self.conf_path,
            # 'parsed_conf_path_': self.parsed_conf_path_,
            'raw_paths': self.raw_paths,
            'count_': self.count_,
            'tr_count_': self.tr_count_,
            'vl_count_': self.vl_count_,
            'conf_': self.conf_,
            'df_conf_': self.df_conf_.to_dict(orient='records'),
        }, fp)

    @staticmethod
    def unserialize(fp):
        """Create Schema instance from config resource

        :param fp:
        :return:
        """
        info = yaml.load(fp)
        this = Schema(info['conf_path'], info['raw_paths'])
        for k, attr in info.items():
            setattr(this, k, attr)

        this.df_conf_ = pd.DataFrame(this.df_conf_, columns=Schema.COL_ATTR)
        this.df_conf_ = this.df_conf_.set_index(Schema.ID, drop=False)
        # specific class for each type
        ser_maps = {Schema.CATG: utils.CatgMapper,
                    Schema.CONT: utils.NumericMapper,
                    Schema.DATETIME: utils.DatetimeMapper}

        this.col_states_ = OrderedDict()
        for _, r in this.df_conf_.query('{}.notnull() and {} == False'\
                                 .format(Schema.TYPE, Schema.AUX)).iterrows():
            this.col_states_[r[Schema.ID]] = \
                utils.BaseMapper.unserialize(ser_maps[r[Schema.M_DTYPE]], r[Schema.COL_STATE])
        return this

class Loader(object):
    logger = env.logger('Loader')

    def __init__(self, conf_path, parsed_conf_path, raw_paths:list=None):
        """

        :param conf_path:
        :param parsed_conf_path:
        :param raw_paths:
        """
        self.conf_path = conf_path
        self.parsed_conf_path = parsed_conf_path
        self.raw_paths = raw_paths
        self.schema = None

    def check_schema(self):
        # init schema
        if self.schema is None:
            # 1. try unserialize
            parsed_conf = io(self.parsed_conf_path)
            if parsed_conf.exists():
                # if os.path.isfile(self.parsed_conf_path):
                self.logger.info('try to unserialize from {}'.format(self.parsed_conf_path))
                with parsed_conf.as_reader():
                    self.schema = Schema.unserialize(parsed_conf.stream)
            # 2. if parsed_conf_path not exists, try re-parse raw config file (conf_path supplied by user)
            else:
                self.logger.info('try to parse {} (user supplied) ...'.format(self.conf_path))
                self.schema = Schema(self.conf_path, self.raw_paths)
                self.schema.init()
        return self

    def transform(self, params, chunksize=20000, reset=False, valid_size=None):
        # reset: remove parsed config file and rebuild
        if reset:
            self.schema = None
            io(self.parsed_conf_path).rm()

        self.check_schema()

        self.logger.info('try to transform {} ... '.format(params.raw_paths))
        df_conf = self.schema.df_conf_
        dtype = self.schema.raw_dtype(df_conf)
        columns = df_conf[Schema.ID].values
        col_states = self.schema.col_states_
        pos = 0

        # make tmp dir
        # tmp_ctx = '/tmp/{}'.format(params.pid)
        # os.makedirs(tmp_ctx, exist_ok=True)
        # trw = codecs.open('{}/data.tr.{}'.format(tmp_ctx, utils.timestamp()), 'w')
        trw = io(params.train_file).as_writer(mode='w')
        vlw, rand_seq = None, None
        if valid_size:
            rand_seq = np.random.random(size=self.schema.count_)
            # vlw = codecs.open('{}/data.te.{}'.format(tmp_ctx, utils.timestamp()), 'a')
            vlw = io(params.valid_file).as_writer(mode='w')
            self.schema.tr_count_ = int(sum(rand_seq > valid_size))
            self.schema.vl_count_ = int(sum(rand_seq <= valid_size))

        # serialize to specific path
        f = io(self.parsed_conf_path)
        if not f.exists():
            with f.as_writer('w'):
                self.schema.serialize(f.stream)

        try:
            s = datetime.now()
            for fpath in params.raw_paths:
                # download bytes from gcs, assume the file size is small
                # because we suggest split large file into multi chunks in a directory
                # bio = BytesIO(utils.gcs_blob(fpath).download_as_string())
                stream = io(fpath).as_reader().stream
                for chunk in pd.read_csv(stream, names=columns, chunksize=chunksize, dtype=dtype):
                    chunk = chunk.where(pd.notnull(chunk), None)[self.schema.cols]
                    for colname, col in chunk.iteritems():
                        # multivalent categorical columns
                        if df_conf.loc[colname, Schema.M_DTYPE] == Schema.CATG and \
                                df_conf.loc[colname, Schema.IS_MULTI]:
                            val = pd.Series(col_states[colname].transform(col))
                            # because of persist to csv, transfer int array to string splitted by comma
                            chunk[colname] = val.map(lambda ary: ','.join(map(str, ary))).tolist()
                        # univalent columns
                        else:
                            chunk[colname] = list(col_states[colname].transform(col))

                    kws = {'index': False, 'header': None}
                    if not valid_size:
                        chunk.to_csv(trw.stream, **kws)
                    else:
                        end_pos = pos + len(chunk)
                        rand_batch = rand_seq[pos:end_pos]
                        # 1 - (valid_size * 100)% training data, (valid_size * 100)% testing data
                        tr_chunk, vl_chunk = chunk[rand_batch > valid_size], chunk[rand_batch <= valid_size]
                        tr_chunk.to_csv(trw.stream, **kws)
                        vl_chunk.to_csv(vlw.stream, **kws)
                        pos = end_pos

                e = datetime.now()
                self.logger.info('[{}]: process take time {}'.format(fpath, e - s))
                s = e
        finally:
            _ = trw.close() if trw is not None else None
            _ = vlw.close() if vlw is not None else None

        return self

    def trans_json(self, data):
        self.check_schema()

        self.logger.info('try to restful transform ... ')
        # json path to read
        if isinstance(data, str):
            with io(data).as_reader('r') as f:
                data = json.load(f.stream)
        else:
            data = data.copy()

        ret = {}
        col_states = self.schema.col_states_
        pad = tf.keras.preprocessing.sequence.pad_sequences
        # loop parsed feature columns, cause maybe there're some noise columns in data
        for colname in self.schema.features:
            col_meta = col_states[colname]
            val = col_meta.transform(data[colname])
            # if multi columns, create sequence columns
            if hasattr(col_meta, 'is_multi') and col_meta.is_multi:
                lens = list(map(len, val))
                ret[colname] = pad(val, padding="post", maxlen=max(lens)).tolist()
                ret[colname + '_len'] = np.array(lens).tolist()
            else:
                ret[colname] = val.tolist()

        # TODO hack for debug online prediction
        # cols = ['query_movie_ids', 'genres', 'avg_rating', 'year', 'candidate_movie_id',
        #         'query_movie_ids_len', 'genres_len']
        # ret = OrderedDict(zip(cols, [ret.get(c) for c in cols]))
        #
        # ret = pd.Series(ret)
        # l = len(ret['genres'])
        # ret.loc['query_movie_ids'] = np.repeat(ret['query_movie_ids'], l, 0).tolist()
        # ret.loc['query_movie_ids_len'] = np.repeat(ret['query_movie_ids_len'], l, 0).tolist()
        # ret = pd.DataFrame(data=ret.to_dict(), columns=ret.keys())
        # ret = [dict(zip(r.keys().tolist(), r.values.tolist())) for _, r in ret.iterrows()]
        return ret

def io(path):
    return FlexIO(path)

class FlexIO(object):

    logger = env.logger('FlexIO')

    def __init__(self, path):
        m = re.search('(?i)^gs://', path)
        self.is_local = True if m is None else False
        self.path = path
        self.mode = None
        self.encoding = None
        self.placeholder = None
        self.stream = None
        self.check()

    def check(self):
        if not self.is_local:
            self.placeholder = utils.gcs_blob(self.path)
        return self

    def exists(self):
        return os.path.exists(self.path) if self.is_local else self.placeholder.exists()

    def rm(self):
        if self.is_local:
            import shutil
            _ = utils.rm_quiet(self.path) if os.path.isfile(self.path) else shutil.rmtree(self.path, ignore_errors=True)
        else:
            _ = [utils.gcs_rm_quiet(e) for e in io(self.path).list()]
        return self

    def mkdirs(self):
        if self.is_local:
            os.makedirs(self.path, exist_ok=True)
        else:
            env.logger('FlexIO').info('mkdirs not support by GCS!')
        return self

    def list(self):
        if self.is_local:
            if os.path.isdir(self.path):
                return [utils.join(root, f).replace('\\', '/') for root, ds, fs in os.walk(self.path) for f in fs]
            return []
        else:
            return list(map(lambda blob: 'gs://' + utils.join(blob.bucket.name, blob.name), utils.gcs_list(self.path)))

    def read(self, mode='rb', encoding=None):
        return self.as_reader(mode=mode, encoding=encoding).stream.read() if self.is_local \
            else self.as_reader(mode=mode, encoding=encoding).stream.getvalue()

    def write(self, data, mode='wb', encoding=None):
        self.as_writer(mode=mode, encoding=encoding).stream.write(data)
        return self

    def as_reader(self, mode='rb', encoding=None):
        self._file_handler('read', mode=mode, encoding=encoding)
        return self

    def as_writer(self, mode='wb', encoding=None):
        self._file_handler('write', mode=mode, encoding=encoding)
        return self

    def _file_handler(self, tpe, mode, encoding):
        if self.stream is None:
            self.mode = mode
            if self.is_local:
                if not self.exists():
                    dirpath = os.path.dirname( os.path.abspath(self.path) )
                    # self.logger.info('try to mkdirs [{}]'.format(dirpath))
                    os.makedirs(dirpath, exist_ok=True)
                self.stream = codecs.open(self.path, mode=mode, encoding=encoding)
            else:
                is_binary = 'b' in mode
                self.stream = BytesIO() if is_binary else StringIO()
                if tpe == 'read':
                    # GCS only accept bytes download ...
                    if is_binary:
                        self.placeholder.download_to_file(self.stream)
                    else:
                        stream_ = BytesIO()
                        self.placeholder.download_to_file(stream_)
                        self.stream.write(stream_.getvalue().decode())
                    self.stream.seek(0)
            self.stream.f = self
        return self.stream

    def close(self):
        if self.stream is not None:
            if not self.is_local and 'w' in self.mode:
                self.logger.info('upload to [{}]'.format(self.path))
                self.stream.seek(0)
                # buff = self.stream
                # if isinstance(self.stream, StringIO):
                #     buff = BytesIO()
                #     buff.write(self.stream.getvalue().encode())
                self.placeholder.upload_from_string(self.stream.getvalue())
            self.stream.close()
            self.stream = None
            self.mode = None
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()