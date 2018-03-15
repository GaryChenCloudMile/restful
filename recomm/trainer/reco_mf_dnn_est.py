import tensorflow as tf, os, traceback, numpy as np
from . import env
from .utils import flex, utils

from collections import OrderedDict

seed = 88

class ModelMfDNN(object):
    logger = env.logger('ModelMfDNN')

    def __init__(self,
                 hparam=None,
                 schema=None):
        self.schema = schema
        self.hparam = hparam
        self.exporter = None
        if hparam is not None:
            self.model_dir = hparam.job_dir

    def graphing(self, features, labels, mode):
        self.logger.info('mode = {}'.format(mode))
        self.features, self.labels = features, labels
        for name, tensor in self.features.items():
            setattr(self, name, tensor)

        col_states = self.schema.col_states_
        emb_dim = np.array([8, 16, 32, 64])

        # both user and item sides need to concat features, so it's needed to count the dimension
        # for dim of bias
        u_dim, i_dim = 0, 0
        with tf.variable_scope("init") as scope:
            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")

            # embedding size in [8, 16, 32, 64], take the logarithmic of unique number
            # of the categorical column,
            # if np.log(n_unique of column) = 15, then embedding size = 16
            with tf.variable_scope("embedding") as scope:
                for side, cols in (('user', self.schema.user_cols), ('item', self.schema.item_cols)):
                    for col in cols:
                        dim = 1
                        if isinstance(col_states[col], utils.CatgMapper):
                            n_unique = col_states[col].n_unique
                            n_emb = emb_dim[ np.digitize([np.log(col_states[col].n_unique)], emb_dim) ][0]
                            w_name = 'w_{}'.format(col)
                            setattr(self, w_name,
                                tf.Variable(emb_init_fn(shape=[n_unique, n_emb]), name=w_name))
                            dim = n_emb

                        if side == 'user':
                            u_dim += dim
                        else:
                            i_dim += dim

                    if side == 'user':
                        self.b_user = tf.Variable(emb_init_fn(shape=[u_dim]), name="b_user")
                    else:
                        self.b_item = tf.Variable(emb_init_fn(shape=[i_dim]), name="b_item")

        def do_concat(side):
            """handle the process of from inputs to hidden layers

            :param side: user or item
            :return: concat result
            """
            # adjust the order of user or item columns
            concat = []
            # setattr(self, '{}_concat'.format(side), concat)
            cols = self.schema.conf_[side]
            for col in cols:
                state = col_states[col]
                if isinstance(state, utils.CatgMapper):
                    w_name = 'w_{}'.format(col)
                    emb_name = 'emb_{}'.format(col)
                    after_emb = tf.nn.embedding_lookup(
                        getattr(self, w_name), getattr(self, col))

                    # categorical column need pooling => sqrtn pooling
                    if state.is_multi:
                        seq_tensor = getattr(self, '{}_len'.format(col))
                        mask = tf.expand_dims(
                            tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(seq_tensor)), 1), -1)
                        after_emb = tf.reduce_sum(after_emb * mask, 1)
                    setattr(self, emb_name, after_emb)
                    concat.append(after_emb)
                else:
                    concat.append(getattr(self, col)[:, tf.newaxis])

            # assign concat to model for debug
            setattr(self, '{}_concat'.format(side), concat)
            return tf.concat(concat, 1) if len(concat) > 1 else concat[0]

        with tf.variable_scope("user_encoding") as scope:
            self.query = do_concat('user')
            self.b_query = tf.matmul(self.query, self.b_user[:, tf.newaxis])
            self.query = tf.layers.dense(self.query, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.query = tf.layers.dense(self.query, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.query = tf.layers.dense(self.query, 32, kernel_initializer=init_fn, activation=tf.nn.selu)

        # encode [item embedding + item metadata]
        with tf.variable_scope("item_encoding") as scope:
            self.item = do_concat('item')
            self.b_item = tf.matmul(self.item, self.b_item[:, tf.newaxis])
            self.item = tf.layers.dense(self.item, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.item = tf.layers.dense(self.item, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.item = tf.layers.dense(self.item, 32, kernel_initializer=init_fn, activation=tf.nn.selu)

        # elements wise dot of user and item embedding
        with tf.variable_scope("gmf") as scope:
            self.gmf = tf.reduce_sum(self.query * self.item, 1, keepdims=True)
            self.gmf = tf.add(self.gmf, self.b_global)
            self.gmf = tf.add(self.gmf, self.b_query)
            self.gmf = tf.add(self.gmf, self.b_item)
            self.infer = tf.reshape(tf.nn.sigmoid(self.gmf, name="infer"), [-1])

            # one query for all items, for predict speed
            self.pred = tf.matmul(self.query, tf.transpose(self.item)) + \
                        tf.reshape(self.b_item, (1, -1)) + \
                        self.b_query + \
                        self.b_global
            self.pred = tf.nn.sigmoid(self.pred, name='pred')

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'query': self.query,
                    'item': self.item,
                    'predictions': self.infer
                })
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=self.pred,
                                              export_outputs=export_outputs)

        with tf.variable_scope("loss") as scope:
            # self.alter_rating = tf.to_float(self.label >= 4)[:, tf.newaxis]
            self.ans = tf.to_float(self.labels)[:, tf.newaxis]
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ans, logits=self.gmf))
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope("metrics") as scope:
            self.auc = tf.metrics.auc(tf.cast(self.labels, tf.bool),
                                      tf.reshape(self.infer, [-1]))
            # tf.summary.scalar('auc', self.auc)

        self.train_op = None
        self.global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("train"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = tf.train.AdamOptimizer().minimize(self.loss, self.global_step)
                    # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        # self.merge = tf.summary.merge_all()

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops={'auc': self.auc},
            evaluation_hooks=[])

    def input_fn(self, filenames, n_batch=128, n_epoch=None, shuffle=True):
        df_conf = self.schema.df_conf_.query('{}.notnull()'.format(self.schema.TYPE))
        defaults = []
        for _, r in df_conf.iterrows():
            if r[self.schema.M_DTYPE] == self.schema.CATG:
                defaults.append([''] if r[self.schema.IS_MULTI] else [0])
            else:
                defaults.append([])
        multi_cols = self.schema.multi_cols

        def _input_fn():
            cols = self.schema.cols

            def parse_csv(value):
                data = tf.decode_csv(value, record_defaults=defaults)
                features = OrderedDict(zip(self.schema.cols, data))
                for col in multi_cols:
                    features[col] = tf.string_to_number(
                        tf.string_split([features[col]], ',').values, out_type=tf.int32)
                return features

            def add_seq_cols(feat):
                for m_col in multi_cols:
                    name = '{}_len'.format(m_col)
                    feat[name] = tf.size(feat[m_col])
                return feat

            dataset = tf.data.TextLineDataset(filenames)
            dataset = dataset.map(parse_csv, num_parallel_calls=4)
            dataset = dataset.map(add_seq_cols, num_parallel_calls=4)

            if shuffle:
                dataset = dataset.shuffle(n_batch * 10, seed=seed)
            dataset = dataset.repeat(n_epoch)

            # if got multi columns, we need padded_batch method, otherwise batch method
            has_multi = (df_conf[self.schema.M_DTYPE] == self.schema.CATG) & \
                        (df_conf[self.schema.IS_MULTI] == True)
            if sum(has_multi):
                final_cols = OrderedDict(zip(cols, tuple([None] if e else [] for e in has_multi)))
                for m_col in multi_cols:
                    final_cols['{}_len'.format(m_col)] = []
                dataset = dataset.padded_batch(n_batch, final_cols)
            else:
                dataset = dataset.batch(n_batch)

            features = dataset.make_one_shot_iterator().get_next()
            # Note that type of schema.label is list !
            return features, features.pop(self.schema.label[0])
        return _input_fn

    def serving_inputs(self):
        placeholders = OrderedDict()
        for name, tensor in self.features.items():
            placeholders[name] = tf.placeholder(shape=tensor.get_shape().as_list(), dtype=tensor.dtype, name=name)

        self.logger.info('{}: serving_inputs placeholders: {}'
                         .format(self.hparam.pid, placeholders))
        return tf.estimator.export.ServingInputReceiver(placeholders, placeholders)
        # import json
        #
        # p = self.hparam
        # loader = flex.Loader(p.conf_path, p.parsed_conf_path)
        # with flex.io('./serving.json').as_reader('r') as f:
        #     json_data = loader.trans_json(json.load(f.stream))[0]
        #
        # data = tf.data.Dataset.from_tensors(json_data)
        # data = data.repeat(1).make_one_shot_iterator().get_next()
        # # TODO: hack
        # print('serving data: ', data)
        # return tf.estimator.export.ServingInputReceiver(data, {'xxx': tf.placeholder(tf.float32, [None])})

    def create_est(self):
        run_config = tf.estimator.RunConfig(
            log_step_count_steps=300,
            tf_random_seed=seed,
            # save_checkpoints_secs=None,
            # save_checkpoints_steps=p.save_every_steps,
        )
        return tf.estimator.Estimator(model_fn=self.graphing, model_dir=self.model_dir, config=run_config)

    def fit(self, train_input, valid_input, reset=True):
        if reset:
            flex.io(self.model_dir).rm()

        p = self.hparam
        # summary_hook = tf.train.SummarySaverHook(
        #     100, output_dir=self.model_dir, summary_op=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
        train_spec = tf.estimator.TrainSpec(train_input, max_steps=p.train_steps, hooks=None)
        # exporter = tf.estimator.LatestExporter(p.export_name, self.serving_inputs)
        self.exporter = BestScoreExporter(p.export_name, self.serving_inputs)
        eval_spec = tf.estimator.EvalSpec(valid_input,
                                          steps=p.eval_steps,
                                          exporters=[self.exporter],
                                          name=p.eval_name,
                                          # throttle_secs=10
                                         )
        # try to build local export directory avoid error
        try:
            os.makedirs( os.path.join(self.model_dir, 'export', p.export_name) )
        except:
            self.logger.warn( traceback.format_exc() )

        self.estimator_ = self.create_est()
        tf.estimator.train_and_evaluate(self.estimator_, train_spec, eval_spec)
        return self

    def predict(self, sess, user_queries, items):
        pass


class MyHook(tf.train.SessionRunHook):
    def __init__(self, tensor):
        self.tensor = tensor

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.tensor)

    def after_run(self, run_context, run_values):
        print(len(run_values.results))


class BestScoreExporter(tf.estimator.Exporter):
    logger = env.logger('BestScoreExporter')

    def __init__(self,
                 name,
                 serving_input_receiver_fn,
                 assets_extra=None,
                 as_text=False):
        self._name = name
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.assets_extra = assets_extra
        self.as_text = as_text
        self.best = None
        self._exports_to_keep = 1
        self.export_result = None
        self.logger.info('BestScoreExporter init')

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):

        curloss = eval_result['loss']
        if self.best is None or self.best >= curloss:
            # clean first, only keep the best weights
            self.logger.info('clean export_path: {}'.format(export_path))
            export_ctx = flex.io(export_path)
            export_ctx.rm().mkdirs()

            self.best = curloss
            self.logger.info('nice eval loss: {}, export to pb'.format(curloss))
            self.export_result = estimator.export_savedmodel(
                export_path,
                self.serving_input_receiver_fn,
                assets_extra=self.assets_extra,
                as_text=self.as_text,
                checkpoint_path=checkpoint_path)
        else:
            self.logger.info('bad eval loss: {}'.format(curloss))

        return self.export_result
