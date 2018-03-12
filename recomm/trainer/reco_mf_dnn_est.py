import tensorflow as tf, os, traceback
from . import env
from .utils import flex, utils

from collections import OrderedDict

seed = 88

class ModelMfDNN(object):
    logger = env.logger('ModelMfDNN')

    def __init__(self,
                 hparam=None,
                 schema=None,
                 n_items=None,
                 n_genres=None):
        self.n_items = n_items
        self.n_genres = n_genres
        self.schema = schema
        self.hparam = hparam
        self.exporter = None
        if hparam is not None:
            self.model_dir = hparam.job_dir

    def graphing(self, features, labels, mode):
        self.logger.info('mode = {}'.format(mode))
        p = self.hparam
        self.features, self.labels = features, labels
        for name, tensor in self.features.items():
            setattr(self, name, tensor)

        with tf.variable_scope("init") as scope:
            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")

            with tf.variable_scope("embedding") as scope:
                self.w_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.n_items, p.dim]), name="w_query_movie_ids")
                self.b_query_movie_ids = tf.Variable(emb_init_fn(shape=[p.dim]), name="b_query_movie_ids")
                self.w_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items, p.dim]), name="w_candidate_movie_id")
                self.b_candidate_movie_id = tf.Variable(init_fn(shape=[p.dim + 8 + 2]), name="b_candidate_movie_id")
                # self.b_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items]), name="b_candidate_movie_id")
                self.w_genres = tf.Variable(emb_init_fn(shape=[self.n_genres, 8]), name="w_genres")

        with tf.variable_scope("user_encoding") as scope:
            # query_movie embedding
            self.emb_query = tf.nn.embedding_lookup(self.w_query_movie_ids, self.query_movie_ids)
            query_movie_mask = tf.expand_dims(
                tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.query_movie_ids_len)), 1), -1)
            self.emb_query = tf.reduce_sum(self.emb_query * query_movie_mask, 1)
            self.query_bias = tf.matmul(self.emb_query, self.b_query_movie_ids[:, tf.newaxis])
            self.emb_query = tf.layers.dense(self.emb_query, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_query = tf.layers.dense(self.emb_query, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_query = tf.layers.dense(self.emb_query, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
            # self.emb_query = tf.layers.dense(self.emb_query, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

        # encode [item embedding + item metadata]
        with tf.variable_scope("item_encoding") as scope:
            # candidate_movie embedding
            self.candidate_emb = tf.nn.embedding_lookup(self.w_candidate_movie_id, self.candidate_movie_id)
            # genres embedding
            self.emb_genres = tf.nn.embedding_lookup(self.w_genres, tf.to_int32(self.genres))
            genres_mask = tf.expand_dims(
                tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.genres_len)), 1), -1)
            self.emb_genres = tf.reduce_sum(self.emb_genres * genres_mask, 1)

            self.emb_item = tf.concat([self.candidate_emb, self.emb_genres, self.avg_rating[:, tf.newaxis], self.year[:, tf.newaxis]], 1)
            self.candidate_bias = tf.matmul(self.emb_item, self.b_candidate_movie_id[:, tf.newaxis])
            self.emb_item = tf.layers.dense(self.emb_item, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_item = tf.layers.dense(self.emb_item, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_item = tf.layers.dense(self.emb_item, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
            # self.emb_item = tf.layers.dense(self.emb_item, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

        # elements wise dot of user and item embedding
        with tf.variable_scope("gmf") as scope:
            self.gmf = tf.reduce_sum(self.emb_query * self.emb_item, 1, keep_dims=True)
            self.gmf = tf.add(self.gmf, self.b_global)
            self.gmf = tf.add(self.gmf, self.query_bias)
            self.gmf = tf.add(self.gmf, self.candidate_bias)
            self.infer = tf.nn.sigmoid(self.gmf, name="infer")

            # one query for all items, for predict speed
            self.pred = tf.matmul(self.emb_query, tf.transpose(self.emb_item)) + \
                        tf.reshape(self.candidate_bias, (1, -1)) + \
                        self.query_bias + \
                        self.b_global
            self.pred = tf.nn.sigmoid(self.pred, name='pred')

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'emb_query': self.emb_query,
                    'emb_item': self.emb_item,
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

    def graphing2(self, features, labels, mode):
        self.logger.info('mode = {}'.format(mode))
        p = self.hparam
        self.features, self.labels = features, labels
        for name, tensor in self.features.items():
            setattr(self, name, tensor)

        with tf.variable_scope("init") as scope:
            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")

            with tf.variable_scope("embedding") as scope:
                self.w_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.n_items, p.dim]), name="w_query_movie_ids")
                self.b_query_movie_ids = tf.Variable(emb_init_fn(shape=[p.dim]), name="b_query_movie_ids")
                self.w_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items, p.dim]), name="w_candidate_movie_id")
                self.b_candidate_movie_id = tf.Variable(init_fn(shape=[p.dim + 8 + 2]), name="b_candidate_movie_id")
                # self.b_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items]), name="b_candidate_movie_id")
                self.w_genres = tf.Variable(emb_init_fn(shape=[self.n_genres, 8]), name="w_genres")

        with tf.variable_scope("user_encoding") as scope:
            # query_movie embedding
            self.emb_query = tf.nn.embedding_lookup(self.w_query_movie_ids, self.query_movie_ids)
            query_movie_mask = tf.expand_dims(
                tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(tf.reshape(self.query_movie_ids_len, [-1]))), 1), -1)
            self.emb_query = tf.reduce_sum(self.emb_query * query_movie_mask, 1)
            self.query_bias = tf.matmul(self.emb_query, self.b_query_movie_ids[:, tf.newaxis])
            self.emb_query = tf.layers.dense(self.emb_query, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_query = tf.layers.dense(self.emb_query, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_query = tf.layers.dense(self.emb_query, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
            # self.emb_query = tf.layers.dense(self.emb_query, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

        # encode [item embedding + item metadata]
        with tf.variable_scope("item_encoding") as scope:
            # candidate_movie embedding
            self.candidate_emb = tf.nn.embedding_lookup(self.w_candidate_movie_id,
                                                        tf.reshape(self.candidate_movie_id, [-1]))
            # genres embedding
            self.emb_genres = tf.nn.embedding_lookup(self.w_genres, tf.to_int32(self.genres))
            genres_mask = tf.expand_dims(
                tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(tf.reshape(self.genres_len, [-1]))), 1), -1)
            self.emb_genres = tf.reduce_sum(self.emb_genres * genres_mask, 1)

            self.emb_item = tf.concat([self.candidate_emb, self.emb_genres, self.avg_rating, self.year], 1)
            self.candidate_bias = tf.matmul(self.emb_item, self.b_candidate_movie_id[:, tf.newaxis])
            self.emb_item = tf.layers.dense(self.emb_item, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_item = tf.layers.dense(self.emb_item, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
            self.emb_item = tf.layers.dense(self.emb_item, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
            # self.emb_item = tf.layers.dense(self.emb_item, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

        # elements wise dot of user and item embedding
        with tf.variable_scope("gmf") as scope:
            self.gmf = tf.reduce_sum(self.emb_query * self.emb_item, 1, keep_dims=True)
            self.gmf = tf.add(self.gmf, self.b_global)
            self.gmf = tf.add(self.gmf, self.query_bias)
            self.gmf = tf.add(self.gmf, self.candidate_bias, name="infer")

            # one query for all items, for predict speed
            self.pred = tf.matmul(self.emb_query, tf.transpose(self.emb_item)) + \
                        tf.reshape(self.candidate_bias, (1, -1)) + \
                        self.query_bias + \
                        self.b_global
            self.pred = tf.nn.sigmoid(self.pred, name='pred')

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'emb_query': self.emb_query,
                    'emb_item': self.emb_item,
                    'pred': self.pred
                })
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=self.pred,
                                              export_outputs=export_outputs)

        with tf.variable_scope("loss") as scope:
            # self.alter_rating = tf.to_float(self.label >= 4)[:, tf.newaxis]
            self.ans = tf.to_float(self.labels)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.ans, logits=self.gmf))
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope("metrics") as scope:
            self.auc = tf.metrics.auc(tf.cast(self.labels, tf.bool),
                                      tf.reshape(tf.nn.sigmoid(self.gmf), [-1]))
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
        cols = ['query_movie_ids', 'genres', 'avg_rating', 'year', 'candidate_movie_id', 'rating']
        defaults = [[''], [''], [], [], [0], [0]]
        multi_cols = ('query_movie_ids', 'genres')

        def _input_fn():
            def parse_csv(value):
                data = tf.decode_csv(value, record_defaults=defaults)
                features = OrderedDict(zip(cols, data))
                for col in multi_cols:
                    features[col] = tf.string_to_number(
                        tf.string_split([features[col]], ',').values, out_type=tf.int32)
                return features

            def add_seq_cols(feat):
                for m_col in multi_cols:
                    name = '{}_len'.format(m_col)
                    feat[name] = tf.size(feat[m_col])
                    cols.append(name)
                return feat

            dataset = tf.data.TextLineDataset(filenames)
            dataset = dataset.map(parse_csv, num_parallel_calls=4)
            dataset = dataset.map(add_seq_cols, num_parallel_calls=4)
            if shuffle:
                dataset = dataset.shuffle(n_batch * 10, seed=seed)
            dataset = dataset.repeat(n_epoch)
            dataset = dataset.padded_batch(n_batch, OrderedDict(zip(cols, ([None], [None], [], [], [], [], [], [], []))))
            features = dataset.make_one_shot_iterator().get_next()
            return features, features.pop('rating')
        return _input_fn

    def input_fn2(self, filenames, n_batch=128, n_epoch=None, shuffle=True):
        cols = ['query_movie_ids', 'genres', 'avg_rating', 'year', 'candidate_movie_id', 'rating']
        defaults = [[''], [''], [], [], [0], [0]]
        multi_cols = ('query_movie_ids', 'genres')

        def _input_fn():
            def parse_csv(value):
                data = tf.decode_csv(value, record_defaults=defaults)
                features = OrderedDict(zip(cols, [tf.expand_dims(v, -1) for v in data]))
                for col in multi_cols:
                    features[col] = tf.string_to_number(
                        tf.string_split(features[col], ',').values, out_type=tf.int32)
                return features

            def add_seq_cols(feat):
                for m_col in multi_cols:
                    name = '{}_len'.format(m_col)
                    feat[name] = tf.expand_dims(tf.size(feat[m_col]), -1)
                    cols.append(name)
                return feat

            dataset = tf.data.TextLineDataset(filenames)
            dataset = dataset.map(parse_csv, num_parallel_calls=4)
            dataset = dataset.map(add_seq_cols, num_parallel_calls=4)
            if shuffle:
                dataset = dataset.shuffle(n_batch * 10, seed=seed)
            dataset = dataset.repeat(n_epoch)
            dataset = dataset.padded_batch(n_batch, OrderedDict(zip(cols, ([None], [None], [1], [1], [1], [1], [1], [1], [1]))))
            features = dataset.make_one_shot_iterator().get_next()
            return features, features.pop('rating')
        return _input_fn

    def serving_inputs(self):
        placeholders = OrderedDict()
        for name, tensor in self.features.items():
            placeholders[name] = tf.placeholder(shape=tensor.get_shape().as_list(), dtype=tensor.dtype, name=name)

        # TODO:hack
        print('\nplaceholders', placeholders)
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

        # self.estimator_ = tf.estimator.Estimator(model_fn=self.graphing, model_dir=self.model_dir, config=run_config)
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
