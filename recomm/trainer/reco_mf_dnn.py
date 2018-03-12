import numpy as np, tensorflow as tf, os, shutil, time

from sklearn.preprocessing import minmax_scale
from keras.activations import selu
from collections import OrderedDict

randomSeed = 88


class ModelMfDNN(object):
    def __init__(self,
                 n_items,
                 n_genres,
                 model_dir,
                 dim=32,
                 learning_rate=0.01,
                 callback=None):
        self.n_items = n_items
        self.n_genres = n_genres
        self.ftr_cols = OrderedDict()
        self.callback = callback

        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope("inputs"):
                self.is_train = tf.placeholder(tf.bool, None)
                # user data
                self.query_movie_ids = tf.placeholder(tf.int32, [None, None])
                self.query_movie_ids_len = tf.placeholder(tf.int32, [None])

                # item data
                self.genres = tf.placeholder(tf.int32, [None, None])
                self.genres_len = tf.placeholder(tf.int32, [None])
                self.avg_rating = tf.placeholder(tf.float32, [None])
                self.year = tf.placeholder(tf.float32, [None])
                self.candidate_movie_id = tf.placeholder(tf.int32, [None])
                self.rating = tf.placeholder(tf.float32, [None])

            init_fn = tf.glorot_normal_initializer()
            emb_init_fn = tf.glorot_uniform_initializer()
            self.b_global = tf.Variable(emb_init_fn(shape=[]), name="b_global")
            with tf.variable_scope("embedding"):
                self.w_query_movie_ids = tf.Variable(emb_init_fn(shape=[self.n_items, dim]), name="w_query_movie_ids")
                self.b_query_movie_ids = tf.Variable(emb_init_fn(shape=[dim]), name="b_query_movie_ids")
                self.w_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items, dim]), name="w_candidate_movie_id")
                self.b_candidate_movie_id = tf.Variable(init_fn(shape=[dim + 8 + 2]), name="b_candidate_movie_id")
                # self.b_candidate_movie_id = tf.Variable(init_fn(shape=[self.n_items]), name="b_candidate_movie_id")
                self.w_genres = tf.Variable(emb_init_fn(shape=[self.n_genres, 8]), name="w_genres")

            with tf.variable_scope("user_encoding"):
                # query_movie embedding
                self.emb_query = tf.nn.embedding_lookup(self.w_query_movie_ids, self.query_movie_ids)
                query_movie_mask = tf.expand_dims(
                    tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(self.query_movie_ids_len)), 1), -1)
                self.emb_query = tf.reduce_sum(self.emb_query * query_movie_mask, 1)
                self.query_bias = tf.matmul(self.emb_query, self.b_query_movie_ids[:, tf.newaxis])
                self.emb_query = tf.layers.dense(self.emb_query, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_query = tf.layers.dense(self.emb_query, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_query = tf.layers.dense(self.emb_query, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_query = tf.layers.dense(self.emb_query, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

            # encode [item embedding + item metadata]
            with tf.variable_scope("item_encoding"):
                # candidate_movie embedding
                self.candidate_emb = tf.nn.embedding_lookup(self.w_candidate_movie_id, self.candidate_movie_id)
                # genres embedding
                self.emb_genres = tf.nn.embedding_lookup(self.w_genres, tf.to_int32(self.genres))
                genres_mask = tf.expand_dims(
                    tf.nn.l2_normalize(tf.to_float(tf.sequence_mask(tf.reshape(self.genres_len, [-1]))), 1), -1)
                self.emb_genres = tf.reduce_sum(self.emb_genres * genres_mask, 1)

                self.emb_item = tf.concat([self.candidate_emb, self.emb_genres, self.avg_rating[:, tf.newaxis], self.year[:, tf.newaxis]], 1)
                self.candidate_bias = tf.matmul(self.emb_item, self.b_candidate_movie_id[:, tf.newaxis])
                self.emb_item = tf.layers.dense(self.emb_item, 128, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_item = tf.layers.dense(self.emb_item, 64, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_item = tf.layers.dense(self.emb_item, 32, kernel_initializer=init_fn, activation=tf.nn.selu)
                self.emb_item = tf.layers.dense(self.emb_item, 16, kernel_initializer=init_fn, activation=tf.nn.selu)

            # elements wise dot of user and item embedding
            with tf.variable_scope("gmf"):
                self.gmf = tf.reduce_sum(self.emb_query * self.emb_item, 1, keep_dims=True)
                self.gmf = tf.add(self.gmf, self.b_global)
                self.gmf = tf.add(self.gmf, self.query_bias)
                self.gmf = tf.add(self.gmf, self.candidate_bias, name="infer")

                # one query for all items, for predict speed
                self.pred = tf.matmul(self.emb_query, tf.transpose(self.emb_item)) + \
                            tf.reshape(self.candidate_bias, (1, -1)) + \
                            self.query_bias + \
                            self.b_global
                self.pred = tf.nn.sigmoid(self.pred)

            with tf.variable_scope("loss"):
                self.alter_rating = tf.to_float(self.rating >= 4)[:, tf.newaxis]
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.alter_rating, logits=self.gmf))

            with tf.variable_scope("train"):
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                # self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
                pass

            self.saver = tf.train.Saver(tf.global_variables())
            self.graph = graph
            self.model_dir = model_dir

    def reset_model(self, model_dir):
        shutil.rmtree(path=model_dir, ignore_errors=True)
        os.makedirs(model_dir)

    def feed_dict(self, data, mode="train"):
        ret = {
            self.query_movie_ids: data["query_movie_ids"],
            self.query_movie_ids_len: data["query_movie_ids_len"],
            self.genres: data["genres"],
            self.genres_len: data["genres_len"],
            self.avg_rating: data["avg_rating"],
            self.year: data["year"],
            self.candidate_movie_id: data["candidate_movie_id"]
        }
        ret[self.is_train] = False
        if mode != "infer":
            ret[self.rating] = data["rating"]
            if mode == "train":
                ret[self.is_train] = True
            elif mode == "eval":
                pass
        return ret

    def fit(self, sess, trainGen, testGen, reset=False, n_epoch=50):
        sess.run(tf.global_variables_initializer())
        if reset:
            print("reset model: clean model dir: {} ...".format(self.model_dir))
            self.reset_model(self.model_dir)
        # try: 試著重上次儲存的model再次training
        self.ckpt(sess, self.model_dir)

        start = time.time()
        print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
        minLoss = 1e7
        for ep in range(1, n_epoch + 1):
            tr_loss, tr_total = 0, 0
            for i, data in enumerate(trainGen(), 1):
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=self.feed_dict(data, mode="train"))
                batch_len = len(data["query_movie_ids"])
                tr_loss += loss * batch_len
                tr_total += batch_len
                print("\rtrain loss: {:.3f}".format(loss), end="")

            if testGen is not None:
                te_loss = self.epoch_loss(sess, testGen)

            tpl = "\r%02d\t%.3f\t\t%.3f\t\t%.3f secs"
            if minLoss > te_loss:
                tpl += ", saving ..."
                self.saver.save(sess, os.path.join(self.model_dir, 'model'), global_step=ep)
                minLoss = te_loss

            end = time.time()
            print(tpl % (ep, tr_loss / tr_total, te_loss, end - start))
            start = end
        return self

    def ckpt(self, sess, model_dir):
        """load latest saved model"""
        latestCkpt = tf.train.latest_checkpoint(model_dir)
        if latestCkpt:
            self.saver.restore(sess, latestCkpt)
        return latestCkpt

    def epoch_loss(self, sess, data_gen):
        tot_loss, tot_cnt = 0, 0
        for data in data_gen():
            loss_tensor = self.loss
            loss = sess.run(loss_tensor, feed_dict=self.feed_dict(data, mode="eval"))
            tot_loss += loss * len(data["query_movie_ids"])
            tot_cnt += len(data["query_movie_ids"])
        return tot_loss / tot_cnt

    def predict(self, sess, user_queries, items):
        self.ckpt(sess, self.model_dir)
        return sess.run(self.pred, feed_dict={
            self.is_train: False,
            self.query_movie_ids: user_queries["query_movie_ids"],
            self.query_movie_ids_len: user_queries["query_movie_ids_len"],

            self.genres: items["genres"],
            self.genres_len: items["genres_len"],
            self.avg_rating: items["avg_rating"],
            self.year: items["year"],
            self.candidate_movie_id: items["candidate_movie_id"]
        })

