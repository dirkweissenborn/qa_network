import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.ops.rnn import dynamic_rnn
from query import *
import tf_util


def default_init():
    return tf.random_normal_initializer(0.0, 0.1)

class QAModel:
    """
    A QAModel embeds ClozeQuery (see query.py) and their respective support to answer the query by selecting one of the
    specified answer candidates. This is an implementation of the system described in [to appear].
    """

    def __init__(self, size, batch_size, vocab_size, answer_vocab_size, max_length, is_train=True, learning_rate=1e-2,
                 composition="GRU", max_hops=0, devices=None, keep_prob=1.0):
        """
        :param size: size of hidden states
        :param batch_size: initial batch_size (adapts automatically)
        :param vocab_size: size of input vocabulary (vocabulary of contexts)
        :param answer_vocab_size: size of answer (candidates) vocabulary
        :param max_length: maximum length of an individual context
        :param is_train:
        :param learning_rate:
        :param composition: "GRU", "LSTM", "BiGRU" are possible
        :param max_hops: maximum number of hops, can be set manually to something lower by assigning a different value
        to variable (self.)num_hops which is initialized with max_hops
        :param devices: defaults to ["/cpu:0"], but can be a list of up to 3 devices. The model is automatically
        partitioned into the different devices.
        :param keep_prob: 1.0-dropout rate, that is applied to the input embeddings
        """
        self._vocab_size = vocab_size
        self._max_length = max_length
        self._size = size
        self._batch_size = batch_size
        self._is_train = is_train
        self._composition = composition
        self._max_hops = max_hops
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"

        self._init = tf.random_normal_initializer(0.0, 0.1)
        with tf.device(self._device0):
            with tf.variable_scope(self.name(), initializer=tf.contrib.layers.xavier_initializer()):
                self._init_inputs()
                self.keep_prob = tf.get_variable("keep_prob", [], initializer=tf.constant_initializer(keep_prob))
                with tf.device("/cpu:0"):
                    # embeddings
                    self.output_embedding = tf.get_variable("E_candidate", [answer_vocab_size, self._size],
                                                            initializer=self._init)
                    self.input_embedding = tf.get_variable("E_words", [vocab_size, self._size],
                                                           initializer=self._init)
                    answer, _ = tf.dynamic_partition(self._answer_input, self._query_partition, 2)
                    lookup_individual = tf.nn.embedding_lookup(self.output_embedding, answer)
                    cands, _ = tf.dynamic_partition(self._answer_candidates, self._query_partition, 2)
                    self.candidate_lookup = tf.nn.embedding_lookup(self.output_embedding, cands)

                self.num_hops = tf.Variable(self._max_hops, trainable=False, name="num_queries")
                self.query = self._comp_f()
                answer = self._retrieve_answer(self.query)
                self.score = tf_util.batch_dot(lookup_individual, answer)
                self.scores_with_negs = self._score_candidates(answer)

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.global_step = tf.Variable(0, trainable=False, name="step")

                    self.opt = tf.train.AdamOptimizer(self.learning_rate)

                    current_batch_size = tf.gather(tf.shape(self.scores_with_negs), [0])

                    loss = math_ops.reduce_sum(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(self.scores_with_negs,
                                                                       tf.tile(tf.constant([0], tf.int64),
                                                                               current_batch_size)))

                    train_params = tf.trainable_variables()
                    self.training_weight = tf.Variable(1.0, trainable=False, name="training_weight")

                    self._loss = loss / math_ops.cast(current_batch_size, tf.float32)
                    self._grads = tf.gradients(self._loss, train_params, self.training_weight, colocate_gradients_with_ops=True)

                    if len(train_params) > 0:
                        grads, _ = tf.clip_by_global_norm(self._grads, 5.0)
                        self._update = self.opt.apply_gradients(zip(grads, train_params),
                                                                global_step=self.global_step)
                    else:
                        self._update = tf.assign_add(self.global_step, 1)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

    def _score_candidates(self, answer):
        return tf.squeeze(tf.batch_matmul(self.candidate_lookup, tf.expand_dims(answer, [2])), [2]) + \
                                         self._candidate_mask  # number of negative candidates can vary for each example

    def _composition_function(self, inputs, length, init_state=None):
        if self._composition == "GRU":
            cell = GRUCell(self._size)
            return dynamic_rnn(cell, inputs, sequence_length=length, time_major=True,
                               initial_state=init_state, dtype=tf.float32)[0]
        elif self._composition == "LSTM":
            cell = BasicLSTMCell(self._size)
            init_state = tf.concat(1, [tf.zeros_like(init_state, tf.float32), init_state]) if init_state else None
            outs = dynamic_rnn(cell, inputs, sequence_length=length, time_major=True,
                               initial_state=init_state, dtype=tf.float32)[0]
            return outs
        elif self._composition == "BiGRU":
            cell = GRUCell(self._size // 2, self._size)
            init_state_fw, init_state_bw = tf.split(1, 2, init_state) if init_state else (None, None)
            with tf.variable_scope("forward"):
                fw_outs = dynamic_rnn(cell, inputs, sequence_length=length, time_major=True,
                                      initial_state=init_state_fw, dtype=tf.float32)[0]
            with tf.variable_scope("backward"):
                rev_inputs = tf.reverse_sequence(tf.pack(inputs), length, 0, 1)
                rev_inputs = [tf.reshape(x, [-1, self._size]) for x in tf.split(0, len(inputs), rev_inputs)]
                bw_outs = dynamic_rnn(cell, rev_inputs, sequence_length=length, time_major=True,
                                      initial_state=init_state_bw, dtype=tf.float32)[0]
                bw_outs = tf.reverse_sequence(tf.pack(bw_outs), length, 0, 1)
                bw_outs = [tf.reshape(x, [-1, self._size]) for x in tf.split(0, len(inputs), bw_outs)]
            return [tf.concat(1, [fw_out, bw_out]) for fw_out, bw_out in zip(fw_outs, bw_outs)]
        else:
            raise NotImplementedError("Other compositions not implemented yet.")

    def name(self):
        return self.__class__.__name__

    def _comp_f(self):
        """
        Encodes all queries (including supporting queries)
        :return: encoded queries
        """
        with tf.device("/cpu:0"):
            max_length = tf.cast(tf.reduce_max(self._length), tf.int32)
            context_t = tf.transpose(self._context)
            context_t = tf.slice(context_t, [0, 0], tf.pack([max_length, -1]))
            embedded = tf.nn.embedding_lookup(self.input_embedding, context_t)
            embedded = tf.nn.dropout(embedded, self.keep_prob)
            batch_size = tf.shape(self._context)[0]
            batch_size_32 = tf.reshape(batch_size, [1])
            batch_size_64 = tf.cast(batch_size, tf.int64)

        with tf.device(self._device1):
            #use other device for backward rnn
            with tf.variable_scope("backward"):
                min_end = tf.segment_min(self._ends, self._span_context)
                init_state = tf.get_variable("init_state", [self._size], initializer=self._init)
                init_state = tf.reshape(tf.tile(init_state, batch_size_32), [-1, self._size])
                rev_embedded = tf.reverse_sequence(embedded, self._length, 0, 1)
                # TIME-MAJOR: [T, B, S]
                outs_bw = self._composition_function(rev_embedded, self._length - min_end, init_state)
                # reshape to all possible queries for all sequences. Dim[0]=batch_size*(max_length+1).
                # "+1" because we include the initial state
                outs_bw = tf.reshape(tf.concat(0, [tf.expand_dims(init_state, 0), outs_bw]), [-1, self._size])
                # gather respective queries via their lengths-start (because reversed sequence)
                lengths_aligned = tf.gather(self._length, self._span_context)
                out_bw = tf.gather(outs_bw, (lengths_aligned - self._ends) * batch_size_64 + self._span_context)

        with tf.device(self._device2):
            with tf.variable_scope("forward"):
                #e_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._max_length, embedded)]
                max_start = tf.segment_max(self._starts, self._span_context)
                init_state = tf.get_variable("init_state", [self._size], initializer=self._init)
                init_state = tf.reshape(tf.tile(init_state, batch_size_32), [-1, self._size])
                # TIME-MAJOR: [T, B, S]
                outs_fw = self._composition_function(embedded, max_start, init_state)
                # reshape to all possible queries for all sequences. Dim[0]=batch_size*(max_length+1).
                # "+1" because we include the initial state
                outs_fw = tf.reshape(tf.concat(0, [tf.expand_dims(init_state, 0), outs_fw]), [-1, self._size])
                # gather respective queries via their positions (with offset of batch_size*ends)
                out_fw = tf.gather(outs_fw, self._starts * batch_size_64 + self._span_context)
            # form query from forward and backward compositions
            query = tf.contrib.layers.fully_connected(tf.concat(1, [out_fw, out_bw]), self._size,
                                                      activation_fn=None, weights_initializer=None, biases_initializer=None)
            query = tf.add_n([query, out_bw, out_fw])

        return query

    def set_train(self, sess):
        """
        enables dropout
        :param sess:
        :return:
        """
        sess.run(self.keep_prob.initializer)

    def set_eval(self, sess):
        """
        removes dropout
        :param sess:
        :return:
        """
        sess.run(self.keep_prob.assign(1.0))

    def _retrieve_answer(self, query):
        """
        Retrieves answer based on the specified query. Implements consecutive updates to the query and answer.
        :return: answer, if num_hops is 0, returns query itself
        """
        query, supp_queries = tf.dynamic_partition(query, self._query_partition, 2)
        with tf.variable_scope("support"):
            num_queries = tf.shape(query)[0]

            with tf.device("/cpu:0"):
                _, supp_answer_output_ids = tf.dynamic_partition(self._answer_input, self._query_partition, 2)
                _, supp_answer_input_ids = tf.dynamic_partition(self._answer_word_input, self._query_partition, 2)
                supp_answers = tf.nn.embedding_lookup(self.output_embedding, supp_answer_output_ids)
                aligned_supp_answers = tf.gather(supp_answers, self._support_ids)  # and with respective answers

                if self._max_hops > 1:
                    # used in multihop
                    answer_words = tf.nn.embedding_lookup(self.input_embedding, supp_answer_input_ids)
                    aligned_answers_input = tf.gather(answer_words, self._support_ids)

            self.support_scores = []
            query_as_answer = tf.contrib.layers.fully_connected(query, self._size,
                                                                activation_fn=None, weights_initializer=None,
                                                                biases_initializer=None, scope="query_to_answer")
            query_as_answer = query_as_answer * tf.sigmoid(tf.get_variable("query_as_answer_gate", tuple(),
                                                                           initializer=tf.constant_initializer(0.0)))
            current_answer = query_as_answer
            current_query = query

            aligned_support = tf.gather(supp_queries, self._support_ids)  # align supp_queries with queries
            collab_support = tf.gather(query, self._collab_support_ids)  # align supp_queries with queries
            aligned_support = tf.concat(0, [aligned_support, collab_support])

            query_ids = tf.concat(0, [self._query_ids, self._collab_query_ids])
            self.answer_weights = []


            for i in range(self._max_hops):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                collab_queries = tf.gather(current_query, self._collab_query_ids)  # align supp_queries with queries
                aligned_queries = tf.gather(current_query, self._query_ids)  # align queries
                aligned_queries = tf.concat(0, [aligned_queries, collab_queries])

                with tf.variable_scope("support_scores"):
                    scores = tf_util.batch_dot(aligned_queries, aligned_support)
                    self.support_scores.append(scores)
                    score_max = tf.gather(tf.segment_max(scores, query_ids), query_ids)
                    e_scores = tf.exp(scores - score_max)
                    norm = tf.unsorted_segment_sum(e_scores, query_ids, num_queries) + 0.00001 # for zero norms
                    norm = tf.expand_dims(norm, 1)
                    e_scores = tf.expand_dims(e_scores, 1)

                with tf.variable_scope("support_answers"):
                    aligned_supp_answers_with_collab = tf.concat(0, [aligned_supp_answers, collab_queries])
                    weighted_supp_answers = tf.unsorted_segment_sum(e_scores * aligned_supp_answers_with_collab,
                                                               query_ids, num_queries) / norm

                with tf.variable_scope("support_queries"):
                    weighted_supp_queries = tf.unsorted_segment_sum(e_scores * aligned_support, query_ids, num_queries) / norm
                
                with tf.variable_scope("answer_accumulation"):
                    answer_scores = tf.reduce_max(self._score_candidates(current_answer), [1], keep_dims=True)
                    answer_weight = tf.contrib.layers.fully_connected(tf.concat(1, [query_as_answer * weighted_supp_answers,
                                                                                    weighted_supp_queries * current_query,
                                                                                    answer_scores]),
                                                                      1,
                                                                      activation_fn=tf.nn.sigmoid,
                                                                      weights_initializer=tf.constant_initializer(0.0),
                                                                      biases_initializer=tf.constant_initializer(0.0),
                                                                      scope="answer_weight")

                    new_answer = answer_weight * weighted_supp_answers + current_answer

                    # this condition allows for setting varying number of hops
                    current_answer = tf.cond(tf.greater(self.num_hops, i),
                                             lambda: new_answer,
                                             lambda: current_answer)

                    self.answer_weights.append(answer_weight)

                if i < self._max_hops - 1:
                    with tf.variable_scope("query_update"):
                        # prepare subsequent query
                        aligned_answers_input_with_collab = tf.concat(0, [aligned_answers_input, collab_queries])
                        weighted_answer_words = tf.unsorted_segment_sum(e_scores * aligned_answers_input_with_collab,
                                                                        query_ids, num_queries) / norm

                        c = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_supp_queries, weighted_answer_words]),
                                                              self._size, activation_fn=tf.tanh, scope="update_candidate",
                                                              weights_initializer=None, biases_initializer=None)

                        gate = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_supp_queries]),
                                                                 self._size, activation_fn=tf.sigmoid,
                                                                 weights_initializer=None, scope="update_gate",
                                                                 biases_initializer=tf.constant_initializer(1))
                        current_query = gate * current_query + (1-gate) * c

            return current_answer

    def _init_inputs(self):
        #General
        with tf.device("/cpu:0"):
            self._context = tf.placeholder(tf.int64, shape=[None, self._max_length], name="context")
            self._answer_candidates = tf.placeholder(tf.int64, shape=[None, None], name="candidates")
            self._answer_input = tf.placeholder(tf.int64, shape=[None], name="answer")
            # answer word ids (index to E_embeddings) might differ from answer ids (input to E_candidates)
            self._answer_word_input = tf.placeholder(tf.int64, shape=[None], name="answer_word")
            self._starts = tf.placeholder(tf.int64, shape=[None], name="span_start")
            self._ends = tf.placeholder(tf.int64, shape=[None], name="span_end")
            # holds batch idx for respective span
            self._span_context = tf.placeholder(tf.int64, shape=[None], name="answer_position_context")
            self._candidate_mask = tf.placeholder(tf.float32, shape=[None, None], name="candidate_mask")
            self._length = tf.placeholder(tf.int64, shape=[None], name="context_length")

        self._ctxt = np.zeros([self._batch_size, self._max_length], dtype=np.int64)
        self._len = np.zeros([self._batch_size], dtype=np.int64)

        #Supporting Evidence
        # partition of queries (class 0) and support (class 1)
        self._query_partition = tf.placeholder(tf.int32, [None], "query_partition")
        # aligned support ids with query ids for supporting evidence
        self._support_ids = tf.placeholder(tf.int64, shape=[None], name="support_for_query_ids")
        self._collab_support_ids = tf.placeholder(tf.int64, shape=[None], name="collab_supp_ids")
        self._query_ids = tf.placeholder(tf.int64, shape=[None], name="query_for_support_ids")
        self._collab_query_ids = tf.placeholder(tf.int64, shape=[None], name="collab_query_ids")

        self._feed_dict = {}

    def _change_batch_size(self, batch_size):
        new_ctxt_in = np.zeros([batch_size, self._max_length], dtype=np.int64)
        new_ctxt_in[:self._batch_size] = self._ctxt
        self._ctxt = new_ctxt_in

        new_length = np.zeros([batch_size], dtype=np.int64)
        new_length[:self._batch_size] = self._len
        self._len = new_length

        self._batch_size = batch_size

    def _start_adding_examples(self):
        self._batch_idx = 0
        self._query_idx = 0
        self._support_idx = 0
        self._answer_cands = []
        self._answer_in = []
        self._answer_word_in = []
        self._s = []
        self._e = []
        self._span_ctxt = []
        # supporting evidence
        self._query_part = []
        self.queries_for_support = []
        self.support_for_queries = []
        self._collab_queries = []
        self._collab_support = []

        self.supporting_qa = []

    def _add_example(self, context_queries, is_query=True):
        '''
        All queries and supporting queries are encoded the same. However we keep track of which are queries,
        which are support and how they belong to each other via partition variables and aligned support to query
        and query to support ids.
        :param context_queries: contains all queries about a particular context, see ContextQueries in query.py
        :param is_query: True if this is query, False if this is support
        :return:
        '''
        assert is_query or context_queries.support is None, "Support cannot have support!"
        if self._batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, self._batch_idx))
        self._ctxt[self._batch_idx][:len(context_queries.context)] = context_queries.context
        self._len[self._batch_idx] = len(context_queries.context)

        batch_idx = self._batch_idx
        self._batch_idx += 1
        for i, q in enumerate(context_queries.queries):
            self._s.append(q.start)
            self._e.append(q.end)
            self._span_ctxt.append(batch_idx)
            self._answer_in.append(q.answer if q.answer is not None else q.candidates[0])
            self._answer_word_in.append(q.answer_word)
            cands = [q.answer] if q.answer is not None else []
            if q.candidates is not None:
                cands.extend(c for c in q.candidates if c != q.answer)
            self._answer_cands.append(cands)
            self._query_part.append(0 if is_query else 1)

        if is_query:
            if context_queries.collaborative_support:
                # save queries also as support, only with different query_partition index (1 for support)
                for i in range(len(context_queries.queries)):
                    for j in range(len(context_queries.queries)):
                        if j != i:
                            self._collab_queries.append(self._query_idx+i)
                            self._collab_support.append(self._query_idx+j)

            ### add query specific supports ###
            for i, q in enumerate(context_queries.queries):
                if q.support is not None and self._max_hops > 0:
                    for qs in q.support:
                        if qs.context is None:
                            #supporting context is the same as query context, only add corresponding positions
                            for q in qs.queries:
                                self._s.append(q.start)
                                self._e.append(q.end)
                                self._span_ctxt.append(batch_idx)
                                self._answer_in.append(q.answer)
                                self._answer_word_in.append(q.answer_word)
                                self._answer_cands.append([q.answer])
                                self._query_part.append(1)
                                self.supporting_qa.append((q.context, q.start, q.end, q.answer))
                        else:
                            self._add_example(qs, is_query=False)
                        # align queries with support idxs
                        self.support_for_queries.extend(range(self._support_idx, self._support_idx+len(qs.queries)))
                        self._support_idx += len(qs.queries)
                        self.queries_for_support.extend([self._query_idx] * len(qs.queries))
                self._query_idx += 1

            ### add context specific support to all queries of this context ###
            if context_queries.support is not None and self._max_hops > 0:
                for qs in context_queries.support:
                    if qs.context is None:
                        for q in qs.queries:
                            self._s.append(q.start)
                            self._e.append(q.end)
                            self._span_ctxt.append(batch_idx)
                            self._answer_in.append(q.answer)
                            self._answer_word_in.append(q.answer_word)
                            self._answer_cands.append([q.answer])
                            self._query_part.append(1)
                            self.supporting_qa.append((q.context, q.start, q.end, q.answer))
                    else:
                        self._add_example(qs, is_query=False)
                    # this evidence supports all queries in this context
                    for i, _ in enumerate(context_queries.queries):
                        # align queries with support idxs
                        self.support_for_queries.extend(range(self._support_idx, self._support_idx+len(qs.queries)))
                        self.queries_for_support.extend([self._query_idx - len(context_queries.queries) + i] * len(qs.queries))
                    self._support_idx += len(qs.queries)
        else:
            for i, q in enumerate(context_queries.queries):
                self.supporting_qa.append((q.context, q.start, q.end, q.answer))

    def _finish_adding_examples(self):
        max_cands = max((len(x) for x in self._answer_cands))
        # mask is used to determine which candidates are real candidates and which are dummies,
        # number of candidates can vary from query to query within a batch
        cand_mask = []
        for i in range(len(self._answer_cands)):
            l = len(self._answer_cands[i])
            if self._query_part[i] == 0: # if this is a query (and not supporting evidence)
                mask = [0] * l
            for _ in range(max_cands - l):
                self._answer_cands[i].append(self._answer_cands[i][0])  # dummy
                if self._query_part[i] == 0:
                    mask.append(-1e6)  # this is added to scores, serves basically as a bias mask to exclude dummy negative candidates
            if self._query_part[i] == 0:
                cand_mask.append(mask)

        if self._batch_idx < self._batch_size:
            self._feed_dict[self._context] = self._ctxt[:self._batch_idx]
            self._feed_dict[self._length] = self._len[:self._batch_idx]
        else:
            self._feed_dict[self._context] = self._ctxt
            self._feed_dict[self._length] = self._len
        self._feed_dict[self._starts] = self._s
        self._feed_dict[self._ends] = self._e
        self._feed_dict[self._span_context] = self._span_ctxt
        self._feed_dict[self._answer_input] = self._answer_in
        self._feed_dict[self._answer_word_input] = self._answer_word_in
        self._feed_dict[self._answer_candidates] = self._answer_cands
        self._feed_dict[self._candidate_mask] = cand_mask
        self._feed_dict[self._query_ids] = self.queries_for_support
        self._feed_dict[self._support_ids] = self.support_for_queries
        self._feed_dict[self._collab_query_ids] = self._collab_queries
        self._feed_dict[self._collab_support_ids] = self._collab_support
        self._feed_dict[self._query_partition] = self._query_part

    def get_feed_dict(self):
        return self._feed_dict

    def step(self, sess, queries, mode="update"):
        '''
        :param sess:
        :param queries: list of ContextQueries
        :param mode: "loss" for loss, else performs update on parameters
        :return:
        '''
        assert self._is_train, "model has to be created in training mode!"
        if mode == "loss":
            return self.run(sess, self._loss, queries)
        else:
            return self.run(sess, [self._loss, self._update], queries)[0]

    def run(self, sess, to_run, queries):
        '''
        :param sess:
        :param to_run: target(s) to run, e.g. :
        * self.num_hops,
        * self.query,
        * self.score (only score for provided answers),
        * self.score_with_negs (scores of all candidates where score[0] is score of the answer)
        * self.input_embedding,
        * self.output_embedding,
        * self.support_scores (match-scores for all support with query for each hop. Aligns with
        self.supporting_qa which keeps track of all support QA-pairs and self.queries_for_support which
        defines batch_idx of query for all supporting_qa),,
        * self.answer_weights (weight used to accumulate retrieved answer in each hop)

                answer = self._retrieve_answer(self.query)
                self.score = tf_util.batch_dot(lookup_individual, answer)
                self.scores_with_negs = self._score_candidates(answer)

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.global_step = tf.Variable(0, trainable=False, name="step")

        :param queries: list of ContextQueries
        :return:
        '''
        batch_size = len(queries)
        self._start_adding_examples()
        num_batch_queries = 0
        for batch_idx in range(batch_size):
            context_query = queries[batch_idx]
            num_batch_queries += len(context_query.queries)
            self._add_example(context_query)
        self._finish_adding_examples()

        return sess.run(to_run, feed_dict=self.get_feed_dict())


def test_model():

    model = QAModel(10, 4, 5, 5, 5, max_hops=2)
    # 3 contexts (of length 3) with queries at 2/1/2 (totaling 5) positions
    # and respective negative candidates for each position
    contexts =       [[0, 1, 2]       , [1, 2, 0], [0, 2, 1]]  # 4 => placeholder for prediction position

    support = [ContextQueries(contexts[0], [ClozeQuery(contexts[0], 0,1,0,0,[2,1]),
                                            ClozeQuery(contexts[0], 2,3,2,2,[0,1])]),
               ContextQueries(contexts[1], [ClozeQuery(contexts[1], 1,2,2,2,[0,1])])]

    queries = [ContextQueries(contexts[0], [ClozeQuery(contexts[0], 0,1,0,0,[2,1], support=support),
                                            ClozeQuery(contexts[0], 2,3,2,2,[0,1], support=support)]),
               ContextQueries(contexts[1], [ClozeQuery(contexts[1], 1,2,2,2,[0,1], support=support)]),
               ContextQueries(contexts[2], [ClozeQuery(contexts[2], 1, 2, 1, 1, [0,2], support=support),
                                            ClozeQuery(contexts[2], 2,3,2,2,[0,1], support=support)])]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(model.num_hops.assign(1))
        print("Test update ...")
        for i in range(10):
            print("Loss: %.3f" %
                  model.step(sess, queries)[0])
        print("Test scoring ...")
        print(model.run(sess, model.scores_with_negs, queries))
        print("Done")


if __name__ == '__main__':
    test_model()


"""
Test update ...
Loss: 1.100
Loss: 1.085
Loss: 1.071
Loss: 1.054
Loss: 1.034
Loss: 1.010
Loss: 0.978
Loss: 0.941
Loss: 0.898
Loss: 0.852
Test scoring ...
[[-0.04599455  0.40225014 -0.24225023]
 [ 1.09410524 -0.54674953 -0.43304446]
 [ 0.82218146 -0.37236962 -0.33791351]
 [-0.31010905 -0.36932546  0.79180872]
 [ 1.00475466 -0.49434289 -0.38799521]]
Done
"""