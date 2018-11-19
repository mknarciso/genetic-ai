"""
Deep Q Network (DQN) implementation.
Based on: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5_Deep_Q_Network/DQN_modified.py
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

episode_size = 479

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.0000001,
            reward_decay=0.998,
            e_greedy=0.98,
            replace_target_iter=4*20,       # Replace network wights each 4 episodes
            memory_size=100*episode_size,   # Memory capacity
            permanent_memory_size=0,        # Memory that is not cleared
            batch_size=4*episode_size,      # Episodes used in each train batch
            e_greedy_start=0,               # Offset greedy start
            e_greedy_increment=None,        # Epsilon increase step
            e_exp_decay=None,               # Epsilon exponential decay rate
            training=False,                 # Train the networks or keep weights static
            import_file=None,               # Import pre treined network
            save_file='saved/trained_dqn',  # Save network weights at the end
            mem_file=None,                  # Preload memory file name
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.permanent_memory_size = permanent_memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_start if e_greedy_increment is not None else self.epsilon_max
        self.training = training
        self.save_file = save_file
        self.exp_eps_decay = e_exp_decay

        if e_exp_decay is not None:
            self.epsilon = 0 

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        each_batch_size = n_features * 2 + 2
        if mem_file is None:
            self.memory = np.zeros((self.memory_size, each_batch_size))
            self.memory_counter = 0
        else:
            self.memory = np.zeros((self.memory_size, each_batch_size))
            pre_memory = np.fromfile(mem_file, dtype=float)
            pre_memory = pre_memory.reshape(int(pre_memory.shape[0]/each_batch_size), each_batch_size)
            zlines = pre_memory[np.all(pre_memory == 0, axis=1)].shape[0]
            self.memory[:pre_memory.shape[0],:pre_memory.shape[1]] = pre_memory
            self.memory_counter = pre_memory.shape[0] - zlines

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # Initialize TF
        self.saver = tf.train.Saver()   
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        if import_file is not None:
            self.saver.restore(self.sess, import_file)

    def _build_net(self):
        # ------------------ all inputs ------------------------
        # input State
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # input Next State      
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        # input Reward    
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        # input Action                     
        self.a = tf.placeholder(tf.int32, [None, ], name='a')                       

        # Weights random initializers
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
 
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 10, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 10, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        # Target Action
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')
            self.q_target = tf.stop_gradient(q_target)

        # Eval Action
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    

        # Calculate MSE
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        # Train the network with AdamOptimizer
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        # Organize the tuple
        transition = np.hstack((s, [a, r], s_))
        # Save to memory, replacing old ones if needed
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.permanent_memory_size > 0 and (self.memory_counter % self.memory_size) < self.permanent_memory_size:
            self.memory_counter += self.permanent_memory_size


    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # greedy select the action
            action = np.argmax(actions_value)
        else:
            # random action
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            #print('[epsilon: '+str("%6.4f" % self.epsilon)+'][Target params replaced!]')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        # Train 
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        if self.exp_eps_decay is None:
            if self.epsilon < self.epsilon_max:
                self.epsilon = self.epsilon + self.epsilon_increment  
            else: 
                self.epsilon = self.epsilon_max
        else:
            self.epsilon = 1 - np.power((1/(1+self.exp_eps_decay)),self.learn_step_counter)
        self.learn_step_counter += 1

        # save model
        if self.training:
            self.saver.save(self.sess, self.save_file)
