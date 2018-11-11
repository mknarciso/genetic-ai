"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: r1.2
"""

# Smaller hidden layer

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
            learning_rate=0.0001,
            reward_decay=0.995,
            e_greedy=0.98,
            replace_target_iter=4*20, ## Troca a rede a cada 4 episódios
            memory_size=400*episode_size,
            batch_size=4*episode_size, ## Cara treinameinto usa 2 episódios
            e_greedy_start=0,
            e_greedy_increment=None,
            output_graph=False,
            training=False,
            import_file=None,
            save_file='saved/trained_dqn',
            mem_file=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_start if e_greedy_increment is not None else self.epsilon_max
        self.training = training
        self.save_file = save_file

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
            #import code; code.interact(local=dict(globals(), **locals()))
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

        
        self.saver = tf.train.Saver()   

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        if import_file is not None:
            self.saver.restore(self.sess, import_file)

        #if mem_file is not None:
            #import code; code.interact(local=dict(globals(), **locals()))
            #mem_range = int(self.memory[~np.all(self.memory == 0, axis=1)].shape[0]/48)
            #for c in range(mem_range):
            #    print("Learnin from memory "+str(c)+" of "+ str(mem_range))
            #    self.learn()
            #    print("done")

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
 
        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):

            e1 = tf.layers.dense(self.s, 8, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            eee1 = tf.layers.dense(e1, 8, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='eee1')
            self.q_eval = tf.layers.dense(eee1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 8, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            ttt1 = tf.layers.dense(t1, 8, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='ttt1')
            self.q_next = tf.layers.dense(ttt1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        #if not hasattr(self, 'memory_counter'):
        #    self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1


    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            #import code; code.interact(local=dict(globals(), **locals()))
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('[epsilon: '+str("%6.4f" % self.epsilon)+'][Target params replaced!]')
            #a = np.array([1, 2, 3, 4])
            #np.savetxt('test1.txt', a, fmt='%d')
            #b = np.loadtxt('test1.txt', dtype=int)
            #self.memory.tofile('C:/bvr_ai/memory/dqn5.dat')
            #c = np.fromfile('test2.dat', dtype=int)

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        batch_memory = self.memory[sample_index, :]
        #import code; code.interact(local=dict(globals(), **locals()))
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
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        # save model
        if self.training:
            self.saver.save(self.sess, self.save_file)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

#if __name__ == '__main__':
#    DQN = DeepQNetwork(3,4, output_graph=True)