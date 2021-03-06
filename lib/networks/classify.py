import tensorflow as tf
from networks.network import Network

n_classes = 2

class ClassifyDomainTrain(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, 512])
        self.label = tf.placeholder(tf.int32, shape=[None])
        self.layers = dict({'data':self.data, 'label':self.label})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .fc(1024, name='fc6_d', trainable=self.trainable)
             .dropout(0.5, name='drop6_d')
             .fc(1024, name='fc7_d', trainable=self.trainable)
             .dropout(0.5, name='drop7_d')
             .fc(2, relu=False, name='conf_score', trainable=self.trainable)
             .softmax(name='conf_prob'))

class ClassifyDomainTest(Network):
    def __init__(self, trainable=False):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, 512])
        self.label = tf.placeholder(tf.int32, shape=[None])
        self.layers = dict({'data':self.data, 'label':self.label})
        self.trainable = trainable
        self.setup()

    def setup(self):
        with tf.variable_scope("", reuse=True):
            (self.feed('data')
                 .fc(1024, name='fc6_d', trainable=self.trainable)
                 .fc(1024, name='fc7_d', trainable=self.trainable)
                 .fc(2, relu=False, name='conf_score', trainable=self.trainable)
                 .softmax(name='conf_prob'))
