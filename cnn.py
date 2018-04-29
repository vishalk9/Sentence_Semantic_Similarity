import tensorflow as tf
from utils import *

class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,
                 is_training,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=False,clip_gradients=5.0,decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        

        self.instantiate_weights()
        self.logits = self.inference()
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")

        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
        else:
            self.accuracy = tf.constant(0.5) 

    def instantiate_weights(self):
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])
            
    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss
               

class CNN_Layer():
    def __init__(self, num_classes, embedding_size, filter_sizes, num_filters, n_hidden,
                 input_x1, input_x2, input_y, dropout_keep_prob):
      
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self .num_filters = num_filters
        self.poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        self.input_x1 = input_x1
        self.input_x2 = input_x2
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob

        self.W1 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[0]], "W1_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[0]], "W1_1"),
                   init_weight([filter_sizes[2], embedding_size, 1, num_filters[0]], "W1_2")]
        self.b1 = [tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_1"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), "b1_2")]

        self.W2 = [init_weight([filter_sizes[0], embedding_size, 1, num_filters[1]], "W2_0"),
                   init_weight([filter_sizes[1], embedding_size, 1, num_filters[1]], "W2_1")]
        self.b2 = [tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_0"),
                   tf.Variable(tf.constant(0.1, shape=[num_filters[1], embedding_size]), "b2_1")]
        self.h = num_filters[0]*len(self.poolings)*2 + \
                 num_filters[1]*(len(self.poolings)-1)*(len(filter_sizes)-1)*3 #+ \
                 #len(self.poolings)*len(filter_sizes)*len(filter_sizes)*3
        self.Wh = tf.Variable(tf.random_normal([self.h, n_hidden], stddev=0.01), name='Wh')
        self.bh = tf.Variable(tf.constant(0.1, shape=[n_hidden]), name="bh")

        self.Wo = tf.Variable(tf.random_normal([n_hidden, num_classes], stddev=0.01), name='Wo')

    

    def per_dim_conv_layer(self, x, w, b, pooling):
        
        # unpcak the input in the dim of embed_dim
        input_unstack = tf.unstack(x, axis=2)
        w_unstack = tf.unstack(w, axis=1)
        b_unstack = tf.unstack(b, axis=1)
        convs = []
        for i in range(x.get_shape()[2]):
            conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])
            # [batch_size, sentence_length-ws+1, num_filters_A]
            convs.append(conv)
        conv = tf.stack(convs, axis=2)  # [batch_size, sentence_length-ws+1, embed_size, num_filters_A]
        pool = pooling(conv, axis=1)  # [batch_size, embed_size, num_filters_A]

        return pool

    def pool1(self, x):
        #bulid block A and cal the similarity according to algorithm 1
        out = []
        with tf.name_scope("bulid_block_A"):
            for pooling in self.poolings:
                pools = []
                for i, ws in enumerate(self.filter_sizes):
                    #print x.get_shape(), self.W1[i].get_shape()
                    with tf.name_scope("conv-pool-%s" %ws):
                        conv = tf.nn.conv2d(x, self.W1[i], strides=[1, 1, 1, 1], padding="VALID")
                        #print conv.get_shape()
                        conv = tf.nn.relu(conv + self.b1[i])  # [batch_size, sentence_length-ws+1, 1, num_filters_A]
                        pool = pooling(conv, axis=1)
                    pools.append(pool)
                out.append(pools)
            return out

    def pool2(self, x):
        out = []
        with tf.name_scope("pool2"):
            for pooling in self.poolings[:-1]:
                pools = []
                with tf.name_scope("conv-pool"):
                    for i, ws in enumerate(self.filter_sizes[:-1]):
                        with tf.name_scope("per_conv-pool-%s" % ws):
                            pool = self.per_dim_conv_layer(x, self.W2[i], self.b2[i], pooling)
                        pools.append(pool)
                    out.append(pools)
            return out

    def sentence_similarity_layer(self):
        sent1 = self.pool1(self.input_x1)
        sent2 = self.pool1(self.input_x2)
        fea_h = []
        with tf.name_scope("cal_dis_with_alg1"):
            for i in range(3):
                regM1 = tf.concat(sent1[i], 1)
                regM2 = tf.concat(sent2[i], 1)
                for k in range(self.num_filters[0]):
                    fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))


        fea_a = []
        with tf.name_scope("cal_dis_with_alg2_2-9"):
            for i in range(3):
                for j in range(len(self.filter_sizes)):
                    for k in range(len(self.filter_sizes)):
                        fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))

        sent1 = self.pool2(self.input_x1)
        sent2 = self.pool2(self.input_x2)

        fea_b = []
        with tf.name_scope("cal_dis_with_alg2_last"):
            for i in range(len(self.poolings)-1):
                for j in range(len(self.filter_sizes)-1):
                    for k in range(self.num_filters[1]):
                        fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
        return tf.concat(fea_h + fea_b, 1)

    def similarity_measure_layer(self):
        fea = self.sentence_similarity_layer()
   
        with tf.name_scope("full_connect_layer"):
            h = tf.nn.tanh(tf.matmul(fea, self.Wh) + self.bh)
            h = tf.nn.dropout(h, self.dropout_keep_prob)
            o = tf.matmul(h, self.Wo)
            return o

def init_weight(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)