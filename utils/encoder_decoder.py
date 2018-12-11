import tensorflow as tf
import numpy as np
import time

class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, stride, kernel_shape, rand_seed, index=0):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, stride, stride, 1], padding="VALID")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

class deconv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, stride, kernel_shape, rand_seed, index=0):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('deconv_layer_%d' % index):
            with tf.name_scope('deconv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='deconv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('deconv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='deconv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            
            #output_shape = 
            conv_out = tf.nn.conv2d_transpose(input_x, weight, strides=[1, stride, stride, 1], padding="VALID")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('deconv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('deconv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x, is_training):
        """
        :param input_x: The input that needed for normalization.
        :param is_training: To control the training or inference phase
        """
        with tf.variable_scope('batch_norm'):
            batch_mean, batch_variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            ema = tf.train.ExponentialMovingAverage(decay=0.99)

            def True_fn():
                ema_op = ema.apply([batch_mean, batch_variance])
                with tf.control_dependencies([ema_op]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

            def False_fn():
                return ema.average(batch_mean), ema.average(batch_variance)

            mean, variance = tf.cond(is_training, True_fn, False_fn)

            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=None,
                                                 scale=None,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

def loss_mse(output, input_y):
    with tf.name_scope('loss_mse'):
        
        loss = tf.reduce_mean(tf.square(output-input_y))

    return loss

def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step

def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        error_num = tf.reduce_mean(tf.square(output-input_y))
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num

def encoder_decoder(input_x, input_y, is_training, seed=235):
    
    # define encoder network
    
    batch_num, img_len, img_len, channel_num = input_x.shape
    
    conv_layer_0 = conv_layer(input_x=input_x, in_channel=channel_num, out_channel= 8, stride = 2,
                              kernel_shape= 5, rand_seed=seed,index = 0)
    conv_layer_1 = conv_layer(input_x=conv_layer_0.output(), in_channel= 8, out_channel= 16, stride = 2,
                              kernel_shape= 5, rand_seed=seed,index = 1)
    conv_layer_2 = conv_layer(input_x=conv_layer_1.output(), in_channel= 16, out_channel= 32, stride = 2,
                              kernel_shape= 3, rand_seed=seed,index = 2)
    
    # define decoder network
    deconv_layer_0 = deconv_layer(input_x=conv_layer_2.output(), in_channel= 32, out_channel= 16, stride = 2,
                              kernel_shape= 2, rand_seed=seed,index = 0)
    
    deconv_layer_1 = deconv_layer(input_x=deconv_layer_0.output(), in_channel= 16, out_channel= 8, stride = 2,
                              kernel_shape= 4, rand_seed=seed, index = 1)
    
    deconv_layer_2 = deconv_layer(input_x=deconv_layer_1.output(), in_channel= 8, out_channel= 1, stride = 2,
                              kernel_shape= 4, rand_seed=seed,index = 2)
    
    
    # saving parameters for l2_norm loss
    conv_w = [conv_layer_0.weight, conv_layer_1.weight, conv_layer_2.weight,
              deconv_layer_0.weight, deconv_layer_1.weight, deconv_layer_2.weight]
    '''
    with tf.name_scope("loss"):
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])
        
        mse_loss = tf.reduce_mean(tf.square(deconv_layer_2.output()-input_y),name = "mse_loss")
        loss = tf.add(mse_loss, l2_norm * l2_loss, name='loss')
        
        tf.summary.scalar('encoder_decoder_loss', loss)
    '''   
    return deconv_layer_2.output()
    
                    
