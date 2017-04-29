# 3th party lib
import tensorflow as tf
import numpy as np
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)
import matplotlib.pyplot as plt
import csv

# OWN libraries
from foo_funcs import *

import os
os.chdir('/Users/Laurens_240/Documents/GitHub/astrohack/')


class Flag(object):
    """ Contains all settings """
    image_width = 64
    lr = 0.0
    bool_load = True
    iters = 100

    layer_size = [[64, 64], [30, 30], [13, 13], [5, 5], [1, 1]]
    kernel_size = [[5, 5], [5, 5], [4, 4], [5, 5]]
    kernel_depth = [1, 10, 10, 10, 1]
    layer_types = ['convstride', 'convstride', 'convstride', 'conv']


def open_info(name):
    sample_list = []

    with open(name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(reader, None)
        for row in reader:
            row_i = [row[0] , np.float32(row[1]), np.float32(row[2]), np.float32(row[3])]
            sample_list.append(row_i)

    # remove header
    return sample_list


def data_i(i_start, i_end):
    folder_info = 'C:/Users/Laurens_240/Desktop/astrohack/Data/Sample/'
    name_info = folder_info + 'sample.csv'

    image_folder = folder_info + 'Sample_Data/Sample_Data/SAMPLE/'

    # C:\Users\Laurens_240\Desktop\astrohack\Data\Sample\Sample_Data.zip\Sample_Data\SAMPLE

    info_list = open_info(name_info)

    n_inputs = len(info_list)

    for n_i in range(n_inputs):
        info_i = info_list[n_i]
        print(info_i)

        name_y = image_folder + info_i[0] +'-g.csv'

        a = open_csv(name_y)

        print(np.shape(a))

        1 / 0



    #
    #
    # folder = 'C:/Users/Laurens_240/Documents/GitHub/astrohack/crop_0/'
    #
    # folder2 = 'C:/Users/Laurens_240/Documents/GitHub/astrohack/crop_0/'
    #
    # x = []
    # y = []
    # for im_i in range(i_start, i_end):
    #     name_x = folder + 'cropped_{}.csv'.format(im_i)
    #     name_y = folder + 'info_{}.csv'.format(im_i)
    #
    #     x.append(open_csv(name_x))
    #
    #     y_i = open_csv(name_y)
    #
    #     y.append(y_i)
    #
    # # 1/0
    #
    # x = np.asarray(x)
    # y = np.asarray(y)
    #
    # # n_batch, height, width, depth
    # batch_X = np.reshape(x, (-1, 64, 64, 1))
    # # n_batch, 1, 1, parameters to guess
    # batch_Y = np.reshape(y, (-1, 1, 1, 4))
    #
    # return batch_X, batch_Y

def train_data():
    return data_i(0, 40)


def test_data():
    n_images = 53
    return data_i(40, n_images)


def leaky_relu(x, alpha=0., max_value=None):
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, 0., max_value)
    x -= tf.constant(alpha) * negative_part
    return x


class Network(object):
    def __init__(self, flag):
        self.flag = flag

    def build_Y(self, X):

        n_layers = len(self.flag.layer_types)

        # DON't scale to log, background is not 0
        # in_pointer = tf.log(tf.maximum(X, 1.0e-10))
        in_pointer = X
        out_pointer = None

        self.params  = {}

        for ind_layer in range(n_layers):

            layer_type = self.flag.layer_types[ind_layer]
            layer_size = self.flag.layer_size[ind_layer]
            kernel_size = self.flag.kernel_size[ind_layer]
            kernel_depth_in = self.flag.kernel_depth[ind_layer]
            kernel_depth_out = self.flag.kernel_depth[ind_layer + 1]

            #weight init
            shape = [kernel_size[0], kernel_size[1], kernel_depth_in, kernel_depth_out]
            stddev = np.sqrt(2 / (shape[0] * shape[1] * shape[2] +
                                  shape[0] * shape[1] * shape[3]))
            initial = tf.truncated_normal(shape, stddev=stddev)
            W = tf.Variable(initial,
                            name='W_{}'.format(ind_layer))

            shape = [1, 1, 1, kernel_depth_out]
            stddev = 0
            initial = tf.truncated_normal(shape, stddev = stddev)
            b = tf.Variable(initial, name='b_{}'.format(ind_layer))

            self.params.update({W.name : W, b.name : b})

            def conv_layer(strides):
                z_i = tf.nn.conv2d(in_pointer, W, strides=strides, padding='VALID') + b
                # y_i = tf.nn.relu(z_i)
                #todo leaky relu
                y_i = leaky_relu(z_i, alpha=0.001)
                return y_i

            if layer_type == 'conv':
                strides = [1, 1, 1, 1]
                out_pointer = conv_layer(strides)
                in_pointer = out_pointer
            elif layer_type == 'convstride':
                strides = [1, 2, 2, 1]
                out_pointer = conv_layer(strides)
                in_pointer = out_pointer

            else:
                #todo
                1/0



        # # flatten the images into a single line of pixels
        # # -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
        # X_flat = tf.reshape(X, [-1, 64 * 64])
        #
        #
        #
        # # The model
        # z1 = tf.matmul(X_flat, W) + b
        #
        # # TODO add a non-linear function
        # def non_lin(z):
        #     # do something...
        #     return z
        #
        # Y = tf.reshape(non_lin(z1), [-1, 1, 1, 1])

        self.y_guess = out_pointer

        self.saver = tf.train.Saver(self.params)

    def get_Y(self):
        return self.y_guess


def main():
    # neural network with 1 layer of 10 softmax neurons
    #
    # · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
    # \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
    #   · · · · · · · ·                                              M_guess [batch, 10]

    # The model is:
    #
    # M_guess = softmax( X * W + b)
    #              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
    #              W: weight matrix with 784 lines and 10 columns
    #              b: bias vector with 10 dimensions
    #              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
    #              softmax(matrix) applies softmax on each line
    #              softmax(line) applies an exp to each value then divides by the norm of the resulting line
    #              M_guess: output matrix with 100 lines and 10 columns

    flag = Flag()

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, flag.image_width, flag.image_width, 1], name = 'X')
    d_place = tf.placeholder(tf.float32, [None, 1, 1, 1], name = 'd')

    net = Network(flag)
    net.build_Y(X)

    Y = net.get_Y()

    x_fc = tf.concat([Y, tf.log(d_place)], axis= 3)

    def foo():
        # weight init
        shape = [1, 1, 2, 1]
        stddev = np.sqrt(2 / (shape[0] * shape[1] * shape[2] +
                              shape[0] * shape[1] * shape[3]))
        initial = tf.truncated_normal(shape, stddev=stddev)
        W = tf.Variable(initial,
                        name='W_extra')

        shape = [1, 1, 1, 1]
        stddev = 0
        initial = tf.truncated_normal(shape, stddev=stddev)
        b = tf.Variable(initial, name='b_extra')


        # todo update params
        net.params.update({W.name: W, b.name: b})

        strides = [1,1,1,1]
        z_i = tf.nn.conv2d(x_fc, W, strides=strides, padding='VALID') + b

        return z_i



    M_guess = foo()
    net.saver = tf.train.Saver(net.params)

    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 1, 1, 1], name = 'Y_truth')
    errM_ = tf.placeholder(tf.float32, [None, 1, 1, 1], name='M_rms')



    # loss function: cross-entropy = - sum( Y_i * log(Yi) )
    #                           M_guess: the computed output vector
    #                           Y_: the desired output vector

    ln10 = 2.30258509299

    # cost = tf.reduce_mean(tf.divide(tf.squared_difference(tf.pow(10.0, M_guess), tf.pow(10.0, Y_), name = 'squar_diff'),
    #                                 tf.square( tf.pow(10.0, Y_)*ln10*errM_)), name= 'cost')

    # todo switch between
    # cost = tf.reduce_mean(tf.divide(tf.squared_difference(1.0, tf.pow(10.0, Y_ - M_guess)),
    #                                 tf.square(ln10 * errM_)), name='cost')
    # todo remove 11!
    cost = tf.reduce_mean(tf.divide(tf.squared_difference(1.0, tf.pow(10.0, Y_ - M_guess)),
                                                          tf.square(ln10 * errM_)), name='cost')
    cost = tf.log(cost)

    # cost = tf.reduce_mean(tf.squared_difference(Y_, M_guess))




    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    # correct_prediction = tf.equal(tf.argmax(M_guess, 1), tf.argmax(Y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training, learning rate = 0.005
    # train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(flag.lr).minimize(cost)

    # matplotlib visualisation
    # allweights = tf.reshape(W, [-1])
    # allbiases = tf.reshape(b, [-1])
    # I = tensorflowvisu.tf_format_mnist_images(X, M_guess, Y_)  # assembles 10x10 images by default
    # It = tensorflowvisu.tf_format_mnist_images(X, M_guess, Y_, 1000, lines=25)  # 1000 images on 25 lines
    # datavis = tensorflowvisu.MnistDataVis()

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # TODO own x/y
    batch_X, batch_Y = train_data()
    batch_X_te, batch_Y_te = test_data()
    # You can call this function in a loop to train the model, 100 images at a time

    c_t = []
    c_t_test = []

    def info_foo(x_in, y_in, errM, d):

        [c, y] = sess.run([cost, M_guess], feed_dict={X: x_in, Y_: y_in, errM_: errM, d_place: d})

        y_mean_guess = np.mean(y)
        y_mean = np.mean(y_in)

        print('M_true {} M_guess {}'.format(y_mean, y_mean_guess))

        print(str(i) + ": loss: " + str(c))

        return c

    def training_step(i, update_test_data, update_train_data):

        # training on batches of 100 images with 100 labels

        # compute training values for visualisation
        if update_train_data:
            print('train:')

            batch_Y_mass = batch_Y[:, :, :, 1:2]

            errM = batch_Y[:, :, :, 2:3]

            d= batch_Y[:, :, :, 3:4]

            c = info_foo(batch_X, batch_Y_mass, errM, d)
            c_t.append(c)

        if update_test_data:
            print('test:')

            batch_Y_mass_te = batch_Y_te[:, :, :, 1:2]
            errM_te = batch_Y_te[:, :, :, 2:3]

            d_te = batch_Y_te[:, :, :, 3:4]

            c = info_foo(batch_X_te, batch_Y_mass_te, errM_te, d_te)

            c_t_test.append(c)

        # the backpropagation training step
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y_mass, errM_: errM, d_place: d})



    # folder_weights = "C:\Users\Laurens_240\Documents\GitHub\astrohack\cnn\tmp"
    folder_weights = "cnn/tmp/"

    if flag.bool_load:
        # net.saver.restore(sess, )
        ckpt = tf.train.get_checkpoint_state(folder_weights)
        print(ckpt.model_checkpoint_path)
        net.saver.restore(sess, ckpt.model_checkpoint_path)

    for i in range(flag.iters):
        training_step(i, update_test_data = True, update_train_data = True)

        # if i%50 == 0:
        #     save_path = net.saver.save(sess, folder_weights+"model.ckpt", global_step=i)
        #     print("Model saved in file: %s" % save_path)

    save_path = net.saver.save(sess, folder_weights + "model.ckpt", global_step=i)
    print("Model saved in file: %s" % save_path)

    # datavis.animate(training_step, iterations=2000 + 1, train_data_update_freq=10, test_data_update_freq=50,
    #                 more_tests_at_start=True)

    # to save the animation as a movie, add save_movie=True as an argument to datavis.animate
    # to disable the visualisation use the following line instead of the datavis.animate line
    # for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

    # print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

    # final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.

    plt.plot(c_t)
    plt.plot(c_t_test)
    plt.ylabel('cost')
    plt.xlabel('epoch')

    plt.show()

if __name__ == '__main__':
    main()