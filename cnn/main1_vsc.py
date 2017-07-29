# 3th party lib
import tensorflow as tf
import numpy as np
# import tensorflowvisu
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)
import matplotlib.pyplot as plt
import csv

# OWN libraries
from foo_funcs import *

#import os
#os.chdir('/Users/Laurens_240/Documents/GitHub/astrohack/')


class Flag(object):
    """ Contains all settings """
    image_width = 64
    lr = 1.0e-5
    bool_load = True
    iters = 1000 #100000
    iters_per_batch = 1000 # can be high
    n_batch = 1000 # size of each batch


    layer_size = [[64, 64], [30, 30], [13, 13], [1, 1]]
    kernel_size = [[5, 5], [5, 5], [13, 13]]
    kernel_depth = [1, 20, 10, 5]
    layer_types = ['convstride', 'convstride', 'conv']


def open_info(name):
    sample_list = []

    with open(name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(reader, None)
        for row in reader:
            if np.float32(row[1]) > -50:
                row_i = [row[0] , np.float32(row[1]), np.float32(row[2]), np.float32(row[3])]
                sample_list.append(row_i)

    # remove header
    return sample_list

def open_info2(name):
    sample_list = []

    with open(name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(reader, None)
        for row in reader:
            row_i = [row[0] , np.float32(row[1])]
            sample_list.append(row_i)


    return sample_list

def test_i(n_inputs = False):
    folder_info = '/scratch/leuven/sys/ASTROHACK_DATA/'
    name_info = folder_info + 'Test_Distance.csv'
    image_folder = folder_info + 'Test/'


    info_list = open_info2(name_info)



    if n_inputs:
        ...
    else:
        n_inputs = len(info_list)

    x = []
    dist = []
    frac = []

    sess = tf.Session()

    id = []

    for n_i in range( n_inputs):
        info_i = info_list[n_i]
        print(info_i)

        name_y = image_folder + info_i[0] +'-g.csv'

        im_i = open_csv(name_y)

        shape_i = np.shape(im_i)

        frac.append(64 / (shape_i[0]))

        # todo bicubic
        x_i = tf.image.resize_bilinear(np.resize(im_i, new_shape=[1, shape_i[0], shape_i[1], 1]), [64, 64])

        x.append(np.asarray(sess.run(x_i)))

        dist.append(info_i[1])

        id.append(info_i[0])

    def flat_reshape(a):
        return np.reshape(np.asarray(a), newshape=(-1, 1, 1, 1))

    x = np.reshape(np.asarray(x), newshape=(-1, 64, 64, 1))
    dist = flat_reshape(dist)
    frac = flat_reshape(frac)

    return x, dist, frac, id

def data_i(i_start, i_end):
    folder_info = '/scratch/leuven/sys/ASTROHACK_DATA/'
    # folder_info = 'C:/Users/Laurens_240/Desktop/astrohack/Data/Sample/'
    name_info = folder_info + 'Train.csv'

    # image_folder = folder_info + 'Sample_Data/Sample_Data/SAMPLE/'
    image_folder = folder_info + 'Train/'


    # C:\Users\Laurens_240\Desktop\astrohack\Data\Sample\Sample_Data.zip\Sample_Data\SAMPLE

    info_list = open_info(name_info)



    n_inputs = len(info_list) #74891 vs 76...


    x = []
    y = []
    dist = []
    err = []
    frac = []

    sess = tf.Session()

    for n_i in range(i_start, i_end):
        info_i = info_list[n_i]
        print(info_i)

        name_y = image_folder + info_i[0] +'-g.csv'

        im_i = open_csv(name_y)

        shape_i = np.shape(im_i)

        frac.append(64/(shape_i[0]))

        # todo bicubic
        x_i = tf.image.resize_bilinear(np.resize(im_i, new_shape=[1, shape_i[0], shape_i[1], 1 ]), [64, 64])

        x.append(np.asarray(sess.run(x_i)))

        y.append(info_i[1])

        dist.append(info_i[3])
        err.append(info_i[2])





        # 1 / 0

    # x =

    #
    #
    # folder = 'C:/Users/Laurens_240/Documents/GitHub/astrohack/crop_0/'
    #
    # folder2 = 'C:/Users/Laurens_240/Documents/GitHub/astrohack/crop_0/'
    #

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

    def flat_reshape(a):
        return np.reshape(np.asarray(a),newshape=(-1, 1, 1, 1))


    x = np.reshape(np.asarray(x), newshape=(-1, 64, 64, 1))
    y = flat_reshape(y)
    dist = flat_reshape(dist)
    err = flat_reshape(err)
    frac =flat_reshape(frac)

    return x, y, dist, err, frac


def train_data():
    return data_i(0, 400)


def test_data():
    n_images = 53
    return data_i(500, 1500)


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

        self.keep_prob = tf.placeholder_with_default(input=1.0, shape=( ), name='keep_prob')

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
                return tf.nn.dropout(y_i, keep_prob= self.keep_prob)

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
    frac_place = tf.placeholder(tf.float32, [None, 1, 1, 1], name='wheight')

    net = Network(flag)
    net.build_Y(X)

    Y = net.get_Y()

    x_fc = tf.concat([Y, tf.log(d_place), tf.log(frac_place)], axis= 3)

    def foo():
        # weight init
        shape = [1, 1, 7, 1]
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


    # cost = tf.reduce_mean(tf.square( tf.divide(1.0 - tf.pow(10.0, Y_ - M_guess),
    #                                                       ln10 * errM_) ), name='cost')
    # cost = tf.log(cost)

    cost = tf.reduce_mean(tf.divide(tf.squared_difference(M_guess, Y_), errM_))

    regularizer2= 0
    for param in net.params:
        if param[0] == 'W':
            # regularizer1 = regularizer1 + tflearn.losses.L1(self.params[param])
            regularizer2 = regularizer2 + tf.reduce_mean(tf.square(net.params[param]))

    # beta1 = 1.0e-5
    beta2 = 1.0e-1

    cost_mean_list = (cost, beta2 * regularizer2)
    cost_tot = cost_mean_list[0] + cost_mean_list[1]

    # cost = tf.reduce_mean(tf.squared_difference(Y_, M_guess))




    # # accuracy of the trained model, between 0 (worst) and 1 (best)
    # correct_prediction = tf.equal(tf.argmax(M_guess, 1), tf.argmax(Y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    train_step = tf.train.AdamOptimizer(flag.lr).minimize(cost_tot)

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


    # c_t = []
    # c_t_test = []

    def info_foo(feed_dict):

        [c, y] = sess.run([cost_mean_list, M_guess], feed_dict=feed_dict)

        y_mean_guess = np.mean(y)
        y_mean = np.mean(feed_dict[Y_])

        print('M_true {} M_guess {}'.format(y_mean, y_mean_guess))

        print(str(i) + ": loss: " + str(c))

        return c

    def training_step(i, update_test_data, update_train_data):

        # training on batches of 100 images with 100 labels

        # compute training values for visualisation
        if update_train_data:
            print('train:')

            # batch_Y_mass = batch_Y[:, :, :, 1:2]
            #
            # errM = batch_Y[:, :, :, 2:3]
            #
            # d= batch_Y[:, :, :, 3:4]


            feed_dict = {X: batch_X, Y_: batch_Y, errM_: batch_err,
                         d_place: batch_dist, frac_place: batch_frac}

            c = info_foo(feed_dict)
            # c_t.append(c)

        if update_test_data:
            print('test:')

            # batch_Y_mass_te = batch_Y_te[:, :, :, 1:2]
            # errM_te = batch_Y_te[:, :, :, 2:3]
            #
            # d_te = batch_Y_te[:, :, :, 3:4]
            #
            # c = info_foo(batch_X_te, batch_Y_mass_te, errM_te, d_te)

            feed_dict_te = {X: batch_X_te, Y_: batch_Y_te, errM_: batch_err_te,
                         d_place: batch_dist_te, frac_place: batch_frac_te}

            c = info_foo(feed_dict_te)

            # c_t_test.append(c)

        # the backpropagation training step
        feed_dict.update({net.keep_prob: 0.9})
        sess.run(train_step, feed_dict=feed_dict)


    # folder_weights = "C:\Users\Laurens_240\Documents\GitHub\astrohack\cnn\tmp"
    folder_weights = "tmp/"

    if flag.bool_load:
        # net.saver.restore(sess, )
        ckpt = tf.train.get_checkpoint_state(folder_weights)
        print(ckpt.model_checkpoint_path)
        net.saver.restore(sess, ckpt.model_checkpoint_path)

    batch_X_te, batch_Y_te, batch_dist_te, batch_err_te, batch_frac_te = test_data()
    n_tot = 74891
    for i in range(flag.iters):


        n_batch = flag.n_batch

        n_sub_iters = int(np.floor(n_tot/n_batch))

        #n_sub_iters = 1
        for batch_i in range(n_sub_iters):
            # TODO own x/y
            # x, y, dist, err, frac

            batch_X, batch_Y, batch_dist, batch_err, batch_frac = \
                data_i(batch_i * n_batch, (batch_i+1) * n_batch)

             # train_data()
            # batch_X_te, batch_Y_te = test_data()
            # You can call this function in a loop to train the model, 100 images at a time

            ipb = flag.iters_per_batch
            for ii in range(ipb):
                training_step(i * n_sub_iters * ipb + batch_i * ipb + ii,
                              update_test_data = True, update_train_data = True)

            # if i%50 == 0:
            save_path = net.saver.save(sess, folder_weights+"model.ckpt", global_step=i * n_sub_iters + batch_i)
            print("Model saved in file: %s" % save_path)

    save_path = net.saver.save(sess, folder_weights + "model.ckpt", global_step=i)
    print("Model saved in file: %s" % save_path)

    x, dist, frac, id = test_i(10)

    feed_dict_out = {X: x, d_place:dist, frac_place: frac}


    # the backpropagation training step


    m_guess = sess.run(M_guess, feed_dict=feed_dict_out)

    out = zip(id, np.reshape(m_guess, newshape=(-1,)))

    writer = csv.writer(open('m_guess.csv', 'w'))
    writer.writerow('pssid, mass')
    for row in out:
        writer.writerow(row)




    # datavis.animate(training_step, iterations=2000 + 1, train_data_update_freq=10, test_data_update_freq=50,
    #                 more_tests_at_start=True)

    # to save the animation as a movie, add save_movie=True as an argument to datavis.animate
    # to disable the visualisation use the following line instead of the datavis.animate line
    # for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

    # print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

    # final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.

    # plt.plot(c_t)
    # plt.plot(c_t_test)
    # plt.ylabel('cost')
    # plt.xlabel('epoch')
    #
    # plt.show()

if __name__ == '__main__':
    main()