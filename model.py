import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import pickle
import random

class CPPN():
    def __init__(self, x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                 color_channels, interpolations_per_image, test=0):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.scale = scale
        self.test = test

        self.neurons_per_layer = neurons_per_layer
        self.number_of_layers = number_of_layers
        self.color_channels = color_channels
        self.interpolations_per_image = interpolations_per_image

        self.X = tf.placeholder(tf.float32, [None, 1])
        self.Y = tf.placeholder(tf.float32, [None, 1])
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.Z = tf.placeholder(tf.float32, [self.z_dim])

        self._ = self.neural_net() # call once

        # Initialize the variables
        self.init = tf.global_variables_initializer()

        # Start session
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.nn = None

        self.saver = tf.train.Saver(tf.all_variables())

        self.curImage = None
        self.scales = []
        self.times = []

    # A hidden layer in a nueral network
    def fully_connected_layer(self, name, inputs, output_dim):
        shape = inputs.get_shape().as_list()
        with tf.variable_scope(name):
            matrix = tf.get_variable("Matrix", [shape[1], output_dim], tf.float32,
                                     tf.random_normal_initializer(stddev=1.0))
            return tf.matmul(inputs, matrix)

    def neural_net(self, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        n_points = self.x_dim * self.y_dim

        z_scaled = tf.reshape(self.Z, [1, self.z_dim]) * tf.ones([n_points, 1],
                              dtype=tf.float32) * self.scale
        z_unroll = tf.reshape(z_scaled, [n_points, self.z_dim])
        z_layer = self.fully_connected_layer('z', z_unroll, self.neurons_per_layer)

        x_unroll = tf.reshape(self.X, [n_points, 1])
        x_layer = self.fully_connected_layer('x', x_unroll, self.neurons_per_layer)

        y_unroll = tf.reshape(self.Y, [n_points, 1])
        y_layer = self.fully_connected_layer('y', y_unroll, self.neurons_per_layer)

        r_unroll = tf.reshape(self.R, [n_points, 1])
        r_layer = self.fully_connected_layer('r', r_unroll, self.neurons_per_layer)

        U = z_layer + x_layer + y_layer + r_layer

        H = tf.nn.tanh(U)

        # Test 1
        if self.test == 0:
            sp_layer = self.fully_connected_layer('g_softplus_1', H, self.neurons_per_layer)
            H = tf.nn.softplus(sp_layer)

            for i in range(self.number_of_layers):
                H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
                H = tf.nn.tanh(H)

            H = self.fully_connected_layer('g_final', H, self.color_channels)
            output = 0.5 * tf.sin(H) + 0.5

        ## Test 2
        elif self.test == 1:
            for i in range(self.number_of_layers):
                H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
                H = tf.nn.tanh(H)

            H = self.fully_connected_layer('g_final', H, self.color_channels)
            output = tf.sigmoid(H)

        ## Test 3
        elif self.test == 2:
            for i in range(self.number_of_layers):
                H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
                H = tf.nn.tanh(H)
            H = self.fully_connected_layer('g_final', H, self.color_channels)
            output = tf.sqrt(1.0-tf.abs(tf.tanh(H)))


        # Test 4
        elif self.test == 3:
            for i in range(self.number_of_layers):
                H = self.fully_connected_layer('g_softplus_1'+str(i), H, self.neurons_per_layer)
                H = tf.nn.softplus(H)
                H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
                H = tf.nn.tanh(H)
            H = self.fully_connected_layer('g_softplus_2', H, self.color_channels)
            H = tf.nn.softplus(H)
            H = self.fully_connected_layer('g_final', H, self.color_channels)
            output = tf.sigmoid(H)

        # Test 5
        elif self.test == 4:
            for i in range(self.number_of_layers):
                H = self.fully_connected_layer('g_softplus_1'+str(i), H, self.neurons_per_layer)
                H = tf.nn.softplus(H)
                H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
                H = tf.nn.tanh(H)
            H = self.fully_connected_layer('g_softplus_2', H, self.color_channels)
            H = tf.nn.softplus(H)
            H = self.fully_connected_layer('g_final', H, self.color_channels)
            output = 0.5 * tf.sin(H) + 0.5

        # Test 6
        elif self.test == 5:
            for i in range(self.number_of_layers):
                H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
                H = H + tf.nn.tanh(H)

            H = self.fully_connected_layer('g_final', H, self.color_channels)
            output = tf.sigmoid(H)

        #####################################################################

        self.nn = tf.reshape(output, [self.y_dim, self.x_dim, self.color_channels])

    # Get the correct coordinates
    def coordinates(self):
        x_dim, y_dim = self.x_dim, self.y_dim
        N = np.mean((x_dim, y_dim))
        x = np.linspace(- x_dim / N * self.scale, x_dim / N * self.scale, x_dim)
        y = np.linspace(- y_dim / N * self.scale, y_dim / N * self.scale, y_dim)

        X, Y = np.meshgrid(x, y)

        x = np.ravel(X).reshape(-1, 1)
        y = np.ravel(Y).reshape(-1, 1)
        r = np.sqrt(x ** 2 + y ** 2)
        return x, y, r

    def to_image(self, image_data):
        img_data = np.array(1-image_data)
        if (self.color_channels > 1):
            img_data = np.array(img_data.reshape((self.y_dim, self.x_dim,
                                                  self.color_channels))*255.0,
                                                  dtype=np.uint8)
        else:
            img_data = np.array(img_data.reshape((self.y_dim,
                                                  self.x_dim))*255.0,
                                                  dtype=np.uint8)
            img_data = np.repeat(img_data[:, :, np.newaxis], 3, axis=2)

        return img_data

    def save_model(self, model_name='model.ckpt', epoch=0, model_dir='saved_models',
                  save_outfile=False):
        checkpoint_path = os.path.join(model_dir, model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)
        print('Saving the model! The model is at %s' % checkpoint_path)

        if save_outfile:
            data = [self.x_dim, self.y_dim, self.z_dim, self.scale,
                    self.neurons_per_layer, self.number_of_layers,
                    self.color_channels, self.test]
            outfile_name = checkpoint_path + 'meta_data'

            with open(outfile_name, 'wb') as f: # can change 'outfile'
                pickle.dump(data, f)
            print('Saved meta data')

    def load_model(self, model_name='model.ckpt', epoch=0,
                   model_dir='many_models/models'):
        self.saver.restore(self.sess, './%s/%s-%d' % (model_dir, model_name, epoch))
        print("Model loaded!")

    def save_png(self, params, filename, save=True, png=False):
        if png == False:
            x_vec, y_vec, r_mat = self.coordinates()
            im = self.to_image(self.sess.run(self.nn,
                               feed_dict={self.X: x_vec, self.Y: y_vec,
                                          self.R:r_mat, self.Z:params}))
            im = Image.fromarray(im)
        else:
            im = params
        self.curImage = im
        if save:
            im.save(filename)

    def save_mp4(self, all_zs, filename, loop=True, linear=False, scale_me=False,
                times=False, random_scale=False):

        images = []
        currentFrame = 0
        curScale = 24
        for i in range(len(all_zs)-1):
            if (times == True):
                self.interpolations_per_image = self.times[i]
            total_frames = self.interpolations_per_image + 2
            if (scale_me == True):
                s1 = self.scales[i]
                s2 = self.scales[i+1]
                delta_s = (s2-s1) / (self.interpolations_per_image + 1)
            # Get the current and next z vectors
            z1 = all_zs[i]
            z2 = all_zs[i+1]
            if (linear):
                delta_z = (z2-z1) / (self.interpolations_per_image + 1)
            if (random_scale == True):
                prevScale = curScale
                curScale = random.randint(5, 24)
                delta_s = (curScale-prevScale) / (self.interpolations_per_image + 1)

            for i in range(total_frames):
                if (scale_me == True):
                    self.scale = s1 + delta_s*i
                if (random_scale == True):
                    self.scale = prevScale + delta_s*i

                # Calculate looping like in the distill article
                if (linear):
                    z = z1 + delta_z*i
                else:
                    t = i / total_frames / 2
                    t = (1.0-np.cos(2.0*np.pi*t))/2.0
                    z = z1*(1.0-t) + z2*t
                    z *= (1.0 + t*(1.0-t))

                x_vec, y_vec, r_mat = self.coordinates()

                # Run the network, turn the output into an image and add it to
                # the image list, images
                image = self.to_image(self.sess.run(self.nn,
                                 feed_dict={self.X: x_vec, self.Y: y_vec,
                                            self.R:r_mat, self.Z:z}))
                images.append(image)
                # name = './photos/{num:0{width}}'.format(num=currentFrame, width=6)
                # self.save_png(image, name, png=True)
                currentFrame += 1

                print("processing image ", i)

        # Loop by adding a reverse of the list onto the images list
        if loop:
            revImages = list(images)
            revImages.reverse()
            revImages = revImages[1:]
            images = images+revImages

        # Make a mp4 of the image size at 24 fps
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(filename, fourcc, 24.0, (self.x_dim, self.y_dim))
        for im in images:
            video.write(im)
        cv2.destroyAllWindows()
        video.release()

    def close(self):
        tf.reset_default_graph()
        self.sess.close()
