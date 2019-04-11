import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

class CPPN():
    def __init__(self, x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                 color_channels, interpolations_per_image):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.scale = scale
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

        # Customize The Outputs Here ########################################
        sp_layer = self.fully_connected_layer('g_softplus_1', H, self.neurons_per_layer)
        H = tf.nn.softplus(sp_layer)

        for i in range(self.number_of_layers):
            H = self.fully_connected_layer('g_tanh_'+str(i), H, self.neurons_per_layer)
            H = tf.nn.tanh(H)

        H = self.fully_connected_layer('g_final', H, self.color_channels)

        output = 0.5 * tf.sin(H) + 0.5
        #####################################################################

        self.nn = tf.reshape(output, [self.y_dim, self.x_dim, self.color_channels])

    # Get the correct coordinates
    def coordinates(self):
        x_dim, y_dim, z_dim, scale = self.x_dim, self.y_dim, self.z_dim, self.scale
        n_points = x_dim * y_dim
        x_range = scale*(np.arange(self.x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
        y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
        x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
        y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
        r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
        x_mat = np.tile(x_mat.flatten(), 1).reshape(n_points, 1)
        y_mat = np.tile(y_mat.flatten(), 1).reshape(n_points, 1)
        r_mat = np.tile(r_mat.flatten(), 1).reshape(n_points, 1)
        return x_mat, y_mat, r_mat

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

    def save_model(self, model_name='model.ckpt', epoch=0):
        checkpoint_path = os.path.join('saved_models', model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)
        print('Saving the model! The model is at %s' % checkpoint_path)

    def load_model(self, model_name='model.ckpt', epoch=0):
        self.saver.restore(self.sess, './saved_models/%s-%d' % (model_name, epoch))
        print("Model loaded!")

    def save_png(self, params, filename):
        x_vec, y_vec, r_mat = self.coordinates()
        im = self.to_image(self.sess.run(self.nn,
                           feed_dict={self.X: x_vec, self.Y: y_vec,
                                      self.R:r_mat, self.Z:params}))
        im = Image.fromarray(im)
        im.save(filename)

    def save_mp4(self, all_zs, filename):
        x_vec, y_vec, r_mat = self.coordinates()

        total_frames = self.interpolations_per_image + 2
        images = []

        for i in range(len(all_zs)-1):
            # Get the current and next z vectors
            z1 = all_zs[i]
            z2 = all_zs[i+1]
            for i in range(total_frames):
                # Calculate looping like in the distill article
                t = i / total_frames / 2
                t = (1.0-np.cos(2.0*np.pi*t))/2.0
                params = z1*(1.0-t) + z2*t
                params *= 1.0 + t*(1.0-t)

                # Run the network, turn the output into an image and add it to
                # the image list, images
                image = self.to_image(self.sess.run(self.nn,
                                 feed_dict={self.X: x_vec, self.Y: y_vec,
                                            self.R:r_mat, self.Z:params}))
                images.append(image)

                print("processing image ", i)

        # Loop by adding a reverse of the list onto the images list
        # revImages = list(images)
        # revImages.reverse()
        # revImages = revImages[1:]
        # images = images+revImages

        # Make a mp4 of the image size at 24 fps
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(filename, fourcc, 24.0, (self.x_dim, self.y_dim))
        for im in images:
            video.write(im)
        cv2.destroyAllWindows()
        video.release()
