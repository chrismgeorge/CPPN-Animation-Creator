import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import math

# neurons per layer
x_dim = 1080
y_dim = 720
z_dim = 12
color = 3
scale = 8

# tf graph input
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])
R = tf.placeholder(tf.float32, [None, 1])
Z = tf.placeholder(tf.float32, [z_dim])

# A hidden layer in a nueral network!
def layer(name, inputs, output_dim):
    shape = inputs.get_shape().as_list()
    with tf.variable_scope(name or "FC"):
        matrix = tf.get_variable("Matrix", [shape[1], output_dim], tf.float32,
                                 tf.random_normal_initializer(stddev=1.0))
        return tf.matmul(inputs, matrix)

def neural_net(x_dim, y_dim, z_dim, reuse=False, scale_=1.):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    n_points = x_dim * y_dim
    net_size = 12

    z_scaled = tf.reshape(Z, [1, z_dim]) * tf.ones([n_points, 1], dtype=tf.float32) * scale_
    z_unroll = tf.reshape(z_scaled, [n_points, z_dim])
    z_layer = layer('z', z_unroll, net_size)

    x_unroll = tf.reshape(X, [n_points, 1])
    x_layer = layer('x', x_unroll, net_size)

    y_unroll = tf.reshape(Y, [n_points, 1])
    y_layer = layer('y', y_unroll, net_size)

    r_unroll = tf.reshape(R, [n_points, 1])
    r_layer = layer('r', r_unroll, net_size)

    U = z_layer + x_layer + y_layer + r_layer

    H = tf.nn.tanh(U)

    # Customize The Outputs Here #
    sp_layer = layer('g_softplus_1', H, net_size)
    H = tf.nn.softplus(sp_layer)

    for i in range(7):
        H = layer('g_tanh_'+str(i), H, net_size)
        H = tf.nn.tanh(H)

    H = layer('g_final', H, color)

    output = 0.5 * tf.sin(H) + 0.5

    result = tf.reshape(output, [y_dim, x_dim, color])

    return result

# Get the correct coordinates.
def coordinates(x_dim, y_dim, scale_=1.):
    n_points = x_dim * y_dim
    x_range = scale_*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale_*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), 1).reshape(n_points, 1)
    y_mat = np.tile(y_mat.flatten(), 1).reshape(n_points, 1)
    r_mat = np.tile(r_mat.flatten(), 1).reshape(n_points, 1)
    return x_mat, y_mat, r_mat

def to_image(image_data):
    img_data = np.array(1-image_data)
    y_dim = image_data.shape[0]
    x_dim = image_data.shape[1]
    if color > 1:
      img_data = np.array(img_data.reshape((y_dim, x_dim, color))*255.0, dtype=np.uint8)
    else:
      img_data = np.array(img_data.reshape((y_dim, x_dim))*255.0, dtype=np.uint8)
    return img_data

def save_anim_gif(all_zs, filename, transitionCount=60, x_dim=1080, y_dim=720, scale_=10.0, loop=True):
    G = neural_net(x_dim, y_dim, z_dim, scale_=scale_, reuse=True)
    x_vec, y_vec, r_mat = coordinates(x_dim, y_dim, scale_=scale_)

    total_frames = transitionCount + 2
    images = []


    curSin = 0
    for index, z1 in enumerate(all_zs):
        if (index == len(all_zs) - 1):
            break
        else:
            z2 = all_zs[index+1]

        # delta_z = (z2-z1) / (transitionCount+1)

        for i in range(total_frames):
            # curFrameRatio = i / (total_frames)
            # multiplicationValue = (math.tanh(2*(curFrameRatio*2) - 2) + 1)/2

            # z = z1 + delta_z*float(i)

            t = i / total_frames / 2
            t = (1.0-np.cos(2.0*np.pi*t))/2.0       # looping & easing
            params = z1*(1.0-t) + z2*t      # blending
            params *= 1.0 + t*(1.0-t)

            images.append(to_image(sess.run(G, feed_dict={X: x_vec, Y: y_vec, R:r_mat, Z:params})))
            print("processing image ", i)

    if loop == True: # go backwards in time back to the first state
        revImages = list(images)
        revImages.reverse()
        revImages = revImages[1:]
        images = images+revImages

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(filename, fourcc, 24.0, (x_dim, y_dim))

    for im in images:
        video.write(im)
        print('write')

    cv2.destroyAllWindows()
    video.release()

    print("writing gif file...")

    #writeGif(filename, images, duration = durations)

G = neural_net(x_dim, y_dim, z_dim, scale_=scale)

# Initialize the variables (i.e. assign their default value)
# Needs to be called after all variables created!
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    z_count = 5
    zs = []
    for i in range(z_count):
        zs.append(np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32))

    save_anim_gif(zs, './first_video2.mp4', x_dim=x_dim, y_dim=y_dim, scale_=scale)

    # G = neural_net(x_dim, y_dim, z_dim, scale_=scale, reuse=True)
    # x_vec, y_vec, r_mat = coordinates(x_dim, y_dim, scale_=scale)
    # image = sess.run(G, feed_dict={X: x_vec, Y: y_vec, R:r_mat, Z:z1})
    # save_png(image, './test8.png')


