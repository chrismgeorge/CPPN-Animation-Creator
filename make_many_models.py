from model import CPPN
import numpy as np
import argparse
import pickle
import os

def makeKey(z, neurons, layers, t):
    return '%s-%s-%s-%s_' % (str(z), str(neurons), str(layers), str(t))

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    checkDirectory('./many_models/model_images/')
    checkDirectory('./many_models/models/')

    # Variables for different images
    x_dim = 1440
    y_dim = 1080

    color_channels = 1

    # Edit these to make specific types of models
    zs = list(range(3, 4))
    neurons = list(range(15, 20))
    layers = list(range(3, 5))

    tests = [0, 1, 2, 3, 4, 5] # the different types of networks
    scale = 24

    for z_dim in zs:
        for neurons_per_layer in neurons:
            for number_of_layers in layers:
                for t in tests:
                        # Make a new CPPN
                        interpolations_per_image = 1
                        cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer,
                                    number_of_layers, color_channels,
                                    interpolations_per_image, test=t)
                        cppn.neural_net(True)

                        dict_key = makeKey(z_dim, neurons_per_layer, number_of_layers, t)

                        # Save the model
                        cppn.save_model(model_name=dict_key, model_dir='many_models/models',
                                        save_outfile=True)

                        z = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)

                        # Save a test image
                        filename = "./many_models/model_images/%s.png" % (dict_key)
                        cppn.save_png(z, filename, save=True)

                        cppn.close()

                print('num/l', number_of_layers, 'neur/l',
                      neurons_per_layer, 'z', z_dim, 'all_tests completed')


if __name__ == '__main__':
    main()
