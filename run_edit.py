from model import CPPN
import numpy as np
import argparse
import pickle

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--x_dim", type=int, default=1280)
    parser.add_argument("--y_dim", type=int, default=720)
    parser.add_argument("--z_dim", type=int, default=3)
    parser.add_argument("--scale", type=int, default=8)

    parser.add_argument("--neurons_per_layer", type=int, default=20)
    parser.add_argument("--number_of_layers", type=int, default=3)
    parser.add_argument("--color_channels", type=int, default=1)

    parser.add_argument("--number_of_stills", type=int, default=3)
    parser.add_argument("--interpolations_per_image", type=int, default=6)
    parser.add_argument("--file_name", type=str, default='./videos/test01.mp4')

    # Parse arguments
    args = parser.parse_args()

    return args

def main(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
         color_channels, number_of_stills, interpolations_per_image, file_name):

    # Initialize CPPN with parameters #########################################
    print('Initializing CPPN...')
    cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                color_channels, interpolations_per_image)
    cppn.neural_net(True)
    ###########################################################################

    ### Save CPPN if you want to. Change the name if you don't want to override a
    ### previous model.
    # cppn.save_model('new_model')


    ### Load the correct CPPN based on the name you saved.
    cppn.load_model('new_model')

    ### Save single random image
    # z = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
    # cppn.save_png(z, file_name)

    # # Can additionally save latent vector with
    # with open('outfile', 'wb') as f: # 'outfile' can be renamed
    #     pickle.dump([z.tolist()], f)


    ### Make 100 random images and save their latent vectors
    # With a specific model loaded
    # z_vectors = []
    # for i in range(100):
    #     file_name = './photos/test%d.png' % i
    #     z_vectors = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
    #     cppn.save_png(z_vectors, file_name)
    #     zs.append(z.tolist())
    #     print('Done! The image is at %s' % file_name)
    # with open('outfile', 'wb') as f: # can change 'outfile'
    #     pickle.dump(zs, f)


    ### Re-display a specific image from a saved model.
    # With a specific model loaded
    # with open ('outfile', 'rb') as fp: # 'outfile' can be renamed
    #     reloaded_vectors = pickle.load(fp)
    # z = np.array(reloaded_vectors[10]) # would be named test10.png
    # cppn.save_png(z, file_name) # make sure you name this something you'll remember


    ### Wibble around a specific image and make a video from it.
    # # With a specific model loaded
    # with open ('outfile', 'rb') as fp: # 'outfile' can be renamed
    #     reloaded_vectors = pickle.load(fp)
    # z_start = np.array(reloaded_vectors[10]) # would be named test10.png
    # zs = []
    # for i in range(10): # how many 'key frames' you want
    #     z = np.random.uniform(-.1, .1, size=(z_dim)).astype(np.float32)
    #     z = np.add(z_start, z)
    #     zs.append(z)
    #     print('Done! The image is at %s' % file_name)
    # cppn.save_mp4(zs, file_name) # make sure this is named correctly


    ### Make a list of random video
    zs = [] # list of latent vectors
    for i in range(number_of_stills):
        zs.append(np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32))
    cppn.save_mp4(zs, file_name)


if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()
    # Run function
    main(args.x_dim, args.y_dim, args.z_dim, args.scale, args.neurons_per_layer,
         args.number_of_layers, args.color_channels, args.number_of_stills,
         args.interpolations_per_image, args.file_name)


