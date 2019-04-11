from model import CPPN
import numpy as np
import argparse

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--x_dim", type=int, default=1280)
    parser.add_argument("--y_dim", type=int, default=720)
    parser.add_argument("--z_dim", type=int, default=12)
    parser.add_argument("--scale", type=int, default=8)

    parser.add_argument("--neurons_per_layer", type=int, default=12)
    parser.add_argument("--number_of_layers", type=int, default=7)
    parser.add_argument("--color_channels", type=int, default=3)

    parser.add_argument("--number_of_stills", type=int, default=4)
    parser.add_argument("--interpolations_per_image", type=int, default=12)
    parser.add_argument("--file_name", type=str, default='./videos/please_name_me.mp4')

    # Parse arguments
    args = parser.parse_args()

    return args

def main(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
         color_channels, number_of_stills, interpolations_per_image, file_name):

    print('Initializing CPPN...')
    cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                color_channels, interpolations_per_image)
    cppn.neural_net(True)

    print('Making latent(z) vectors')
    # Make a list of random latent vectors
    zs = [] # list of latent vectors
    for i in range(number_of_stills):
        zs.append(np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32))

    # Save to a video file
    print('Making Video!')
    cppn.save_mp4(zs, file_name)

    print('Done! The video is at %s' % file_name)

if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()
    # Run function
    main(args.x_dim, args.y_dim, args.z_dim, args.scale, args.neurons_per_layer,
         args.number_of_layers, args.color_channels, args.number_of_stills,
         args.interpolations_per_image, args.file_name)


