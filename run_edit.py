from model import CPPN
import numpy as np
import argparse
import pickle
import random
import os

from run_edit_ops import make_random_video, make_gui_video, getMetaData

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--ml", type=str, default=None) # the model name
    parser.add_argument("--fc", type=str, default=None) # the function to be used
    parser.add_argument("--ofs", type=str, default=None) # outfiles, comma seperated
    parser.add_argument("--fn", type=str, default=None) # the file name to be called

    # Parse arguments
    args = parser.parse_args()

    return args

def main(model_name, function, outfiles=None, file_name=None):

    # Load a saved CPPN and set variables
    meta_data = getMetaData(model_name)

    (x_dim, y_dim, z_dim, scale, neurons_per_layer,
     number_of_layers, color_channels, test) = meta_data
    interpolations_per_image = 24

    # Initialize CPPN with parameters #########################################
    print('Initializing CPPN...')
    cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                color_channels, interpolations_per_image, test=test)
    cppn.neural_net(True)
    cppn.load_model(model_name=model_name)
    ###########################################################################

    if (function == 'make_random_video'):
        if (file_name != None):
            make_random_video(cppn, file_name, z_dim, number_of_stills=5)
        else:
            print('Please specify a file name. Visit the readme.')
    elif (function == 'make_gui_video'):
        if (outfiles != None and file_name != None):
            outfiles = outfiles.split(',')
            make_gui_video(cppn, outfiles, file_name)
        else:
            print('Please specify an outfile and a file name. Visit the readme.')
    else:
        print('Please enter a valid function. Visit the readme.')

if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()

    # Run function
    main(args.ml, args.fc, args.ofs, args.fn)


