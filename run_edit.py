from model import CPPN
import numpy as np
import argparse
import pickle
import random
import os

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--x_dim", type=int, default=1280)
    parser.add_argument("--y_dim", type=int, default=720)
    parser.add_argument("--z_dim", type=int, default=5)
    parser.add_argument("--scale", type=int, default=16)

    parser.add_argument("--neurons_per_layer", type=int, default=6)
    parser.add_argument("--number_of_layers", type=int, default=8)
    parser.add_argument("--color_channels", type=int, default=3)

    parser.add_argument("--number_of_stills", type=int, default=10)
    parser.add_argument("--interpolations_per_image", type=int, default=24)
    parser.add_argument("--file_name", type=str, default='./videos/test01.mp4')
    parser.add_argument("--load_model", type=bool, default=False)

    # Parse arguments
    args = parser.parse_args()

    return args

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_single_random_image(cppn, file_name, z_dim):
    z = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
    cppn.save_png(z, file_name)
    with open('outfile', 'wb') as f: # 'outfile' can be renamed
        pickle.dump([z.tolist()], f)

def save_many_images(cppn, outfile, z_dim):
    zs = []
    for i in range(25):
        file_name = './photos/test%d.png' % i
        z_vector = np.random.uniform(-1., 1., size=(z_dim)).astype(np.float32)
        cppn.save_png(z_vector, file_name)
        zs.append(z_vector.tolist())
        print('Done! The image is at %s' % file_name)
    with open(outfile, 'wb') as f: # can change 'outfile'
        pickle.dump(zs, f)

def redisplay_image(cppn, outfile, index, file_name):
    with open (outfile, 'rb') as fp: # 'outfile' can be renamed
        reloaded_vectors = pickle.load(fp)
    z = np.array(reloaded_vectors[index]) # would be named test10.png
    cppn.save_png(z, file_name) # make sure you name this something you'll remember

def wibble_around_image(cppn, outfile, index, file_name, z_dim):
    with open (outfile, 'rb') as fp: # 'outfile' can be renamed
        reloaded_vectors = pickle.load(fp)
    z_start = np.array(reloaded_vectors[index]) # would be named test10.png
    zs = []
    for i in range(10): # how many 'key frames' you want
        z = np.random.uniform(-.2, .2, size=(z_dim)).astype(np.float32)
        z = np.add(z_start, z)
        zs.append(z)
    cppn.save_mp4(zs, file_name, loop=False, linear=False)

def evolve_image(cppn, outfile, index, new_outfile, folder_name, z_dim):
    checkDirectory(folder_name)
    with open (outfile, 'rb') as fp: # 'outfile' can be renamed
        reloaded_vectors = pickle.load(fp)
    z_start = np.array(reloaded_vectors[index]) # would be named test10.png
    zs = []
    for i in range(20): # how many 'key frames' you want
        file_name = './%s/test%d.png' % (folder_name, i)
        #z_vector = np.random.uniform(-.1, 0, size=(z_dim)).astype(np.float32)
        z_vector = np.zeros(z_dim)
        randomIndex = random.randint(0, z_dim-1)
        z_vector[randomIndex] = random.random()
        z_vector = np.add(z_start, z_vector)
        cppn.save_png(z_vector, file_name)
        zs.append(z_vector.tolist())
        print('Done! The image is at %s' % file_name)
    with open(new_outfile, 'wb') as f: # can change 'outfile'
        pickle.dump(zs, f)

def make_random_video(cppn, file_name, number_of_stills, z_dim):
    zs = [] # list of latent vectors
    z = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
    for i in range(number_of_stills):
        zs.append(z)
        z = z + np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)

    cppn.save_mp4(zs, file_name, loop=False, random_scale=True) # set to false if you don't want a loop

def make_gui_video(cppn, outfile, file_name):
    with open (outfile, 'rb') as fp: # 'outfile' can be renamed
        reloaded_vectors = pickle.load(fp)

    zs = []
    scales = []
    #times = [56, 6, 48, 6, 56, 6, 12, 6, 12, 6, 12, 6, 56, 6, 12, 6, 12, 6, 12, 6, 56, 6, 128, 90]

    for i in range(len(reloaded_vectors)): # how many 'key frames' you want
        zs.append(np.array(reloaded_vectors[i][0:-1]))
        scales.append(reloaded_vectors[i][-1])

    zs.append(np.array(reloaded_vectors[0][0:-1]))
    scales.append(reloaded_vectors[0][-1])

    cppn.scales = scales
    #cppn.times = times
    cppn.save_mp4(zs, file_name, loop=False, linear=False, scale_me=True, times=False)

def make_gui_video_from_mutliple(cppn, outfiles, file_name):
    zs = []
    scales = []
    # times = [56, 6, 48, 6, 56, 6, 14, 6, 14, 6, 14, 6, 56, 6, 14, 6, 14, 6, 14, 6, 64, 6, 128, 256]
    for outfile in outfiles:
        with open (outfile, 'rb') as fp: # 'outfile' can be renamed
            reloaded_vectors = pickle.load(fp)
        for i in range(len(reloaded_vectors)): # how many 'key frames' you want
            zs.append(np.array(reloaded_vectors[i][0:-1]))
            scales.append(reloaded_vectors[i][-1])

    print(len(times), len(scales))

    cppn.scales = scales
    #cppn.times = times
    cppn.save_mp4(zs, file_name, loop=False, linear=False, scale_me=True, times=False)

def main(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
         color_channels, number_of_stills, interpolations_per_image, file_name,
         load_model):
    load_model = True
    ## Change name to save a new model.
    model_name = '9-17-3-0_'
    model_dir = 'many_models/models'

    if load_model:
        ## Load a saved CPPN
        checkpoint_path = os.path.join(model_dir, model_name)
        outfile_name = checkpoint_path + 'meta_data'
        with open (outfile_name, 'rb') as fp: # 'outfile' can be renamed
            data = pickle.load(fp)

        x_dim = data[0]
        y_dim = data[1]
        z_dim = data[2]
        scale = data[3]
        neurons_per_layer = data[4]
        number_of_layers = data[5]
        color_channels = data[6]
        test = data[7]

        # Initialize CPPN with parameters ######################################
        print('Initializing CPPN...')
        cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                    color_channels, interpolations_per_image, test=test)
        cppn.neural_net(True)
        ########################################################################

        cppn.load_model(model_name=model_name, model_dir=model_dir)

    else:
        # Initialize CPPN with parameters ######################################
        print('Initializing CPPN...')
        cppn = CPPN(x_dim, y_dim, z_dim, scale, neurons_per_layer, number_of_layers,
                    color_channels, interpolations_per_image, test=test)
        cppn.neural_net(True)
        ########################################################################

    # (1) Set to_run
    to_run = 5
    # (2) Change any variable names to your preferred name.
    # (3) Run code!

    ## Save CPPN if you want to.
    # cppn.save_model(model_name=model_name, save_outfile=True)

    if (to_run == 0):
        ## Save single random image
        file_name = './photos/000.png'
        save_single_random_image(cppn, file_name, z_dim)
    elif (to_run == 1):
        ## Make 100 random images and save their latent vectors
        outfile = 'new_outfile'
        save_many_images(cppn, outfile, z_dim)
    elif (to_run == 2):
        ## Re-display a specific image from a saved model.
        outfile = 'outfile'
        index = 0
        file_name = 'rename_me.png'
        redisplay_image(cppn, outfile, index, file_name)
    elif (to_run == 3):
        ## Wibble around a specific image and make a video from it.
        outfile = 'outfile_crow'
        index = 9
        file_name = './videos/crow.mp4'
        wibble_around_image(cppn, outfile, index, file_name, z_dim)
    elif (to_run == 4):
        ## Generate images from base image
        outfile = 'outfile'
        index = 118
        new_outfile = 'outfile_z_test'
        folder_name = 'z'
        evolve_image(cppn, outfile, index, new_outfile, folder_name, z_dim)
    elif (to_run == 5):
        ## Make a list of random video
        file_name = './videos/bw_02.mp4'
        make_random_video(cppn, file_name, number_of_stills, z_dim)
    elif (to_run == 6):
        ## Make a video using gui data
        outfile = '8-19-3-5_'
        file_name = './videos/bw.mp4'
        make_gui_video(cppn, outfile, file_name)
    elif (to_run == 7):
        ## Make a video using gui data
        outfiles = ['gui_scene_1', 'gui_scene_2', 'gui_scene_3']
        file_name = './videos/all_scenes_2.mp4'
        make_gui_video_from_mutliple(cppn, outfiles, file_name)


if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()
    # Run function
    main(args.x_dim, args.y_dim, args.z_dim, args.scale, args.neurons_per_layer,
         args.number_of_layers, args.color_channels, args.number_of_stills,
         args.interpolations_per_image, args.file_name, args.load_model)


