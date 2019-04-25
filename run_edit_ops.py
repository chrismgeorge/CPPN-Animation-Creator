from model import CPPN
import numpy as np
import argparse
import pickle
import random
import os

def checkDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def make_random_video(cppn, file_name, z_dim, number_of_stills=5):
    zs = [] # list of latent vectors
    z = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
    for i in range(number_of_stills):
        zs.append(z)
        z = z + np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)

    file_name = './videos/'+file_name
    cppn.save_mp4(zs, file_name, loop=True, random_scale=False) # set to false if you don't want a loop

def make_gui_video(cppn, outfiles, file_name):
    zs = []
    scales = []

    for outfile in outfiles:
        with open (outfile, 'rb') as fp: # 'outfile' can be renamed
            reloaded_vectors = pickle.load(fp)
        for i in range(len(reloaded_vectors)): # how many 'key frames' you want
            zs.append(np.array(reloaded_vectors[i][0:-1]))
            scales.append(reloaded_vectors[i][-1])

    zs.append(np.array(reloaded_vectors[0][0:-1]))
    scales.append(reloaded_vectors[0][-1])

    cppn.scales = scales
    # times = []
    #cppn.times = times
    file_name = './videos/'+file_name
    cppn.save_mp4(zs, file_name, loop=False, linear=False, scale_me=True, times=False)

def getMetaData(model_name):
    meta_data_name = os.path.join('./many_models/models/', model_name) + 'meta_data'
    with open (meta_data_name, 'rb') as fp: # 'outfile' can be renamed
        meta_data = pickle.load(fp)
    return meta_data
