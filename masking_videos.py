from model import CPPN
import numpy as np
import argparse
import pickle
import random
import os
import cv2

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--mn", type=str, default=None) # main video
    parser.add_argument("--mk", type=str, default=None) # video mask
    parser.add_argument("--ml", type=str, default=None) # one of the model names
    parser.add_argument("--fn", type=str, default=None) # video file name

    # Parse arguments
    args = parser.parse_args()

    return args

def getDims(model):
    meta_data_name = os.path.join('./many_models/models', model) + 'meta_data'
    with open (meta_data_name, 'rb') as fp: # 'outfile' can be renamed
        meta_data = pickle.load(fp)

    x_dim = meta_data[0]
    y_dim = meta_data[1]

    return x_dim, y_dim

def makeMaskedVideo(main_v, mask_v, model, filename, x_dim, y_dim):
    FPS = 24.0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter('./videos/'+filename, fourcc, FPS, (x_dim, y_dim), 0)

    capMain = cv2.VideoCapture('./videos/'+main_v)
    capMain.set(cv2.CAP_PROP_FPS, FPS)

    capMask = cv2.VideoCapture('./videos/'+mask_v)
    capMask.set(cv2.CAP_PROP_FPS, FPS)

    while(True):
            # Capture frame-by-frame
            retMain, frameMain = capMain.read()
            retMask, frameMask = capMask.read()
            if not retMain or not retMask:
                break

            # Convert to black and white, then threshold
            frameMain = cv2.cvtColor( frameMain, cv2.COLOR_BGR2GRAY )
            ret,frameMain = cv2.threshold(frameMain,127,255,cv2.THRESH_BINARY)
            frameMask = cv2.cvtColor( frameMask, cv2.COLOR_BGR2GRAY )
            ret,frameMask = cv2.threshold(frameMask,127,255,cv2.THRESH_BINARY)

            # convert to float so subtract works correctly
            frameMain = frameMain.astype(float)
            frameMask = frameMask.astype(float)

            # subtract the mask from the main, and absolute value it to
            # simulate inversion of the masks
            outFrame = np.subtract(frameMask, frameMain)
            outFrame = np.uint8(np.absolute(outFrame))

            video.write(outFrame)

    cv2.destroyAllWindows()
    video.release()

def main(main_v, mask_v, model, filename):
    if (main_v != None and mask_v != None and model != None):
        x_dim, y_dim = getDims(model)
        makeMaskedVideo(main_v, mask_v, model, filename, x_dim, y_dim)


if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()
    # Run function
    main(args.mn, args.mk, args.ml, args.fn)


