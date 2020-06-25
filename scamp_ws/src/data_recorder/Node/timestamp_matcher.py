#!/usr/bin/env python

#Import dependencies
import re
import os
import numpy as np
import glob

folder = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data"
assert folder, "Provide the dataset folder"
experiment = glob.glob(folder + "/*")

def timeFromFromFile(file_name):
    angular_stamps = []
    # Read the text file, and extract the timestamps
    try:
        angular_stamps = np.loadtxt(file_name,usecols=1, delimiter=',', skiprows=1, dtype=int)

        # I got an error for loading so I made following change:
        #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
        # angular_stamps=np.loadtxt(file_name,usecols=1, delimiter=',', skiprows=1,dtype=float) #
        # angular_stamps=angular_stamps.astype(int)                                             #
        # angular_stamps=np.delete(angular_stamps,0)                                            #
        #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
        
        print("Done")
    except:
        print(file_name)
    return angular_stamps

def angularDataFromFile(fname, idx):
    # Match the angular data to the matrix
    mat = []
    try:
        mat = np.loadtxt(fname, usecols=3, skiprows=1,delimiter=',')
        mat = mat[idx,:]
    except:
        print(fname)
    return mat

def getMatching(arr1, arr2):
    match_stamps = []
    match_idx = []
    for i in arr1:
        dist = abs(i-arr2)
        idx = np.where(dist == 0)[0]
        match_stamps.append(arr2[idx])
        match_idx.append(idx)
        print(match_idx)
    return match_stamps, match_idx

for exp in experiment:
    # Read the image
    images = [os.path.basename(x) for x in glob.glob(exp + "/img/*.jpeg")]
    im_stamp = []
    for im in images:
        stamp = int(re.sub(r'\.jpeg$','',im))
        im_stamp.append(stamp)
    im_stamp = np.array(sorted(im_stamp))
    np.savetxt(exp + "/test.txt", im_stamp, delimiter=",",fmt="%d",header="AngularV")

   
    # Extract time from the file
    file_name = exp + "/Velocity.csv"
    angular_stamps = timeFromFromFile(file_name)

    # Match the time stamps
    match_stamps, match_idx = getMatching(im_stamp, angular_stamps)
    match_idx = np.array(match_idx)
    print(match_idx)
    match_idx = match_idx[:,0]
    
    
     #================================This is what I can undertsand so far ===================
    
    # Get matched commands
    #original_fname = exp + "/Velocity.csv"
    #angular_steer = angularDataFromFile(original_fname, match_idx)

    #new_fname = exp + "/steer.txt"
    #np.savetxt(new_fname, angular_steer, delimiter=",",header="AngularV")
