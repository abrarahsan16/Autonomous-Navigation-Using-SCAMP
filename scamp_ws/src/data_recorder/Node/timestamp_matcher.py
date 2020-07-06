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
    # Read the CSV files from the individual data folders and extract the timestamps
    angular_stamps = []
    try:
        angular_stamps = np.loadtxt(file_name,usecols=1, delimiter=',', skiprows=1)
    except:
        print("Error at: " + file_name)
    return angular_stamps

def angularDataFromFile(fname, idx):
    # Match the angular data to the matrix
    mat = []
    try:
        mat = np.loadtxt(fname, usecols=(2,3), skiprows=1,delimiter=',')
        mat = mat[idx,:]
    except:
        print("Error at: " + fname)
    return mat

def getMatching(arr1, arr2):
    match_stamps = []
    match_idx = []
    for i in arr1:
        dist = abs(i-arr2)
        idx = np.where(dist == 0)[0]
        match_stamps.append(arr2[idx])
        match_idx.append(idx)
    return match_stamps, match_idx

for exp in experiment:
    # Read the image
    images = [os.path.basename(x) for x in glob.glob(exp + "/img/*.jpeg")]
    im_stamp = []
    for im in images:
        stamp = int(re.sub(r'\.jpeg$','',im))
        im_stamp.append(stamp)
    im_stamp = np.array(sorted(im_stamp))

    # Extract time from the file
    file_name = exp + "/Velocity.csv"
    angular_stamps = timeFromFromFile(file_name)

    # Match the time stamps
    match_stamps, match_idx = getMatching(im_stamp, angular_stamps)
    match_idx = np.array(match_idx)
    match_idx = match_idx[:,0]

    original_fname = exp + "/Velocity.csv"
    angular_steer = angularDataFromFile(original_fname, match_idx)

    new_fname = exp + "/Velocity.txt"
    np.savetxt(new_fname, angular_steer, delimiter=",",fmt="%f",header="LinearA, AngularV")
