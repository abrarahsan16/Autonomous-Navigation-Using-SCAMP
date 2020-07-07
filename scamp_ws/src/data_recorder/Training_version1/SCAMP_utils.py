#!/usr/bin/env python

import re
import os
import numpy as np
import tensorflow as tf
import json
import cv2
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json


#import SCAMP_img 

class DroneDataGenerator(ImageDataGenerator):

    def flow_from_directory(self, directory, target_size=(224,224),
            crop_size=(250,250), color_mode='grayscale', batch_size=32,
            shuffle=True, seed=None, follow_links=False):
        return DroneDirectoryIterator(
                directory, self,
                target_size=target_size, crop_size=crop_size, color_mode=color_mode,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                follow_links=follow_links)

#==============================PART 1=========================================

class DroneDirectoryIterator(Iterator):

    def __init__(self, directory, image_data_generator,
            target_size=(224,224), crop_size = (250,250), color_mode='grayscale',
            batch_size=32, shuffle=True, seed=None, follow_links=False):
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.crop_size = tuple(crop_size)
        self.follow_links = follow_links
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.crop_size + (3,)
        else:
            self.image_shape = self.crop_size + (1,)

        # First count how many experiments are out there
        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
	    #print(subdir)
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
	#print(experiments)
        self.num_experiments = len(experiments)
        self.formats = {'jpeg'}

  
	 # Idea = associate each filename with a corresponding steering or label
        self.filenames = []
        self.ground_truth = []

        # Determine the type of experiment (steering or collision) to compute
        # the loss
        self.exp_type = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
	    print(subpath)
            self._decode_experiment_dir(subpath) 

	self.ground_truth = np.array(self.ground_truth, dtype = K.floatx())

        #assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} worlds.'.format(
                self.samples, self.num_experiments))
        super(DroneDirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)




    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                key=lambda tpl: tpl[0])




    def _decode_experiment_dir(self, dir_subpath):
	
        # Load steerings or labels in the experiment dir
        steerings_filename = os.path.join(dir_subpath, "Velocity.txt")
	

	try:
            ground_truth = np.loadtxt(steerings_filename, usecols=0,
                                  delimiter=',', skiprows=1)
            exp_type = 1
        except OSError as e:
            # Try load collision labels if there are no steerings
            try:
                ground_truth = np.loadtxt(labels_filename, usecols=0)
                exp_type = 0
            except OSError as e:
                print("Neither steerings nor labels found in dir {}".format(
                dir_subpath))
                raise IOError	


        # Now fetch all images in the image subdir
        image_dir_path=os.path.join(dir_subpath, "img")
	
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
			#print("true")
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path,
                            self.directory))
                    self.ground_truth.append(ground_truth[frame_number])
                    self.exp_type.append(exp_type)
                    self.samples += 1
	
	#print(len(self.filenames))




#==============================PART 2=========================================

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


    def _get_batches_of_transformed_samples(self,index_array):
        """
        Public function to fetch next batch.

        # Returns
            The next batch of images and labels.
        """
        current_batch_size = index_array.shape[0]
        # Image transformation is not under thread lock, so it can be done in
        # parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                dtype=K.floatx())
        batch_steer = np.zeros((current_batch_size, 1,),
                dtype=K.floatx())
        batch_coll = np.zeros((current_batch_size, 1,),
                dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j] 		

				# filenames=[".../data/House/1234.jpeg",".../data/House/1235.jpeg"....] len=20759
				# fname=".../data/House/1234.jpeg"
				# fname=".../data/House/1235.jpeg"
				# .....

            #x=SCAMP_img.load_img(os.path.join(self.directory, fname), 
            #        grayscale=grayscale,
            #        crop_size=self.crop_size, #### something wrong here, keeps getting broadcast input array error
            #        target_size=self.target_size)

	    #print(self.filenames)
	
	    x=cv2.imread(os.path.join(self.directory, fname))
	    x=cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

	    if (x.shape[0], x.shape[1])!=(480,640):
            	print("size is wrong")	


	    x=x.reshape((x.shape[0],x.shape[1], 1))
	    x=np.asarray(x) # image array (480,640,1)

	    #print(x)
	   

            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)

            batch_x[i]=x  # list len=32 (batch size)


            # Build batch of steering and collision data
            if self.exp_type[index_array[i]] == 1:
		
                # Steering experiment (t=1)
                #batch_steer[i,0] =1.0
                batch_steer[i,0]=self.ground_truth[index_array[i]]
                #batch_coll[i] = np.array([1.0, 0.0])
            else:
                # Collision experiment (t=0)
                batch_steer[i] = np.array([0.0, 0.0])
                batch_coll[i,0] = 0.0
                batch_coll[i,1] = self.ground_truth[index_array[i]]

        batch_y = [batch_steer]
	
	#print("/////////////////////////////////////////")

	#print(batch_x)
	#print(batch_y[0].shape)
	#print(batch_x.shape)

        return batch_x, batch_y



#===========================Testing use====================================================
train_datagen=DroneDataGenerator(rotation_range = 0.2,
                                             rescale = 1./255,
                                             width_shift_range = 0.2,
                                             height_shift_range=0.2)

train_generator=train_datagen.flow_from_directory("/home/andrew/turtlebot3_ws/src/Converter/src/Test1/data",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(480,640),
                                                        crop_size=(480,640),
                                                        batch_size = 32)



val_datagen =DroneDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory("/home/andrew/turtlebot3_ws/src/Converter/src/Test1/testfolder",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(480,640),
                                                        crop_size=(480,640),
                                                        batch_size = 32)

a,b=train_generator.next()






