import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import json
import cv2
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json

class DataGenerator(ImageDataGenerator):

    def flow_from_directory(self, directory, target_size=(480,256),
            crop_size=(256,256), color_mode='grayscale', batch_size=3,
            shuffle=True, seed=None, follow_links=False):

        return DirectoryIterator(directory, self, target_size=target_size,
                crop_size=crop_size, color_mode=color_mode,
                batch_size=batch_size, shuffle=shuffle,
                seed=seed, follow_links=follow_links)

class DirectoryIterator(Iterator):
    """
    We assume that the folder structure is:
    root_folder/
           folder_1/
                    L/
                    S/
                    R/

           folder_2/
                    L/
                    S/
                    R/

           folder_n/
                    L/
                    S/
                    R/
    """

    def __init__(self, directory, image_data_generator, target_size=(480,256),
            crop_size=(256,256), color_mode='grayscale',batch_size=32,
            shuffle=True,seed=None,follow_links=False):

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
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)
        self.formats = {'png', 'jpeg'}

        # Idea = associate each filename with a corresponding steering angle
        self.filenames = []

        # Determine the type of experiment (steering) to compute
        # the loss
        self.exp_type = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            self._decode_experiment_dir(subpath)

        assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))
        super(DirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, dir_subpath):
        exp_type = 1

        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "img2")

        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path,
                            self.directory))
                    self.exp_type.append(exp_type)
                    self.samples += 1

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array) :
        """
        Public function to fetch next batch.
        # Returns
            The next batch of images and labels.
        """
        current_batch_size = index_array.shape[0]

        # Image transformation is not under thread lock, so it can be done in
        # parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape,dtype=K.floatx())
        batch_steer = np.zeros((current_batch_size, 2,),dtype=K.floatx())
        batch_coll = np.zeros((current_batch_size, 2,),dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            labelName = fname.split("img2/",1)[1]

            #print(index_array[i])

            #if labelName.startswith("L"):
            #    batch_steer[i,0:2] = (1,0,0)
            #elif labelName.startswith("R"):
            #    batch_steer[i,0:2] = (0,0,1)
            #else:
            #    batch_steer[i,0:2] = (0,1,0)

            #print(batch_steer)

            x = cv2.imread(os.path.join(self.directory,fname))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            #center_width = int(x.shape[1]/2)
	    #center_height = int(450)


            #x = x[center_height - int(self.crop_size[0]):center_height, center_width - int(self.crop_size[1]/2):center_width + int(self.crop_size[1]/2)]
            x = cv2.resize(x, dsize=(256, 256)) # needs center crop

            x = np.asarray(x, dtype=np.int32)
            x = x.reshape((x.shape[0],x.shape[1],1))

            batch_x[i] = x
            cv2.destroyAllWindows()

            # Build batch of steering and collision data
            #if self.exp_type[index_array[i]] == 1:
                #print(self.ground_truth[index_array[i]])
                # Steering experiment (t=1)
                #batch_steer[i,0:2] = self.ground_truth[index_array[i],0:2]
            if labelName.startswith("L"):
                batch_steer[i,0] = 1
                batch_steer[i,1] = 0
                batch_coll[i] = np.array(1,0)
            elif labelName.startswith("R"):
                batch_steer[i,0] = 0
                batch_steer[i,1] = 1
                batch_coll[i] = np.array(1,0)
            #elif self.exp_type[index_array[i]] == 0:
            if labelName.startswith("S"):
                batch_coll[i,0] = 1
                batch_coll[i,1] = 0
                batch_steer[i] = np.array(1,1)
            elif labelName.startswith("B"):
                batch_coll[i,0] = 0
                batch_coll[i,1] = 1
                batch_steer[i] = np.array(0,0)

		#name as LS LB, RS,RB

        batch_y = [batch_steer, batch_coll]

        return batch_x, batch_y
