import re
import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
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

    def flow_from_directory(self, directory, target_size=(480,640),
            crop_size=(256,256), color_mode='grayscale', batch_size=32,
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
                    img/
                    Velocity.txt
           folder_2/
                    img/
                    Velocity.txt
           .
           .
           folder_n/
                    img/
                    Velocity.txt
    """

    def __init__(self, directory, image_data_generator, target_size=(480,640),
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
        self.ground_truth = []

        # Determine the type of experiment (steering) to compute
        # the loss
        self.exp_type = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            self._decode_experiment_dir(subpath)

        # Conversion of list into array
        self.ground_truth = np.array(self.ground_truth, dtype = K.floatx())

        assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))
        super(DirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, dir_subpath):
        # Load steerings angle in the experiment dir
        steerings_filename = os.path.join(dir_subpath, "Velocity.txt")

        ground_truth = np.loadtxt(steerings_filename, usecols=(2,3,4),
                              delimiter=',', skiprows=1)
        print("Loaded Velocity from {}".format(dir_subpath))

        exp_type = 1


        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "img")

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
                    self.ground_truth.append(ground_truth[frame_number])
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
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                dtype=K.floatx())
        batch_steer = np.zeros((current_batch_size, 3,),
                dtype=K.floatx())

        #batch_coll = np.zeros((current_batch_size, 1,),dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'

        # Build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = cv2.imread(os.path.join(self.directory,fname))
            # x = load_img(os.path.join(self.directory,fname))
            #x = cv2.resize(x,(int(x.shape[1] * 0.5), int(x.shape[0]*0.5)))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            center_width = int(x.shape[1]/2)
            center_height = int(x.shape[0]/2)
            x = x[center_height - int(self.crop_size[0]/2):center_height + int(self.crop_size[0]/2),
                    center_width - int(self.crop_size[1]/2):center_width + int(self.crop_size[1]/2)]


            x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)

            x = x.reshape((x.shape[0],x.shape[1],1))
            x = np.asarray(x, dtype=np.int32)
            #print(x.shape)
            batch_x[i] = x
            cv2.destroyAllWindows()

            # Build batch of steering and collision data
            if self.exp_type[index_array[i]] == 1:
                #print(self.ground_truth[index_array[i]])
                # Steering experiment (t=1)
                batch_steer[i,0:2] = self.ground_truth[index_array[i],0:2]
                #print(batch_x.shape)
            else:
                # Collision experiment (t=0)
                batch_steer[i] = np.array([0.0, 0.0])
                batch_coll[i,0] = 0.0
                batch_coll[i,1] = self.ground_truth[index_array[i]]

        batch_y = batch_steer


        return batch_x, batch_y

def compute_predictions_and_gt(model, generator, steps,
                                     max_q_size=10,
                                     pickle_safe=False, verbose=0):
    """
    Generate predictions and associated ground truth
    for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.
    Function adapted from keras `predict_generator`.
    # Arguments
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.
    # Returns
        Numpy array(s) of predictions and associated ground truth.
    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outs = []
    all_labels = []
    all_ts = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_lab = generator_output
            elif len(generator_output) == 3:
                x, gt_lab, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        outs = model.predict_on_batch(x)
        if not isinstance(outs, list):
            outs = [outs]
        if not isinstance(gt_lab, list):
            gt_lab = [gt_lab]

        if not all_outs:
            for out in outs:
            # Len of this list is related to the number of
            # outputs per model(1 in our case)
                all_outs.append([])

        if not all_labels:
            # Len of list related to the number of gt_commands
            # per model (1 in our case )
            for lab in gt_lab:
                all_labels.append([])
                all_ts.append([])


        for i, out in enumerate(outs):
            all_outs[i].append(out)

        for i, lab in enumerate(gt_lab):
            all_labels[i].append(lab[:,1])
            all_ts[i].append(lab[:,0])

        steps_done += 1
        if verbose == 1:
            progbar.update(steps_done)

    if steps_done == 1:
        return [out for out in all_outs], [lab for lab in all_labels], np.concatenate(all_ts[0])
    else:
        return np.squeeze(np.array([np.concatenate(out) for out in all_outs])).T, \
                          np.array([np.concatenate(lab) for lab in all_labels]).T, \
                          np.concatenate(all_ts[0])



def hard_mining_mse(k):
    """
    Compute MSE for steering evaluation and hard-mining for the current batch.
    # Arguments
        k: number of samples for hard-mining.
    # Returns
        custom_mse: average MSE for the current batch.
    """

    def custom_mse(y_true, y_pred):
        # Parameter t indicates the type of experiment
        t = y_true[:,0]

        # Number of steering samples
        samples_steer = tf.cast(tf.equal(t,1), tf.int32)
        n_samples_steer = tf.reduce_sum(samples_steer)

        if n_samples_steer == 0:
            return 0.0
        else:
            # Predicted and real steerings
            pred_steer = tf.squeeze(y_pred, squeeze_dims=-1)
            true_steer = y_true[:,1]

            # Steering loss
            l_steer = tf.multiply(t, K.square(pred_steer - true_steer))

            # Hard mining
            k_min = tf.minimum(k, n_samples_steer)
            _, indices = tf.nn.top_k(l_steer, k=k_min)
            max_l_steer = tf.gather(l_steer, indices)
            hard_l_steer = tf.divide(tf.reduce_sum(max_l_steer), tf.cast(k,tf.float32))

            return hard_l_steer

    return custom_mse



def hard_mining_entropy(k):
    """
    Compute binary cross-entropy for collision evaluation and hard-mining.
    # Arguments
        k: Number of samples for hard-mining.
    # Returns
        custom_bin_crossentropy: average binary cross-entropy for the current batch.
    """

    def custom_bin_crossentropy(y_true, y_pred):
        # Parameter t indicates the type of experiment
        t = y_true[:,0]

        # Number of collision samples
        samples_coll = tf.cast(tf.equal(t,0), tf.int32)
        n_samples_coll = tf.reduce_sum(samples_coll)

        if n_samples_coll == 0:
            return 0.0
        else:
            # Predicted and real labels
            pred_coll = tf.squeeze(y_pred, squeeze_dims=-1)
            true_coll = y_true[:,1]

            # Collision loss
            l_coll = tf.multiply((1-t), K.binary_crossentropy(true_coll, pred_coll))

            # Hard mining
            k_min = tf.minimum(k, n_samples_coll)
            _, indices = tf.nn.top_k(l_coll, k=k_min)
            max_l_coll = tf.gather(l_coll, indices)
            hard_l_coll = tf.divide(tf.reduce_sum(max_l_coll), tf.cast(k, tf.float32))

            return hard_l_coll

    return custom_bin_crossentropy



def modelToJson(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path,"w") as f:
        f.write(model_json)


def jsonToModel(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model

def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary,f)
        print("Written file {}".format(fname))
