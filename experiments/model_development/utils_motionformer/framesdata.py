"""
framesdata.py: contains the custom FramesDataset.
"""
from torch.utils.data import IterableDataset
import numpy as np
import pickle
if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle
import os, random

class FramesDataset(IterableDataset):
    """
    Wrapper around torch IterableDataset to load videos stored at disk.
    """
    def __init__(self, path, label_translator, fpv=None, skip_every=1, train=False, shuffle=True):
        """
        Loads the machine readable data from experiment and initializes the dataset.

        :param str path: path to experiment output.
        :param int fpv: total number of frames per video. If None, loads videos as-is.
        :param function label_translator: a function that translates labels from the dataset output into 0-indexed integers.
        :param int skip_every: ratio of frames to skip when loading video data (e.g. 3 => load 1/3 of frames).
        :param bool train: whether or not this is a training dataset.
        :param bool shuffle: whether or not to shuffle the dataset before iterating.
        """
        super(FramesDataset).__init__()
        self.path = path
        self.skip_every = skip_every
        self.fpv = fpv
        self.train = train
        self.shuffle = shuffle
        self.label_translator = label_translator
        self.iters = self._get_iters()

    def __len__(self):
        """
        The length of the dataset.

        :return: number of iterations in the experiment output.
        :rtype: int
        """
        return len(self.iters)

    def __getitem__(self, index):
        """
        Loads the iteration at `index`.

        :param int index: index of iteration in the dataset.
        :return: the images and label associated with the iteration.
        :rtype: tuple (np.ndarray[fpv, *frame_shape], int)
        :raises IndexError: if index is not within the length of the dataset.
        """
        if index > self.__len__():
            raise IndexError()
        
        itr = self.iters[index]
        pickle_path = os.path.join(self.path, str(itr), "machine_readable", "iteration_data.pickle")
        return self._load_file(pickle_path)

    def __iter__(self):
        """
        Iterates through the iterations of the experiment in the dataset.

        :return: a generator of images and labels.
        :rtype: generator of tuples (np.ndarray[fpv, *frame_shape], int)
        """
        loader = self._load_all()
        for images, label in loader:
            yield (images, label)

    def _load_file(self, pickle_path):
        """
        Loads an iter file and reads its images and label.

        :param str pickle_path: path to the pickle file to load.
        :return: the images and label at the pickle file.
        :rtype: tuple (np.ndarray[fpv, *frame_shape], int)
        """
        # Load the pickle file at pickle_path
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        # Pre-process the images and label
        images = every_kth(data["images"], self.skip_every)
        if self.fpv != None:
            if self.fpv <= len(images):
                images = images[:self.fpv]
            else:
                images = np.concatenate((images, np.repeat(images[-1:], self.fpv-len(images), axis=0)))
        images = np.asarray(images, dtype="float32")
        label = self.label_translator(data["label"])

        return images, label

    def _get_iters(self, iters = None, cur_max = -1):
        """
        Checks the output path to find the iterations of the experiment. 

        :param list iters: the existing known iterations. If None, initialized to an empty list.
        :param int cur_max: the current max iteration number. If -1, assume no current iterations.
        :return: a list of identifiers for each iteration in the output path.
        :rtype: list
        """

        if iters == None:
            iters = []
        
        # Add iterations with new identifiers to the dataset
        for dirname in os.listdir(self.path):
            if dirname.isdigit() and int(dirname) > cur_max:
                iters.append(int(dirname))

        # Shuffle or sort the iters array.
        if not self.shuffle:
            iters.sort()
        else:
            random.shuffle(iters)

        return iters

    def _load_all(self):
        """
        Loads all the iterations of the experiment in the dataset.

        :return: a generator of images and labels.
        :rtype: generator of tuples (np.ndarray[fpv, *frame_shape], int)
        """

        # Check if any additional iterations were added in the meantime
        path = self.path
        num_iters = len(os.listdir(path))
        max_iter = max(self.iters)
        self.iters = self._get_iters(self.iters, max_iter)

        for i in self.iters:
            # If any new iterations were added, load them
            if len(os.listdir(path)) > num_iters:
                self.iters = self.get_iters(self.iters, max_iter)
                max_iter = max(self.iters)

            # Load current iteration and return its (images, label) pair.
            pickle_path = os.path.join(path, str(i), "machine_readable", "iteration_data.pickle")
            yield self._load_file(pickle_path)

def every_kth(array, k):
    """
    Cuts off the array into |array|/k elements by keeping every k-th element.

    :param np.ndarray array: the array to cut off of size (|array|, *)
    :param int k: determines how to cut off the array.
    :return: a new array where every k-th element of array is preserved.
    :rtype: np.ndarray[|array|/k, *]
    """
    if k == 1:
        return array
    return np.array([array[i] for i in range(len(array)) if i % k == 0])

if __name__ == '__main__':

    # Initialize dataset
    path = "../data/"
    dataset = FramesDataset(path, fpv=350, shuffle=True)

    # You can start a Data Loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset)

    # Or you can start an FFCV Writer and write the dataset into an FFCV file.
    from ffcv.writer import DatasetWriter
    from ffcv.fields import NDArrayField, IntField

    write_path = '../ds.beton'
    writer = DatasetWriter(write_path, {
        'video': NDArrayField(dtype=np.dtype("float32"), shape=(350, 224, 224, 3)),
        'label': IntField()
        },
        page_size = 2<<28)

    writer.from_indexed_dataset(dataset)
    