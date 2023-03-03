"""
framesdata.py: contains the custom FramesDataset.
"""
import torch
from torch.utils.data import IterableDataset
import numpy as np
import pickle

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle
import os, random
import cv2 as cv
from PIL import Image
import yaml

class FramesDataset(IterableDataset):
    """
    Wrapper around torch IterableDataset to load videos stored at disk.
    """

    def __init__(
        self,
        path,
        label_translator,
        fpv=None,
        skip_every=1,
        train=False,
        shuffle=True,
        source_type="pickle",
        yaml_label_key="label",
    ):
        """
        Loads the machine readable data from experiment and initializes the dataset.

        :param str path: path to experiment output.
        :param int fpv: total number of frames per video. If None, loads videos as-is.
        :param function label_translator: a function that translates labels from the dataset output into 0-indexed integers.
        :param int skip_every: ratio of frames to skip when loading video data (e.g. 3 => load 1/3 of frames).
        :param bool train: whether or not this is a training dataset.
        :param bool shuffle: whether or not to shuffle the dataset before iterating.
        :param string source_type: one of 'pickle', 'video', or 'frames'
        """
        super(FramesDataset).__init__()
        self.path = path
        self.skip_every = skip_every
        self.fpv = fpv
        self.train = train
        self.shuffle = shuffle
        self.label_translator = label_translator
        self.source_type = source_type
        self.yaml_label_key = yaml_label_key
        self.iters = self._get_iters()

        if self.source_type == "pickle":
            self.load_file_function = self._load_pickle
        elif self.source_type == "video":
            self.load_file_function = self._load_video
        elif self.source_type == "frames":
            self.load_file_function = self._load_frames
        else:
            raise ValueError(
                'invalid source type: must be "pickle", "video", or "frames"'
            )

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
        return self.load_file_function(self.data_source_path(itr))

    def __iter__(self):
        """
        Iterates through the iterations of the experiment in the dataset.

        :return: a generator of images and labels.
        :rtype: generator of tuples (np.ndarray[fpv, *frame_shape], int)
        """
        loader = self._load_all()
        for images, label in loader:
            yield (images, label)

    def _load_pickle(self, pickle_path):
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
                images = images[: self.fpv]
            else:
                images = np.concatenate(
                    (images, np.repeat(images[-1:], self.fpv - len(images), axis=0))
                )
        images = np.asarray(images, dtype="float32")
        label = self.label_translator(data["label"])

        return images, label

    def _load_video(self, video_path):
        cap = cv.VideoCapture(video_path)

        images = []
        labels = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            images.append(np.asarray(frame, dtype="float32"))

            ...  # TODO: fetch label and append to labels

        return every_kth(images, self.skip_every), every_kth(labels, self.skip_every)

    def _load_frames(self, frames_path):
        # Load images
        images = []
        for frame in os.listdir(frames_path):
            frame_path = os.path.join(frames_path, frame)
            frame = Image.open(frame_path)
            images.append(np.asarray(frame, dtype=np.float32))
            
        images = every_kth(images, self.skip_every)
        
        # Load label
        with open(os.path.join(frames_path, "..", "experiment_stats.yaml")) as f:
            data = yaml.safe_load(f)
            label = data[self.yaml_label_key]
        
        return images, label

    def _get_iters(self, iters=None, cur_max=-1):
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
            yield self.load_file_function(self.data_source_path(i))

    def data_source_path(self, i):
        if self.source_type == "pickle":
            return os.path.join(
                self.path, str(i), "machine_readable", "iteration_data.pickle"
            )
        elif self.source_type == "video":
            return os.path.join(self.path, str(i), "experiment_video.mp4")
        elif self.source_type == "frames":
            return os.path.join(self.path, str(i), "human_readable", "frames")
        else:
            raise ValueError(
                'invalid source type: must be "pickle", "video", or "frames"'
            )


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


def collate_videos(batch):
    """
    A function available to override the default DataLoader collate_fn. Unifies the lengths of videos in a batch by
    repeating the last frame in videos with less frames.

    :param list batch: a batch of videos and labels as a list of tuples (np.array, int)
    :return: a data point where videos and labels are stacked.
    :rtype: tuple[Tensor, Tensor]
    """
    max_len = max([len(i[0]) for i in batch])

    for i in range(len(batch)):
        images, label = batch[i]
        images = np.concatenate(
            (images, np.repeat(images[-1:], max_len - len(images), axis=0))
        )
        batch[i] = (images, label)

    # From here, uses default collate
    data = torch.from_numpy(np.stack([item[0] for item in batch]))
    target = torch.LongTensor([item[1] for item in batch])  # image labels.

    return data, target


if __name__ == "__main__":

    # Initialize dataset
    from utils.translators import GRAVITY

    path = "data/"
    dataset = FramesDataset(path, label_translator=GRAVITY, shuffle=True)

    # You can start a Data Loader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset=dataset, collate_fn=collate_videos, batch_size=5)

    # Or you can start an FFCV Writer and write the dataset into an FFCV file.
    from ffcv.writer import DatasetWriter
    from ffcv.fields import NDArrayField, IntField

    write_path = "../ds.beton"
    writer = DatasetWriter(
        write_path,
        {
            "video": NDArrayField(dtype=np.dtype("float32"), shape=(350, 224, 224, 3)),
            "label": IntField(),
        },
        page_size=2 << 28,
    )

    writer.from_indexed_dataset(dataset)
