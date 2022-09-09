import random
import os

import imageio as imageio
from PIL import Image
from megengine.data.dataset import Dataset
from megengine.data.dataloader import DataLoader
import megengine.data.transform as transforms
from megengine.data import RandomSampler

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.ToMode("CHW")])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToMode("CHW")])  # transform it into a torch tensor


class SIGNSDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    def __init__(self, data_dir, transform=lambda x: x):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        # self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = imageio.imread(self.filenames[idx])  # PIL image
        # image = Image.open(self.filenames[idx])  # PIL image
        # image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                current_dataset = SIGNSDataset(path)
                dl = DataLoader(dataset=current_dataset,
                                sampler=RandomSampler(dataset=current_dataset, batch_size=params.batch_size),
                                transform=train_transformer,
                                num_workers=params.num_workers)
            else:
                current_dataset = SIGNSDataset(path)
                dl = DataLoader(dataset=current_dataset,
                                sampler=RandomSampler(dataset=current_dataset, batch_size=params.batch_size),
                                transform=eval_transformer,
                                num_workers=params.num_workers)

            dataloaders[split] = dl

    return dataloaders
