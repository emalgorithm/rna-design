from torch.utils.data import Dataset
import pickle
import os, os.path


class RNADataset(Dataset):
    def __init__(self, dir_path, x_transform=None, y_transform=None):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.dir_path = dir_path

    def __len__(self):
        return len(os.listdir(self.dir_path))

    def __getitem__(self, idx):
        sample = pickle.load(open(self.dir_path + "{}.rna".format(idx + 1), 'rb'))
        sample_x = sample['sequence']
        sample_y = sample['dot_bracket']

        if self.x_transform:
            sample_x = self.x_transform(sample_x)

        if self.y_transform:
            sample_y = self.y_transform(sample_y)

        return sample_x, sample_y
