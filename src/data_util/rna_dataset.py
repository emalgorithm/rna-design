from torch.utils.data import Dataset
import pickle


class RNADataset(Dataset):
    def __init__(self, file_path, x_transform=None, y_transform=None):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.file_path = file_path

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        sample = pickle.load(open(self.file_path + "{}.rna".format(idx + 1), 'rb'))
        sample_x = sample['sequence']
        sample_y = sample['dot_bracket']

        if self.x_transform:
            sample_x = self.x_transform(sample_x)

        if self.y_transform:
            sample_y = self.y_transform(sample_y)

        return sample_x, sample_y
