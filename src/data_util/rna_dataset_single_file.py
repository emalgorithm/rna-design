from torch.utils.data import Dataset
import pickle


class RNADatasetSingleFile(Dataset):
    def __init__(self, file_path, x_transform=None, y_transform=None, seq_max_len=40):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.file_path = file_path

        self.data = pickle.load(open(file_path, 'rb'))
        self.data = [x for x in self.data if len(x['sequence']) <= seq_max_len]

        print("{} sequences found at path {} with max length {}".format(len(self.data), file_path,
                                                                            seq_max_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_x = sample['sequence']
        sample_y = sample['dot_bracket']

        if self.x_transform:
            sample_x = self.x_transform(sample_x)

        if self.y_transform:
            sample_y = self.y_transform(sample_y)

        return sample_x, sample_y
