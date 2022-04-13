from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch

# Dataset class


class SentenceDataset(Dataset):
    def __init__(self, df, transform, target_transform, slots_transform):
        self.x = df["preprocessed_sentences"].to_numpy()
        self.intent = df["label"].to_numpy()
        self.slots = df["preprocessed_slots"].to_numpy()
        self.transform = transform
        self.target_transform = target_transform
        self.slots_transform = slots_transform

    def __len__(self):
        return len(self.intent)

    def __getitem__(self, idx):

        return torch.IntTensor(self.x[idx]), torch.LongTensor(
            self.target_transform(self.intent[idx])), torch.LongTensor(self.slots_transform(self.slots[idx])).transpose(0, 1)
