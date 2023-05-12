import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class ArtistDataset(Dataset):
    def __init__(self, artist_name):
        self.artist_name = artist_name
        self.data = []
        with open('./training/artist_datasets/' + artist_name + '/prompts.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/artist_datasets/' + self.artist_name + "/" + source_filename)
        target = cv2.imread('./training/artist_datasets/' + self.artist_name + "/" + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


if __name__ == "__main__":
    dataset = ArtistDataset("shadeon")
    item = dataset[1]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)