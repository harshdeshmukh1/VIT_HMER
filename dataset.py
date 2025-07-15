# dataset.py

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import BATCH_SIZE, NUM_WORKERS

from config import (
    TRAIN_CAPTION_PATH,
    TEST_CAPTION_PATH,
    TRAIN_PICKLE_PATH,
    TEST_PICKLE_PATH,
    MAX_SEQ_LEN,
    token2id,
    PAD_ID,
    SOS_ID,
    EOS_ID
)

# 1) ViT‐compatible image transforms
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class AMERDataset(Dataset):
    """
    Dataset for handwritten-equation images + LaTeX token sequences.
    - image_pickle:  .pkl with { key: np_array [1, H, W] }
    - caption_txt:   each line "key <space> tok1 tok2 tok3 ... "
    - token2id:      mapping str → int, incl. PAD_ID, SOS_ID, EOS_ID
    - max_len:       fixed length for token sequences
    - transform:     torchvision transforms
    """
    def __init__(self, image_pickle, caption_txt, token2id, max_len=MAX_SEQ_LEN, transform=None):
        super().__init__()
        self.token2id = token2id
        self.max_len = max_len
        self.transform = transform

        # a) Load images dict from pickle
        with open(image_pickle, "rb") as f:
            self.images = pickle.load(f)  # { key: np_array [1, H, W] }

        # b) Load captions
        self.captions = {}
        with open(caption_txt, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                key, caption = stripped.split("\t", 1) if "\t" in stripped else stripped.split(maxsplit=1)
                tokens = caption.split()
                token_ids = [SOS_ID]
                token_ids += [token2id.get(tok, PAD_ID) for tok in tokens]
                token_ids.append(EOS_ID)

                # Pad/truncate to max_len
                if len(token_ids) < max_len:
                    token_ids += [PAD_ID] * (max_len - len(token_ids))
                else:
                    token_ids = token_ids[:max_len]

                self.captions[key] = torch.tensor(token_ids, dtype=torch.long)

        # c) Only keep keys present in both images and captions
        self.keys = [k for k in self.captions.keys() if k in self.images]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img_np = self.images[key]  # np_array: [1, H, W]

        # Convert to PIL and RGB
        pil_img = Image.fromarray(img_np[0], mode="L").convert("RGB")
        if self.transform:
            pil_img = self.transform(pil_img)  # → [3,224,224]

        caption_ids = self.captions[key]  # [max_len] Tensor
        return pil_img, caption_ids


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Returns:
      train_loader, test_loader
    """
    train_set = AMERDataset(
        image_pickle=TRAIN_PICKLE_PATH,
        caption_txt=TRAIN_CAPTION_PATH,
        token2id=token2id,
        max_len=MAX_SEQ_LEN,
        transform=vit_transform
    )
    test_set = AMERDataset(
        image_pickle=TEST_PICKLE_PATH,
        caption_txt=TEST_CAPTION_PATH,
        token2id=token2id,
        max_len=MAX_SEQ_LEN,
        transform=vit_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Sanity check
    train_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)
    images, captions = next(iter(train_loader))
    print("Images:", images.shape)    # e.g. [8, 3, 224, 224]
    print("Captions:", captions.shape) # e.g. [8, 50]
