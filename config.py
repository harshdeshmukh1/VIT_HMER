# config.py

import torch

# —————————————————————————
# 1) File paths (adjust as needed)
# —————————————————————————
DICTIONARY_PATH    = r"C:/Users/harsh/OneDrive/Desktop/AMER/dictionary.txt"
TRAIN_CAPTION_PATH = r"C:/Users/harsh/OneDrive/Desktop/AMER/train_caption.txt"
TEST_CAPTION_PATH  = r"C:/Users/harsh/OneDrive/Desktop/AMER/test_caption.txt"
TRAIN_PICKLE_PATH  = r"C:/Users/harsh/OneDrive/Desktop/AMER/offline-train.pkl"
TEST_PICKLE_PATH   = r"C:/Users/harsh/OneDrive/Desktop/AMER/offline-test.pkl"

# —————————————————————————
# 2) Hyperparameters
# —————————————————————————
MAX_SEQ_LEN   = 20
BATCH_SIZE    = 2
NUM_WORKERS   = 0
NUM_EPOCHS    = 2
LR_ENCODER    = 3e-5    # lower LR for pretrained ViT
LR_DECODER    = 1e-4    # higher LR for new decoder
WEIGHT_DECAY  = 1e-5

# ViT model (timm)
VIT_MODEL_NAME = "vit_base_patch16_224"

# —————————————————————————
# 3) Device
# —————————————————————————
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# —————————————————————————
# 4) Load token dictionary & define PAD/SOS/EOS
# —————————————————————————
def load_dictionary(dict_path):
    """
    Read dictionary file where each nonempty line has:
      <token> <id>
    (separated by whitespace). Returns token2id dict.
    """
    token2id = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                # skip lines that don't have exactly 2 parts
                continue
            token, idx = parts
            token2id[token] = int(idx)
    return token2id

# 4a) Load raw dictionary
_raw_token2id = load_dictionary(DICTIONARY_PATH)

# 4b) Add special tokens if missing
for special in ["<pad>", "<sos>", "<eos>"]:
    if special not in _raw_token2id:
        _raw_token2id[special] = max(_raw_token2id.values(), default=-1) + 1

# 4c) Final token2id and id2token
token2id = _raw_token2id
id2token = {idx: tok for tok, idx in token2id.items()}

# 4d) PAD / SOS / EOS IDs
PAD_ID = token2id["<pad>"]
SOS_ID = token2id["<sos>"]
EOS_ID = token2id["<eos>"]

# 4e) Vocabulary size
VOCAB_SIZE = len(token2id)
