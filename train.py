# train.py

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LR_ENCODER,
    LR_DECODER,
    WEIGHT_DECAY,
    token2id,
    id2token,
    VOCAB_SIZE
)
from dataset import get_dataloaders
from model import ViTSeq2Seq
from utils import greedy_decode

# —————————————————————————
# 1) Levenshtein (Edit) Distance for WER
# —————————————————————————
def edit_distance(ref_tokens, hyp_tokens):
    """
    Compute Levenshtein (edit) distance between two lists of tokens.
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    # Create a (n+1) x (m+1) matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m]


def compute_wer_and_sacc(model, dataloader, id2token, device):
    """
    Iterate over `dataloader` (test set), run greedy_decode per image,
    and compute:
      - total_edit_distance / total_ref_length  → WER
      - fraction of exact matches → SACC
    """
    model.eval()
    total_edits = 0
    total_ref_len = 0
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)                 # [B, 3, 224, 224]
            captions = captions.to(device)             # [B, MAX_SEQ_LEN]

            batch_size = images.size(0)
            for i in range(batch_size):
                img = images[i].unsqueeze(0)           # [1,3,224,224]
                gt_ids = captions[i].tolist()          # [MAX_SEQ_LEN]

                # Strip off padding after EOS (if present)
                if 0 in gt_ids:
                    eos_idx = gt_ids.index(0)
                    gt_ids = gt_ids[:eos_idx]
                # Also remove any trailing PAD_ID beyond EOS
                if token2id["<eos>"] in gt_ids:
                    last = gt_ids.index(token2id["<eos>"]) + 1
                    gt_ids = gt_ids[:last]

                # Generate prediction
                pred_ids = greedy_decode(model, img, max_len=MAX_SEQ_LEN)

                # Similarly strip pred at EOS
                if token2id["<eos>"] in pred_ids:
                    idx_e = pred_ids.index(token2id["<eos>"]) + 1
                    pred_ids = pred_ids[:idx_e]

                # Convert both to token lists (strings) for comparison
                # (skip the leading <sos>)
                ref_tokens = [id2token[idx] for idx in gt_ids if idx != token2id["<sos>"]]
                hyp_tokens = [id2token[idx] for idx in pred_ids if idx != token2id["<sos>"]]

                # Compute edit distance and reference length
                edits = edit_distance(ref_tokens, hyp_tokens)
                total_edits += edits
                total_ref_len += len(ref_tokens)

                # Check exact match (excluding PAD_ID, but including <eos>)
                if ref_tokens == hyp_tokens:
                    exact_matches += 1

                total_samples += 1

    wer = total_edits / total_ref_len if total_ref_len > 0 else 0.0
    sacc = exact_matches / total_samples if total_samples > 0 else 0.0
    return wer, sacc


# —————————————————————————
# 2) Training & Validation Functions (unchanged, except .reshape)
# —————————————————————————
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)      # [B, 3, 224, 224]
        captions = captions.to(device)  # [B, MAX_SEQ_LEN]

        optimizer.zero_grad()

        decoder_input = captions[:, :-1]  # [B, MAX_SEQ_LEN-1]
        decoder_target = captions[:, 1:]  # [B, MAX_SEQ_LEN-1]

        logits = model(images, decoder_input)  # [B, MAX_SEQ_LEN-1, VOCAB_SIZE]

        # Flatten for loss
        logits_flat = logits.reshape(-1, VOCAB_SIZE)
        target_flat = decoder_target.reshape(-1)
        loss = criterion(logits_flat, target_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"  [Batch {batch_idx}/{len(dataloader)}]  Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            decoder_input = captions[:, :-1]
            decoder_target = captions[:, 1:]

            logits = model(images, decoder_input)
            logits_flat = logits.reshape(-1, VOCAB_SIZE)
            target_flat = decoder_target.reshape(-1)

            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# —————————————————————————
# 3) Main Script (wrapped in __main__)
# —————————————————————————
if __name__ == "__main__":
    # 3.1) DataLoaders
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # 3.2) Model, Loss, Optimizer
    model = ViTSeq2Seq(vocab_size=VOCAB_SIZE, d_model=768, max_len=MAX_SEQ_LEN).to(DEVICE)

    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": LR_ENCODER},
            {"params": decoder_params, "lr": LR_DECODER}
        ],
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(ignore_index=token2id["<pad>"])

    best_sacc = 0.0
    best_wer = float("inf")
    best_checkpoint = None

    # 3.3) Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"  >> Train Loss : {train_loss:.4f}")

        val_loss = validate(model, test_loader, criterion, DEVICE)
        print(f"  >> Val   Loss : {val_loss:.4f}")

        # 3.4) Compute WER and SACC on test set
        current_wer, current_sacc = compute_wer_and_sacc(model, test_loader, id2token, DEVICE)
        print(f"  >> WER : {current_wer:.4f}, SACC : {current_sacc:.4f}")

        # 3.5) Save only if SACC improves (or WER decreases if you prefer)
        if current_sacc > best_sacc:
            best_sacc = current_sacc
            best_wer = current_wer
            best_checkpoint = f"best_vit_seq2seq.pth"
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  >>> Saved new best model: {best_checkpoint} (SACC={best_sacc:.4f})")

    print("\nTraining complete.")
    print(f"Best SACC: {best_sacc:.4f}, Best WER: {best_wer:.4f} → Model saved as: {best_checkpoint}")

    # 3.6) Final Inference Example
    print("\n— Running a quick inference on the first test sample —")
    sample_img, _ = test_loader.dataset[0]
    sample_img = sample_img.unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    pred_ids = greedy_decode(model, sample_img, max_len=MAX_SEQ_LEN)
    pred_tokens = [id2token[i] for i in pred_ids]

    print("Predicted token IDs:", pred_ids)
    print("Predicted tokens   :", pred_tokens)
