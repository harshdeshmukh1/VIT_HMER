# utils.py

import torch
import torch.nn as nn

from config import PAD_ID, SOS_ID, EOS_ID, token2id, id2token, VOCAB_SIZE

def create_decoder_masks(tgt_ids):
    """
    Given tgt_ids: LongTensor [batch, seq_len],
    returns:
      - tgt_mask:         [seq_len, seq_len]  (causal mask)
      - tgt_key_padding:  [batch, seq_len]    (True at PAD positions)
    """
    batch_size, seq_len = tgt_ids.size()
    device = tgt_ids.device

    # 1) Causal mask so each position only attends to previous ones
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

    # 2) Padding mask (True where token == PAD_ID)
    tgt_key_padding = (tgt_ids == PAD_ID)

    return tgt_mask, tgt_key_padding


def greedy_decode(model, image_tensor, max_len=50):
    """
    Greedily generate a token sequence from a single image.

    Args:
      - model:         ViTSeq2Seq instance
      - image_tensor:  [1, 3, 224, 224] (already normalized)
      - max_len:       maximum output length (incl. SOS and EOS)

    Returns:
      - generated_ids: Python list of token IDs (with SOS_ID, EOS_ID)
    """
    model.eval()
    generated = [SOS_ID]

    with torch.no_grad():
        for _ in range(max_len - 1):
            # Prepare partial sequence
            tgt_ids = torch.tensor(generated, dtype=torch.long, device=image_tensor.device).unsqueeze(0)
            logits = model(image_tensor, tgt_ids)         # [1, cur_len, VOCAB_SIZE]
            next_logits = logits[0, -1, :]                 # [VOCAB_SIZE]
            next_token = next_logits.argmax().item()
            generated.append(next_token)
            if next_token == EOS_ID:
                break

    return generated


if __name__ == "__main__":
    # Quick check
    print(f"PAD_ID={PAD_ID}, SOS_ID={SOS_ID}, EOS_ID={EOS_ID}, VOCAB_SIZE={VOCAB_SIZE}")
    # Check that functions import correctly
    dummy = torch.randint(0, VOCAB_SIZE, (2, 10))
    tgt_mask, tgt_pad = create_decoder_masks(dummy)
    print(f"tgt_mask shape: {tgt_mask.shape}, tgt_pad shape: {tgt_pad.shape}")
