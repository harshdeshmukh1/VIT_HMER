# model.py

import torch
import torch.nn as nn
import timm

from config import VIT_MODEL_NAME, MAX_SEQ_LEN, token2id, VOCAB_SIZE, PAD_ID
from utils import create_decoder_masks

# —————————————————————————
# 1) ViTEncoder
# —————————————————————————
class ViTEncoder(nn.Module):
    """
    Wraps a pretrained timm ViT (vit_base_patch16_224). Strips the classifier head
    so forward(x) → [batch, 768].
    """
    def __init__(self, model_name=VIT_MODEL_NAME, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        # Remove classification head
        if hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, "fc"):
            self.vit.fc = nn.Identity()
        else:
            raise RuntimeError("Expected ViT to have head or fc attribute.")
        # (Optional) Projection layer; keep output dim=768 by default
        self.proj = nn.Identity()

    def forward(self, x):
        """
        x: [batch, 3, 224, 224]
        returns: [batch, 768]
        """
        feats = self.vit(x)       # [batch, 768]
        feats = self.proj(feats)  
        return feats


# —————————————————————————
# 2) TransformerDecoder
# —————————————————————————
class TransformerDecoder(nn.Module):
    """
    Standard PyTorch TransformerDecoder for seq-to-seq. Given:
      - tgt_ids [batch, seq_len]
      - memory  [batch, d_model]
    returns:
      logits [batch, seq_len, vocab_size]
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=768, nhead=8, num_layers=6, max_len=MAX_SEQ_LEN, pad_id=PAD_ID):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len

        # 1) Token embedding + scale
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # 2) Learned positional embeddings [1, max_len, d_model]
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        # 3) TransformerDecoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4) Final projection to vocab logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        # 5) Layer norm for decoder output
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tgt_ids, memory):
        """
        tgt_ids: [batch, seq_len]
        memory:  [batch, d_model]
        returns: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = tgt_ids.size()

        # 1) Embed tokens + add positional embeddings
        tok_emb = self.token_embedding(tgt_ids) * (self.d_model ** 0.5)  # [batch, seq_len, d_model]
        pos_emb = self.pos_embedding[:, :seq_len, :].to(tgt_ids.device)  # [1, seq_len, d_model]
        dec_input = tok_emb + pos_emb                                         # [batch, seq_len, d_model]

        # 2) Create masks
        tgt_mask, tgt_key_padding = create_decoder_masks(tgt_ids)
        # tgt_mask:        [seq_len, seq_len]
        # tgt_key_padding: [batch, seq_len]

        # 3) Reshape to [seq_len, batch, d_model], and memory to [1, batch, d_model]
        dec_input = dec_input.permute(1, 0, 2)  # [seq_len, batch, d_model]
        memory = memory.unsqueeze(0)           # [1, batch, d_model]

        # 4) Run through TransformerDecoder
        dec_output = self.transformer_decoder(
            tgt=dec_input,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding
        )  # → [seq_len, batch, d_model]

        # 5) Reshape back to [batch, seq_len, d_model], LayerNorm, project to vocab
        dec_output = dec_output.permute(1, 0, 2)  # [batch, seq_len, d_model]
        dec_output = self.ln(dec_output)           # [batch, seq_len, d_model]
        logits = self.output_proj(dec_output)      # [batch, seq_len, vocab_size]
        return logits


# —————————————————————————
# 3) Combined Seq2Seq Model
# —————————————————————————
class ViTSeq2Seq(nn.Module):
    """
    Vision Transformer encoder + Transformer decoder for image→token tasks.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=768, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.encoder = ViTEncoder(model_name=VIT_MODEL_NAME, pretrained=True)
        self.decoder = TransformerDecoder(vocab_size=vocab_size, d_model=d_model,
                                          nhead=8, num_layers=6, max_len=max_len, pad_id=PAD_ID)

    def forward(self, images, tgt_ids):
        """
        images:  [batch, 3, 224, 224]
        tgt_ids: [batch, seq_len]
        returns: [batch, seq_len, vocab_size]
        """
        memory = self.encoder(images)           # [batch, d_model]
        logits = self.decoder(tgt_ids, memory)   # [batch, seq_len, vocab_size]
        return logits


if __name__ == "__main__":
    # Sanity-check
    model = ViTSeq2Seq(vocab_size=VOCAB_SIZE, d_model=768, max_len=MAX_SEQ_LEN)
    dummy_imgs = torch.randn(2, 3, 224, 224)
    dummy_tgt = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ_LEN))
    out = model(dummy_imgs, dummy_tgt)
    print("Output shape:", out.shape)  # expect [2, MAX_SEQ_LEN, VOCAB_SIZE]
