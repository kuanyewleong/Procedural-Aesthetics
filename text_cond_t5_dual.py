import torch, torch.nn.functional as F
from typing import List, Optional
from transformers import T5Tokenizer, T5EncoderModel

class T5DualTextCond:
    def __init__(self, model_id: str = "t5-small", device: str = "cuda"):
        self.device = device
        self.tok = T5Tokenizer.from_pretrained(model_id)
        self.enc = T5EncoderModel.from_pretrained(model_id).to(device).eval()
        self.d_ctx = self.enc.config.d_model

    @torch.no_grad()
    def _encode(self, texts: List[str], max_length: int = 64) -> torch.Tensor:
        tok = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
        out = self.enc(**tok).last_hidden_state                    # [B,L,D]
        mask = tok.attention_mask.unsqueeze(-1)                    # [B,L,1]
        pooled = (out*mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)                         # [B,D]

    @torch.no_grad()
    def encode_dual(self, nat_texts, tech_texts=None, w_nat=0.5, w_tech=0.5):
        nat = self._encode(nat_texts)
        fused = nat if tech_texts is None else F.normalize(w_nat*nat + w_tech*self._encode(tech_texts), dim=-1)
        return fused.unsqueeze(1)  # [B,1,D]
