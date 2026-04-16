"""Focal loss + label smoothing for PubMedBERT extractive QA, and a Trainer subclass.

Setting gamma=0 and label_smoothing=0 recovers plain cross-entropy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class FocalLabelSmoothingLoss(nn.Module):
    """Focal loss composed with label smoothing, for single-token classification.

    Used on each of start/end logit vectors independently. Shape conventions::

        logits : [batch, seq_len]
        target : [batch]               gold position index in [0, seq_len)

    Args:
        gamma            : focal focusing parameter. 0 means no reweighting.
        label_smoothing  : epsilon for uniform smoothing. 0 means no smoothing.
        ignore_index     : positions labelled with this value are masked out.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not valid.any():
            return logits.new_zeros(())
        logits = logits[valid]
        target = target[valid]

        num_classes = logits.size(-1)
        target = target.clamp(min=0, max=num_classes - 1)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        gather_idx = target.unsqueeze(-1)
        pt = probs.gather(-1, gather_idx).squeeze(-1)
        nll = -log_probs.gather(-1, gather_idx).squeeze(-1)

        eps = self.label_smoothing
        if eps > 0:
            smooth = -log_probs.mean(dim=-1)
            ce_term = (1.0 - eps) * nll + eps * smooth
        else:
            ce_term = nll

        if self.gamma > 0:
            focal_w = (1.0 - pt).pow(self.gamma)
            loss = focal_w * ce_term
        else:
            loss = ce_term

        return loss.mean()


class QASpanTrainer(Trainer):
    """HuggingFace Trainer subclass that swaps CE for FocalLabelSmoothingLoss
    on both start and end logits.

    Usage::

        trainer = QASpanTrainer(
            model=model, args=training_args, train_dataset=..., eval_dataset=...,
            loss_fn=FocalLabelSmoothingLoss(gamma=2.0, label_smoothing=0.1),
        )
        trainer.train()
    """

    def __init__(self, *args, loss_fn: nn.Module | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn if loss_fn is not None else FocalLabelSmoothingLoss()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        start_positions = inputs.pop("start_positions", None)
        end_positions = inputs.pop("end_positions", None)

        outputs = model(**inputs)

        if start_positions is None or end_positions is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        start_loss = self.loss_fn(outputs.start_logits, start_positions)
        end_loss = self.loss_fn(outputs.end_logits, end_positions)
        loss = (start_loss + end_loss) / 2.0

        return (loss, outputs) if return_outputs else loss
