# coding=utf-8
"""Formula-level tests for DFlash/D-PACE loss behavior.

The tests load ``specforge/core/dflash.py`` directly with a lightweight draft
model stub so they can run on CPU without importing the full modeling stack.
They still exercise ``OnlineDFlashModel.forward``: anchor sampling, draft
output, and LM head output are made deterministic.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]

_pkg_specforge = types.ModuleType("specforge")
_pkg_specforge.__path__ = [str(REPO / "specforge")]
sys.modules.setdefault("specforge", _pkg_specforge)

_pkg_modeling = types.ModuleType("specforge.modeling")
_pkg_modeling.__path__ = [str(REPO / "specforge" / "modeling")]
sys.modules.setdefault("specforge.modeling", _pkg_modeling)

_pkg_draft = types.ModuleType("specforge.modeling.draft")
_pkg_draft.__path__ = [str(REPO / "specforge" / "modeling" / "draft")]
sys.modules.setdefault("specforge.modeling.draft", _pkg_draft)

_stub_dflash_draft = types.ModuleType("specforge.modeling.draft.dflash")


class _DFlashDraftStub(nn.Module):
    pass


_stub_dflash_draft.DFlashDraftModel = _DFlashDraftStub
sys.modules["specforge.modeling.draft.dflash"] = _stub_dflash_draft

_spec = importlib.util.spec_from_file_location(
    "specforge.core.dflash", REPO / "specforge" / "core" / "dflash.py"
)
_dflash_module = importlib.util.module_from_spec(_spec)
sys.modules["specforge.core.dflash"] = _dflash_module
_spec.loader.exec_module(_dflash_module)
OnlineDFlashModel = _dflash_module.OnlineDFlashModel


class _FixedDraft(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        bsz, draft_len = noise_embedding.shape[:2]
        return torch.zeros(
            bsz,
            draft_len,
            self.hidden_size,
            dtype=noise_embedding.dtype,
            device=noise_embedding.device,
        )


class _FixedHead(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, hidden_states):
        return self.fixed_logits.to(device=hidden_states.device)


def _fixed_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
    bsz, n_blocks = anchor_positions.shape
    return torch.zeros(
        bsz,
        n_blocks * self.block_size,
        self.embed_tokens.embedding_dim,
        dtype=torch.double,
        device=input_ids.device,
    )


def _fixed_anchor_sampler(anchors, keep_mask):
    def _sample(self, seq_len, loss_mask, device):
        return anchors.to(device), keep_mask.to(device)

    return _sample


def _make_model(logits, anchors, keep_mask, **kwargs):
    bsz, n_blocks, block_size, vocab_size = logits.shape
    model = OnlineDFlashModel(
        draft_model=_FixedDraft(hidden_size=4),
        target_lm_head=_FixedHead(
            logits.reshape(bsz, n_blocks * block_size, vocab_size)
        ),
        target_embed_tokens=nn.Embedding(vocab_size, 4).double(),
        mask_token_id=0,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=n_blocks,
        **kwargs,
    ).double()
    model._sample_anchor_positions = types.MethodType(
        _fixed_anchor_sampler(anchors, keep_mask), model
    )
    model._create_noise_embed = types.MethodType(_fixed_noise_embed, model)
    return model


def _sample_tensors():
    torch.manual_seed(123)
    bsz, n_blocks, block_size, vocab_size = 2, 2, 5, 13
    seq_len = 9
    logits = torch.randn(
        bsz, n_blocks, block_size, vocab_size, dtype=torch.double
    )
    input_ids = torch.tensor(
        [
            [1, 4, 2, 8, 3, 7, 5, 6, 9],
            [2, 5, 1, 4, 7, 3, 8, 10, 11],
        ],
        dtype=torch.long,
    )
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.double)
    loss_mask[0, 7] = 0.0
    loss_mask[1, 6] = 0.0
    anchors = torch.tensor([[0, 3], [1, 4]], dtype=torch.long)
    keep_mask = torch.tensor([[True, True], [True, False]])
    hidden_states = torch.zeros(bsz, seq_len, 4, dtype=torch.double)
    return logits, input_ids, loss_mask, hidden_states, anchors, keep_mask


def _targets_and_mask(input_ids, loss_mask, anchors, keep_mask, block_size):
    bsz, seq_len = input_ids.shape
    n_blocks = anchors.shape[1]
    offsets = torch.arange(block_size).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    targets = torch.gather(
        input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
        2,
        safe_indices,
    )
    binary_mask = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).double()
    binary_mask = binary_mask * (label_indices < seq_len).double()
    binary_mask = binary_mask * (offsets > 0).double()
    gathered_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
        2,
        safe_indices,
    )
    binary_mask = binary_mask * gathered_loss_mask
    return targets, binary_mask


def _neg_log_q(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)


def _naive_dpace_weight(prob, binary_mask, alpha, variant):
    smooth = (1.0 - alpha) * prob + alpha
    smooth = torch.where(binary_mask > 0, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)
    if variant == "p":
        return prefix
    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
        dims=[-1],
    )
    if variant == "full":
        return suffix
    if variant == "f":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(variant)


def _naive_dflash_loss(neg_log_q, binary_mask, gamma):
    weight = binary_mask
    if gamma is not None and gamma > 0:
        block_size = neg_log_q.shape[-1]
        positions = torch.arange(block_size, dtype=neg_log_q.dtype).view(1, 1, -1)
        decay = torch.exp(-(positions - 1).clamp(min=0) / gamma)
        weight = weight * decay
    return (neg_log_q * weight).sum() / (weight.sum() + 1e-6)


class TestDFlashLosses(unittest.TestCase):
    def setUp(self):
        (
            self.logits,
            self.input_ids,
            self.loss_mask,
            self.hidden_states,
            self.anchors,
            self.keep_mask,
        ) = _sample_tensors()
        self.targets, self.binary_mask = _targets_and_mask(
            self.input_ids,
            self.loss_mask,
            self.anchors,
            self.keep_mask,
            self.logits.shape[2],
        )
        self.neg_log_q = _neg_log_q(self.logits, self.targets)
        self.q = torch.exp(-self.neg_log_q)

    def _forward_loss(self, **kwargs):
        model = _make_model(self.logits, self.anchors, self.keep_mask, **kwargs)
        loss, accuracy = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        return loss

    def test_dflash_default_matches_existing_weighted_mean(self):
        got = self._forward_loss()
        want = _naive_dflash_loss(self.neg_log_q, self.binary_mask, gamma=None)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_dflash_decay_gamma_is_preserved(self):
        gamma = 7.0
        got = self._forward_loss(loss_type="dflash", loss_decay_gamma=gamma)
        want = _naive_dflash_loss(self.neg_log_q, self.binary_mask, gamma=gamma)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-8)

    def test_dpace_full_matches_naive_reference(self):
        alpha = 0.5
        got = self._forward_loss(loss_type="dpace", dpace_alpha=alpha)
        weight = _naive_dpace_weight(self.q, self.binary_mask, alpha, "full")
        want = (self.neg_log_q * weight * self.binary_mask).sum() / float(
            self.input_ids.shape[0]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_dpace_p_matches_naive_reference(self):
        alpha = 0.5
        got = self._forward_loss(loss_type="dpace_p", dpace_alpha=alpha)
        weight = _naive_dpace_weight(self.q, self.binary_mask, alpha, "p")
        want = (self.neg_log_q * weight * self.binary_mask).sum() / float(
            self.input_ids.shape[0]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_dpace_f_matches_naive_reference(self):
        alpha = 0.5
        got = self._forward_loss(loss_type="dpace_f", dpace_alpha=alpha)
        weight = _naive_dpace_weight(self.q, self.binary_mask, alpha, "f")
        want = (self.neg_log_q * weight * self.binary_mask).sum() / float(
            self.input_ids.shape[0]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_dpace_loss_reduces_by_batch_size(self):
        alpha = 0.5
        got = self._forward_loss(loss_type="dpace", dpace_alpha=alpha)
        weight = _naive_dpace_weight(self.q, self.binary_mask, alpha, "full")
        weighted_sum = (self.neg_log_q * weight * self.binary_mask).sum()
        token_count_loss = weighted_sum / (
            (weight * self.binary_mask).sum() + 1e-6
        )
        batch_loss = weighted_sum / float(self.input_ids.shape[0])
        torch.testing.assert_close(got, batch_loss, rtol=0, atol=1e-10)
        self.assertFalse(torch.allclose(got, token_count_loss))

    def test_alpha_changes_dpace_loss(self):
        low_alpha = self._forward_loss(loss_type="dpace", dpace_alpha=0.1)
        high_alpha = self._forward_loss(loss_type="dpace", dpace_alpha=0.9)
        self.assertNotAlmostEqual(low_alpha.item(), high_alpha.item(), places=8)

    def test_invalid_loss_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "loss_type"):
            _make_model(
                self.logits,
                self.anchors,
                self.keep_mask,
                loss_type="topk_mask",
            )

    def test_invalid_dpace_alpha_rejected(self):
        with self.assertRaisesRegex(ValueError, "dpace_alpha"):
            _make_model(
                self.logits,
                self.anchors,
                self.keep_mask,
                loss_type="dpace",
                dpace_alpha=1.5,
            )


if __name__ == "__main__":
    unittest.main()
