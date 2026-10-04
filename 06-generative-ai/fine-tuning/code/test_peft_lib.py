import math
import pytest
import torch
import torch.nn as nn
from peft_lib import *


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.other = nn.Linear(16, 4)

    def forward(self, x):
        return self.other(torch.relu(self.q_proj(x)) + self.v_proj(x))


def test_lora_starts_identical_and_counts_params():
    torch.manual_seed(0)
    base = nn.Linear(32, 24)
    x = torch.randn(5, 32)
    lora = LoRALinear(base, r=4, alpha=8)
    assert torch.allclose(lora(x), base(x))                       # B = 0
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert trainable == 4 * (32 + 24)
    with pytest.raises(ValueError):
        LoRALinear(base, r=0)


def test_merge_equals_unmerged():
    torch.manual_seed(1)
    lora = LoRALinear(nn.Linear(16, 12), r=3, alpha=6)
    with torch.no_grad():
        lora.B.normal_()
    x = torch.randn(7, 16)
    assert torch.allclose(lora(x), lora.merged()(x), atol=1e-5)


def test_apply_lora_freezes_base_and_trains_only_adapters():
    torch.manual_seed(2)
    m = Toy()
    n = apply_lora(m, {"q_proj", "v_proj"}, r=2, alpha=4)
    assert n == 2
    names = {k for k, p in m.named_parameters() if p.requires_grad}
    assert names == {"q_proj.A", "q_proj.B", "v_proj.A", "v_proj.B"}
    before = m.other.weight.clone()
    x, y = torch.randn(64, 16), torch.randn(64, 4)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = nn.functional.mse_loss(m(x), y)
        loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert torch.equal(m.other.weight, before)                    # untouched
    t, total = count_params(m)
    assert t < total
    merge_lora(m)
    assert not any(isinstance(c, LoRALinear) for c in m.modules())


def test_adapter_zero_init_is_identity():
    a = BottleneckAdapter(16, 4)
    x = torch.randn(3, 16)
    assert torch.allclose(a(x), x)


def test_quantization_levels_and_error():
    torch.manual_seed(3)
    nf, un = nf_style_levels(), uniform_levels()
    assert nf.numel() == 16 and float(nf.abs().max()) == 1.0 and torch.all(nf[1:] > nf[:-1])
    w = torch.randn(64, 128)
    errs = {}
    for name, lv in (("nf", nf), ("uni", un)):
        codes, scales, shape = blockwise_quantize(w, lv, block=64)
        assert codes.dtype == torch.uint8 and int(codes.max()) < 16
        rec = blockwise_dequantize(codes, scales, shape, lv)
        assert rec.shape == w.shape
        errs[name] = float(((rec - w) ** 2).mean())
    assert errs["nf"] < errs["uni"]            # quantile levels suit Gaussian-like weights
    assert errs["nf"] < 0.02


def test_quantization_handles_non_multiple_of_block():
    w = torch.randn(5, 13)
    lv = nf_style_levels()
    codes, scales, shape = blockwise_quantize(w, lv, block=16)
    assert blockwise_dequantize(codes, scales, shape, lv).shape == (5, 13)


def test_qlora_layer_trains_only_lora_and_saves_memory():
    torch.manual_seed(4)
    base = nn.Linear(128, 64)
    q = QLoRALinear(base, r=4, alpha=8)
    x = torch.randn(8, 128)
    assert float((q(x) - base(x)).abs().mean().detach()) < 0.1  # close to the fp32 layer
    q(x).sum().backward()
    assert q.A.grad is not None and q.B.grad is not None
    quant_bytes, fp16_bytes = q.stored_bytes()
    assert quant_bytes < fp16_bytes / 3


def test_sft_masks_prompt_tokens():
    ids, labels = build_sft_example([5, 6, 7], [8, 9])
    assert labels == [-100, -100, -100, 8, 9]
    logits = torch.randn(1, 5, 20)
    lab = torch.tensor([labels])
    full = causal_lm_loss(logits, lab)
    # changing logits at a position that predicts a masked token must not change the loss
    logits2 = logits.clone(); logits2[0, 0, 3] += 100.0         # position 0 predicts token 1 (masked)
    assert torch.allclose(full, causal_lm_loss(logits2, lab))
    logits3 = logits.clone(); logits3[0, 2, 8] += 5.0           # position 2 predicts token 3 (label 8, supervised)
    assert not torch.allclose(full, causal_lm_loss(logits3, lab))


def test_sequence_logprob_matches_manual():
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[-100, 2, 3, 1]])
    lp = torch.log_softmax(logits, -1)
    manual = lp[0, 0, 2] + lp[0, 1, 3] + lp[0, 2, 1]
    assert torch.allclose(sequence_logprob(logits, labels)[0], manual, atol=1e-6)


def test_dpo_properties():
    z = torch.zeros(4)
    assert math.isclose(float(dpo_loss(z, z, z, z)), math.log(2), rel_tol=1e-6)    # policy == reference
    better = dpo_loss(z + 2, z - 2, z, z)                      # policy prefers chosen
    worse = dpo_loss(z - 2, z + 2, z, z)
    assert better < math.log(2) < worse
    # larger beta sharpens the effect
    assert dpo_loss(z + 2, z - 2, z, z, beta=1.0) < better


def test_rlhf_formulas():
    assert reward_model_loss(torch.tensor([2.0]), torch.tensor([-2.0])) < reward_model_loss(torch.tensor([-2.0]), torch.tensor([2.0]))
    r = kl_shaped_reward(torch.tensor([1.0]), torch.tensor([-1.0]), torch.tensor([-3.0]), beta=0.5)
    assert math.isclose(float(r), 0.0)                          # 1 - 0.5 * 2
    adv = torch.tensor([1.0])
    big = ppo_clip_objective(torch.tensor([2.0]), torch.tensor([0.0]), adv)
    assert math.isclose(float(big), 1.2, rel_tol=1e-6)         # ratio clipped at 1 + eps
