import pytest
import torch
from multimodal_core import (
    ToyPatchVisionEncoder,
    ToyTextEncoder,
    CLIPModel,
    clip_loss,
    zero_shot_classifier,
    VLMProjector,
    build_multimodal_input,
    AudioFrameProjector,
    DiffusionNoiseSchedule,
    classifier_free_guidance,
    MultimodalStore
)


def test_clip_forward_and_loss():
    torch.manual_seed(42)
    B = 4
    images = torch.randn(B, 3, 16, 16)
    texts = torch.randint(0, 100, (B, 8))

    model = CLIPModel(d_vision=32, d_text=32, embed_dim=16)
    logits_img, logits_txt = model(images, texts)

    assert logits_img.shape == (B, B)
    assert logits_txt.shape == (B, B)

    loss = clip_loss(logits_img, logits_txt)
    assert loss.item() > 0.0

    # Loss is differentiable
    loss.backward()
    assert model.v_proj.weight.grad is not None
    assert model.t_proj.weight.grad is not None


def test_zero_shot_classification():
    torch.manual_seed(0)
    model = CLIPModel(d_vision=32, d_text=32, embed_dim=16)
    images = torch.randn(2, 3, 16, 16)
    candidate_tokens = torch.randint(0, 100, (3, 6))  # 3 candidate classes

    probs = zero_shot_classifier(model, images, candidate_tokens)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_vlm_projector_and_prefix_injection():
    B, num_patches, d_v, d_llm = 2, 16, 32, 48
    patch_tokens = torch.randn(B, num_patches, d_v)

    proj = VLMProjector(d_vision=d_v, d_llm=d_llm)
    projected = proj(patch_tokens)
    assert projected.shape == (B, num_patches, d_llm)

    text_tokens = torch.randn(B, 10, d_llm)
    multimodal_seq = build_multimodal_input(projected, text_tokens)
    assert multimodal_seq.shape == (B, num_patches + 10, d_llm)


def test_audio_projector():
    B, n_mels, time_steps, d_llm = 2, 80, 50, 48
    mel = torch.randn(B, n_mels, time_steps)

    aud_proj = AudioFrameProjector(n_mels=n_mels, d_llm=d_llm, stride=2)
    tokens = aud_proj(mel)
    # Stride 2 downsamples time from 50 to 25
    assert tokens.shape == (B, 25, d_llm)


def test_diffusion_noise_schedule_and_cfg():
    sched = DiffusionNoiseSchedule(num_timesteps=100)
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.tensor([0, 50, 99, 10])

    xt = sched.q_sample(x0, t)
    assert xt.shape == x0.shape

    # CFG formula verification
    eps_uncond = torch.tensor([1.0, 2.0])
    eps_cond = torch.tensor([2.0, 4.0])
    guided = classifier_free_guidance(eps_uncond, eps_cond, guidance_scale=3.0)
    # 1.0 + 3.0 * (2.0 - 1.0) = 4.0; 2.0 + 3.0 * (4.0 - 2.0) = 8.0
    assert torch.allclose(guided, torch.tensor([4.0, 8.0]))


def test_multimodal_store():
    store = MultimodalStore()
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])
    v3 = torch.tensor([0.7, 0.7, 0.0])

    store.add("text_1", "text", "Document about cats", v1)
    store.add("img_1", "image", "cat_photo.jpg", v3)
    store.add("text_2", "text", "Article on astrophysics", v2)

    res = store.search(torch.tensor([1.0, 0.0, 0.0]), top_k=2)
    assert len(res) == 2
    assert res[0][0].id == "text_1"
    assert res[1][0].id == "img_1"

    # Filter modality
    img_only = store.search(torch.tensor([1.0, 0.0, 0.0]), top_k=2, filter_modality="image")
    assert len(img_only) == 1
    assert img_only[0][0].id == "img_1"
