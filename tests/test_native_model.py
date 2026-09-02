import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic import llama


def tiny_config():
    return llama.DualARModelArgs(
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        head_dim=4,
        dim=16,
        intermediate_size=32,
        max_seq_len=32,
        codebook_size=8,
        num_codebooks=2,
        n_fast_layer=1,
        fast_dim=16,
        fast_n_head=4,
        fast_n_local_heads=2,
        fast_head_dim=4,
        fast_intermediate_size=32,
        attention_qk_norm=True,
        use_gradient_checkpointing=False,
    )


def _reference_attention(module, x, freqs, mask):
    batch, length, _ = x.shape
    q_size = module.n_head * module.head_dim
    kv_size = module.n_local_heads * module.head_dim
    q, k, v = module.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
    q = module.q_norm(q.view(batch, length, module.n_head, module.head_dim))
    k = module.k_norm(k.view(batch, length, module.n_local_heads, module.head_dim))
    v = v.view(batch, length, module.n_local_heads, module.head_dim)
    q = llama.apply_rotary_emb(q, freqs).transpose(1, 2)
    k = llama.apply_rotary_emb(k, freqs).transpose(1, 2)
    v = v.transpose(1, 2)
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    return module.wo(output.transpose(1, 2).reshape(batch, length, q_size))


def test_attention_matches_sdpa_prefill():
    torch.manual_seed(4)
    attention = llama.Attention(tiny_config()).eval()
    x = torch.randn(1, 5, 16)
    freqs = llama.precompute_freqs_cis(5, 4)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
    expected = _reference_attention(attention, x, freqs, mask)
    actual = attention(x, freqs, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_shipped_fish_attention_policy_is_preserved():
    model = llama.DualARTransformer(tiny_config()).eval()
    assert all(layer.attention.use_sdpa for layer in model.layers)
    assert all(not layer.attention.use_sdpa for layer in model.fast_layers)
    assert "causal_mask" not in dict(model.named_buffers())
