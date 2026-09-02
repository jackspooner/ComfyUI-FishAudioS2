from types import SimpleNamespace

import pytest
import torch

from fish_speech.models.text2semantic.inference import decode_n_tokens


def test_semantic_decode_checks_comfy_interruption_before_each_token():
    class Interrupted(Exception):
        pass

    decode_calls = []
    model = SimpleNamespace(
        config=SimpleNamespace(num_codebooks=2),
        tokenizer=SimpleNamespace(get_token_id=lambda _: 1),
    )

    with pytest.raises(Interrupted):
        decode_n_tokens(
            model=model,
            cur_token=torch.zeros((1, 3, 1), dtype=torch.int),
            input_pos=torch.tensor([0]),
            num_new_tokens=4,
            temperature=torch.tensor(0.8),
            top_p=torch.tensor(0.8),
            top_k=30,
            semantic_logit_bias=torch.zeros((1, 1, 2)),
            audio_masks=torch.zeros((1, 1), dtype=torch.bool),
            audio_parts=torch.zeros((1, 1), dtype=torch.bool),
            decode_one_token=lambda **_: decode_calls.append(True),
            interruption_check=lambda: (_ for _ in ()).throw(Interrupted()),
        )

    assert decode_calls == []
