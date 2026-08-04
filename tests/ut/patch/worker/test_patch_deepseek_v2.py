# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _filter_unregistered_ascend_auxiliary_weights,
    _should_skip_indexer_init,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_dense_checkpoint_only_alpha_is_filtered():
    weight = torch.ones(1)
    weights = [
        ("layers.0.mlp.down_proj.alpha", weight),
        ("layers.0.mlp.down_proj.weight", weight),
    ]

    filtered = list(
        _filter_unregistered_ascend_auxiliary_weights(
            weights,
            {"layers.0.mlp.down_proj.weight"},
        )
    )

    assert filtered == [("layers.0.mlp.down_proj.weight", weight)]


def test_registered_or_expert_alpha_is_preserved():
    weight = torch.ones(1)
    registered_alpha = "layers.0.mlp.down_proj.alpha"
    expert_alpha = "layers.3.mlp.experts.0.down_proj.alpha"
    unrelated_unknown = "layers.0.mlp.unknown_metadata"
    weights = [
        (registered_alpha, weight),
        (expert_alpha, weight),
        (unrelated_unknown, weight),
    ]

    filtered = list(
        _filter_unregistered_ascend_auxiliary_weights(
            weights,
            {registered_alpha},
        )
    )

    assert filtered == weights


def test_unregistered_c8_kv_metadata_is_filtered():
    weight = torch.ones(1)
    ckv_a_alpha = "layers.0.self_attn.ckv_a_alpha"
    indexer_hadamard = "layers.0.self_attn.indexer.hadamard_matrix"
    weights = [
        (ckv_a_alpha, weight),
        (indexer_hadamard, weight),
    ]

    filtered = list(
        _filter_unregistered_ascend_auxiliary_weights(weights, set())
    )

    assert filtered == []


def test_registered_c8_kv_metadata_is_preserved():
    weight = torch.ones(1)
    ckv_a_alpha = "layers.0.self_attn.ckv_a_alpha"
    indexer_hadamard = "layers.0.self_attn.indexer.hadamard_matrix"
    weights = [
        (ckv_a_alpha, weight),
        (indexer_hadamard, weight),
    ]

    filtered = list(
        _filter_unregistered_ascend_auxiliary_weights(
            weights,
            {ckv_a_alpha, indexer_hadamard},
        )
    )

    assert filtered == weights
