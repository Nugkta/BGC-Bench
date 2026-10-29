"""Encoder input formats, vocabulary coverage and A-domain token alignment."""

import re

import pytest

from bgcbench.encoders.base import REGISTRY, adomain_table, load_inputs

ACC = re.compile(r"^PF\d{5}$")


def test_encoder_input_kinds():
    assert {n for n, s in REGISTRY.items() if s.input == "pfam_named"} == {"bigcarp", "kawano"}
    assert {n for n, s in REGISTRY.items() if s.input == "pfam_acc"} == {
        "bag_of_pfam", "pfam_rand", "deepbgc"}


def test_pfam_inputs(dataset):
    acc, named = load_inputs("pfam_acc"), load_inputs("pfam_named")
    assert len(acc) == 2631 and len(named) == 2626
    assert sum(1 for v in acc.values() if v) == 2626
    assert all(ACC.match(p) for v in acc.values() for p in v)
    tokens = [p for v in named.values() for p in v]
    assert sum(1 for p in tokens if ACC.match(p)) < 0.01 * len(tokens)


def test_adomain_token_indices_point_at_the_a_domain(dataset):
    tab = adomain_table()
    acc, named = load_inputs("pfam_acc"), load_inputs("pfam_named")
    sub = tab[tab["token_index"] >= 0]
    assert len(sub) > 2000
    assert all(named[b][t] == "AMP-binding" for b, t in zip(sub["bgc_id"], sub["token_index"]))
    raw = tab[tab["token_index_raw"] >= 0]
    assert all(acc[b][t] == "PF00501" for b, t in zip(raw["bgc_id"], raw["token_index_raw"]))
    assert tab["nt_seq"].notna().all()
    assert not {"substrate", "substrate_group", "gcf_id"} & set(tab.columns)  # no labels


def test_bigcarp_unknown_token_rate_on_named_input(dataset):
    from bgcbench.encoders.bigcarp import BiGCARP
    from bgcbench.encoders.hub import model_file

    try:
        for f in ("config.json", "vocab.json", "model.safetensors"):
            model_file("bigcarp", f)
    except Exception as exc:  # no network
        pytest.skip(f"BiGCARP weights not available: {exc}")
    enc = BiGCARP(device="cpu")
    named = load_inputs("pfam_named")
    unk = enc.domains["UNK"]
    ids = [int(t) for v in named.values() if v for t in enc.tokenize(v)]
    rate = sum(t == unk for t in ids) / len(ids)
    # 9.8 % of tokens are outside BiGCARP's 2022 vocabulary; accessions would give 100 %
    assert 0.05 < rate < 0.15


def test_evo2_requires_logical_gpu_zero_before_loading_weights():
    from bgcbench.encoders.evo2 import Evo2

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        Evo2(device="cuda:1")
