# Third-party notices

## ByteNet (src/bgcbench/encoders/bytenet.py)

Adapted from [microsoft/protein-sequence-models](https://github.com/microsoft/protein-sequence-models),
commit `af695772c4a1c056d930c95ec7e6428aa042f5cd` (`sequence_models/convolutional.py`,
`sequence_models/layers.py`, and `_pad` from `sequence_models/collaters.py`), reduced to the
inference path BiGCARP uses. Its licence:

```
Copyright (c) 2022
Microsoft [All rights reserved].

1. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
THIS SOFTWARE IS PROVIDED BY [Name of Organization] “AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL [Name of Organisation] BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Model weights and data

No model weights or data are part of this repository. They are downloaded at pinned
revisions from Hugging Face, each under its own licence (see the licence table in README.md
and each model card): the BGC-Bench dataset (CC BY 4.0, built from MIBiG 4.0), the mirrors
of BiGCARP (MIT), DeepBGC (MIT), Kawano et al. Model I (CC BY-NC 4.0) and LC-PLM
(CC BY-NC 4.0), and the upstream repositories of ESM-2 (MIT), ESM C 600M (released under the
EvolutionaryScale Cambrian Non-Commercial License; the repository, now
`biohub/esmc-600m-2024-12`, currently lists MIT) and Evo 2 7B (Apache-2.0).

## ESM

The `bgcat` encoder is Built with ESM. It runs ESM C 600M through the `esm` package
(github.com/evolutionaryscale/esm, pinned commit `453b2e2e713ee35a1dcbc684b11d03303a59294e`
in `pyproject.toml`), a runtime dependency under the Cambrian Open License at that commit.
