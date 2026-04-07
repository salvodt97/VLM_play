# VLM Play

A compact PyTorch implementation of a vision-language model inspired by PaliGemma.

This repository brings together:

- a `SigLIP`-style vision encoder
- a `Gemma`-style language decoder
- a multimodal projection layer
- autoregressive generation with KV cache

The goal is simple: load a local checkpoint, feed an image plus a text prompt, and generate a textual response.

## Why This Repo

`VLM Play` is built to be readable and hackable.

It is a good fit if you want to:

- study how a VLM is assembled end to end
- inspect image-token fusion in a compact codebase
- experiment with local PaliGemma-style checkpoints
- debug inference issues without hiding behind large frameworks

## Features

- Clean separation between vision, language, and multimodal fusion
- Local Hugging Face checkpoint loading
- Image preprocessing pipeline for PaliGemma-style inputs
- Greedy decoding and top-p sampling
- KV cache support for autoregressive generation
- Lightweight shell entrypoint for quick experiments

## Project Layout

- `inference.py` — inference entrypoint
- `launch_inference.sh` — quick launcher for local runs
- `final_model.py` — multimodal model wrapper
- `ViT/siglip.py` — vision tower
- `LLM/gemma.py` — decoder-only language model
- `LLM/kvcache.py` — KV cache implementation
- `ImgProcessing/processing_images.py` — image + prompt processor
- `generic/generic_functions.py` — checkpoint loading utilities

## Installation

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `numpy`
- `pillow`
- `fire`
- `transformers`
- `safetensors`

## Model Checkpoints

The project expects local checkpoints inside `HuggingFaceModel/`.

Typical directories:

- `HuggingFaceModel/paligemma-3b-pt-224`
- `HuggingFaceModel/paligemma-3b-mix-224`

## Quick Start

Run the default launcher:

```bash
bash launch_inference.sh
```

Or call inference directly:

```bash
python3 inference.py \
  --model_path HuggingFaceModel/paligemma-3b-pt-224 \
  --prompt "caption en" \
  --image_file_path test_images/image2.jpg \
  --max_tokens_to_generate 100 \
  --temperature 0.8 \
  --top_p 0.9 \
  --do_sample False \
  --only_cpu False
```

The script automatically selects `cuda`, `mps`, or `cpu`, depending on what is available.

## Prompting Tips

### With `paligemma-3b-pt-224`

Use task-style prompts such as:

- `caption en`
- `caption it`
- `detect`
- `segment`

This checkpoint is much happier with explicit task prefixes than with open-ended natural prompts.

### With `paligemma-3b-mix-224`

If you want more flexible prompts such as:

- `Describe this image.`
- `What is happening here?`
- `The building is`

then a `mix` checkpoint is usually the better choice.

## Common Pitfall: Broken `mix` Tokenizer Files

If `paligemma-3b-mix-224` fails with an error like:

```text
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

your tokenizer files were likely not downloaded correctly.

A common cause is Git LFS pointers being present instead of the real files.

Typical signs:

- `tokenizer.json` is only a few bytes long
- the file starts with `version https://git-lfs.github.com/spec/v1`

If that happens, re-download the model properly with Git LFS or Hugging Face tooling.

## Debugging Checkpoint Loading

The loader prints any checkpoint mismatch it finds, including:

- `Missing keys`
- `Unexpected keys`

These messages are extremely useful when debugging partially loaded models.

For example, a single missing key such as:

```text
language_model.lm_head.weight
```

can be acceptable in this codebase because the output head is tied to the input embeddings right after loading.

## When Outputs Look Wrong

If the model generates text that looks broken, repetitive, or pseudo-random, the most likely causes are:

- the checkpoint does not match the prompting style
- some model weights were not loaded due to naming mismatches
- tokenizer files are incomplete or corrupted
- a `pt` checkpoint is being used with very free-form prompts

## Current Status

At the moment, the repository is able to:

- load the local `paligemma-3b-pt-224` checkpoint
- run image-plus-text inference end to end
- produce coherent outputs with prompts such as `caption en`

## Next Improvements

- stronger support for `mix` checkpoints
- explicit validation for missing Git LFS assets
- batch inference beyond the current `1 image + 1 prompt` flow
- cleaner separation between input prompt and generated completion
- automated checks for checkpoint key alignment

## Notes

This is a learning-oriented implementation, not a drop-in replacement for the official Transformers stack.

If you want maximum robustness and broader checkpoint compatibility, the official Hugging Face `transformers` implementation is still the reference path. This repo is best used when you want to understand the internals, experiment locally, and iterate quickly on the model code itself.
