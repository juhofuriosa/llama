# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import core_aten_decompositions

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def core_aten_backend(gm, inputs):
    print("Compile using core_aten_backend")
    gm = make_fx(
        gm,
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
        decomposition_table=core_aten_decompositions()
    )(*inputs)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    #gm.print_readable()
    return gm


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    temperature: float,
    top_p: float,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    if len(checkpoints) > 0:
        ckpt_path = checkpoints[0]
        print(f"Use checkpoint file {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    else:
        print("Skip loading checkpoint")
        checkpoint = None
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.FloatTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    if checkpoint:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model = torch.compile(model, backend=core_aten_backend, fullgraph=False)

    generator = LLaMA(model, tokenizer, temperature=temperature, top_p=top_p)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str = '.',
    tokenizer_path: str = './tokenizer.model',
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
):
    torch.manual_seed(1)
    generator = load(
        ckpt_dir,
        tokenizer_path,
        max_seq_len,
        max_batch_size,
        temperature=temperature,
        top_p=top_p
    )

    # First run
    prompts = [
        "I believe the meaning of life",
    ]
    results = generator.generate(prompts, max_gen_len=4)

    for result in results:
        print(result)
        print("\n==================================\n")

    # Second run
    # Total_len should be the same with first run to use the same graph
    prompts = [
        "I believe the meaning of life is",
    ]
    results = generator.generate(prompts, max_gen_len=3)

    for result in results:
        print(result)
        print("\n==================================\n")

    # Third run
    # Total_len should be the same with second run to use the same graph
    prompts = [
        "A B C D E F G",
    ]
    results = generator.generate(prompts, max_gen_len=3)

    for result in results:
        print(result)
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
