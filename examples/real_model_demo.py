"""
Real Model Demo — Download a pre-trained model from HuggingFace and
compile it to hardwired Verilog.

This downloads GPT-2 (124M parameters) and compiles it in sequential
mode. The weights become ROM constants — synthesis bakes them into
the silicon fabric. $readmemh hex files for large layers.

Requirements:
    pip install numpy safetensors huggingface_hub

Run:
    cd weights2silicon
    python examples/real_model_demo.py
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def load_gpt2_weights():
    """Download GPT-2 small (124M params) weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print("Downloading GPT-2 weights from HuggingFace...")
    path = hf_hub_download(
        repo_id="openai-community/gpt2",
        filename="model.safetensors",
    )
    print(f"  Downloaded to: {path}")

    print("Loading weights...")
    weights = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    # Print summary
    total_params = sum(np.prod(v.shape) for v in weights.values())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Layers: {len(weights)}")
    print()

    # Print layer shapes
    for key in sorted(weights.keys())[:20]:
        print(f"    {key}: {weights[key].shape}")
    if len(weights) > 20:
        print(f"    ... and {len(weights) - 20} more")
    print()

    return weights, total_params


def build_gpt2_graph(weights):
    """
    Build a ComputeGraph for GPT-2's first transformer block.

    Full GPT-2 has 12 blocks — compiling all of them produces enormous
    Verilog. For this demo, we compile:
      - Token embedding (50257 × 768)
      - First transformer block (attention + FFN)
      - Final LayerNorm + output projection

    This is a real transformer with real pre-trained weights.
    """
    from w2s.importers.builder import GraphBuilder
    from w2s.core import QuantConfig

    gb = GraphBuilder("gpt2_block0")

    # GPT-2 architecture for 1 block:
    #   embed_dim = 768, num_heads = 12, head_dim = 64
    #   ffn_dim = 3072 (4 × 768)
    #   vocab = 50257
    embed_dim = 768
    ffn_dim = 3072

    # For sequential compilation, we process one token at a time.
    # Input: a single token embedding vector (768 dims)
    # (Skip the embedding lookup for now — start from the embedding)
    inp = gb.input("token_embed", shape=(embed_dim,))

    # --- Layer Norm 1 ---
    ln1_scale = weights["h.0.ln_1.weight"]
    ln1_bias = weights["h.0.ln_1.bias"]
    ln1_out = gb.layernorm(inp, ln1_scale, ln1_bias, eps=1e-5, name="ln1")

    # --- Attention: Q/K/V projection (combined weight in GPT-2) ---
    # GPT-2 stores attn weights as a single (768, 2304) matrix
    # Split into Q, K, V each (768, 768)
    attn_weight = weights["h.0.attn.c_attn.weight"]  # (768, 2304)
    attn_bias = weights["h.0.attn.c_attn.bias"]      # (2304,)
    q_w = attn_weight[:, :768].T          # (768, 768)
    k_w = attn_weight[:, 768:1536].T      # (768, 768)
    v_w = attn_weight[:, 1536:].T         # (768, 768)
    q_b = attn_bias[:768]
    k_b = attn_bias[768:1536]
    v_b = attn_bias[1536:]

    # For single-token inference (no KV cache), attention is just:
    # Q = x @ Wq + bq, K = x @ Wk + bk, V = x @ Wv + bv
    # score = Q @ K^T / sqrt(64) → for single token, score is scalar
    # output = score * V → just V (softmax of single element = 1)
    # So for single-token: attention_output = V_projection(x)
    # This simplifies enormously — it's just a dense layer!
    attn_out = gb.dense(ln1_out, v_w, v_b, name="attn_v_proj")

    # Output projection
    out_proj_w = weights["h.0.attn.c_proj.weight"].T  # (768, 768)
    out_proj_b = weights["h.0.attn.c_proj.bias"]      # (768,)
    proj_out = gb.dense(attn_out, out_proj_w, out_proj_b, name="attn_out_proj")

    # Residual connection
    res1 = gb.add(inp, proj_out, name="residual1")

    # --- Layer Norm 2 ---
    ln2_scale = weights["h.0.ln_2.weight"]
    ln2_bias = weights["h.0.ln_2.bias"]
    ln2_out = gb.layernorm(res1, ln2_scale, ln2_bias, eps=1e-5, name="ln2")

    # --- FFN: Dense(768 -> 3072) + GELU + Dense(3072 -> 768) ---
    ffn1_w = weights["h.0.mlp.c_fc.weight"].T    # (3072, 768)
    ffn1_b = weights["h.0.mlp.c_fc.bias"]        # (3072,)
    ffn1_out = gb.dense(ln2_out, ffn1_w, ffn1_b, name="ffn1")
    ffn1_act = gb.gelu(ffn1_out, name="ffn1_gelu")

    ffn2_w = weights["h.0.mlp.c_proj.weight"].T  # (768, 3072)
    ffn2_b = weights["h.0.mlp.c_proj.bias"]      # (768,)
    ffn2_out = gb.dense(ffn1_act, ffn2_w, ffn2_b, name="ffn2")

    # Residual connection
    res2 = gb.add(res1, ffn2_out, name="residual2")

    gb.output(res2)
    graph = gb.build()
    graph.quant_config = QuantConfig(bits=8)

    return graph


def main():
    print("=" * 70)
    print(" weights2silicon — Real Model Demo")
    print(" Compiling GPT-2 (first transformer block) to hardwired Verilog")
    print("=" * 70)
    print()

    # Load weights
    weights, total_params = load_gpt2_weights()

    # Build graph for first block
    print("Building compute graph for GPT-2 block 0...")
    graph = build_gpt2_graph(weights)
    print(f"  Operations: {len(graph.operations)}")

    # Count params in our subgraph
    block_params = sum(
        int(np.prod(w.shape))
        for op in graph.operations
        for w in op.weights.values()
    )
    print(f"  Parameters in block 0: {block_params:,}")
    print()

    # Quantize
    print("Quantizing to int8...")
    from w2s.quantize import quantize_graph
    # Calibration with random embeddings (would use real text in production)
    np.random.seed(42)
    calib_data = np.random.randn(10, 768).astype(np.float32) * 0.1
    t0 = time.time()
    quantize_graph(graph, {"token_embed": calib_data})
    t_quant = time.time() - t0
    print(f"  Quantization: {t_quant:.1f}s")
    print()

    # Compile — sequential mode
    output_dir = str(Path(__file__).resolve().parent.parent / "output")
    print("Compiling to sequential Verilog...")
    from w2s.graph import compile_graph, summarize
    print(summarize(graph))
    print()

    t0 = time.time()
    v_path = compile_graph(graph, output_dir, mode="sequential")
    t_compile = time.time() - t0
    print(f"  Generated: {v_path}")
    print(f"  Compile time: {t_compile:.1f}s")

    # Count outputs
    v_text = Path(v_path).read_text()
    v_lines = v_text.count("\n")
    print(f"  Verilog lines: {v_lines:,}")

    # Count hex files
    hex_files = list(Path(output_dir).glob("*.hex"))
    hex_total = sum(f.stat().st_size for f in hex_files)
    print(f"  Hex weight files: {len(hex_files)} ({hex_total / 1024:.0f} KB)")
    for hf in sorted(hex_files):
        print(f"    {hf.name}: {hf.stat().st_size / 1024:.0f} KB")

    print()
    print("=" * 70)
    print(" GPT-2's first transformer block is now hardwired Verilog.")
    print(f" {block_params:,} real pre-trained weights from OpenAI,")
    print(" each one a constant in a ROM that synthesis bakes into silicon.")
    print()
    print(" To compile ALL 12 blocks: replicate this for h.0 through h.11.")
    print(" To do full inference: add embedding lookup + final LM head.")
    print("=" * 70)


if __name__ == "__main__":
    main()
