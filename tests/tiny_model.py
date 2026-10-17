"""Build a miniature, randomly initialised Qwen3.5 checkpoint with the REAL tokenizer.

Same architecture (Gated-DeltaNet + gated attention, multimodal wrapper, 248k vocab) as the
real models, but a few MB, so the plumbing (loading, LoRA targets, log-probs, generation,
GGUF conversion) can be tested on any laptop. It knows nothing; never train it for real.

    python tests/tiny_model.py <tokenizer_dir_with_config.json> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path


def make_tiny(src: str, out: str) -> str:
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    src, out = Path(src), Path(out)
    cfg = json.loads((src / "config.json").read_text())
    t = cfg["text_config"]
    t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
             layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=2,
             num_key_value_heads=1, head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
             linear_num_key_heads=2, linear_num_value_heads=2, full_attention_interval=4)
    cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg))
    config = AutoConfig.from_pretrained(out)
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    model.save_pretrained(out)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    return str(out)


if __name__ == "__main__":
    print(make_tiny(sys.argv[1], sys.argv[2]))
