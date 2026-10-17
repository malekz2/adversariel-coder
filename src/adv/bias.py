"""Overthinking logit penalties.

You supplied 49 marker words (" wait", " Hmm", " Alternatively", ...) and token ids for a
-2 logit bias. Token ids are tokenizer-specific (checked: your ids match the Qwen3.5/3.8
248k-vocab tokenizer 49/49, but only 1/49 on Qwen3/Qwen2.5). So that a different model can be
dropped in safely, the ids are re-derived from the WORDS with the real tokenizer at startup,
and only words that are a single token are kept (a multi-token word like " backtrack" would
otherwise penalise its first piece, " back").

The penalty is applied only while the model is inside its <think> block, because words like
" or", " error" and " could" are everywhere in real Python code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MarkerReport:
    ids: dict[int, float]          # token id -> bias
    single: dict[str, int]         # word -> id (used)
    multi: dict[str, list[int]]    # word -> ids (skipped)
    reference_matches: int         # how many of your pasted ids equal the derived ones
    reference_total: int

    def summary(self) -> str:
        return (f"{len(self.single)}/{len(self.single) + len(self.multi)} marker words are single tokens "
                f"(penalised); {len(self.multi)} multi-token words skipped; "
                f"{self.reference_matches}/{self.reference_total} of the pasted ids match this tokenizer")


def load_markers(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(tok, markers: dict, bias: float | None = None) -> MarkerReport:
    bias = markers.get("bias", -2.0) if bias is None else bias
    words = markers["words"]
    ref = markers.get("reference_ids") or []
    single, multi = {}, {}
    for w in words:
        ids = tok.encode(w)
        if len(ids) == 1:
            single[w] = ids[0]
        else:
            multi[w] = ids
    # the pasted ids are sorted, not in word order, so compare as sets
    matches = len(set(single.values()) & set(ref))
    return MarkerReport(ids={i: float(bias) for i in single.values()}, single=single, multi=multi,
                        reference_matches=matches, reference_total=len(ref))


if __name__ == "__main__":  # python -m adv.bias <folder with tokenizer.json>
    import sys
    from transformers import AutoTokenizer

    class _Tok:
        def __init__(self, p):
            self.t = AutoTokenizer.from_pretrained(p)

        def encode(self, w):
            return self.t.encode(w, add_special_tokens=False)

    root = Path(__file__).resolve().parents[2]
    rep = resolve(_Tok(sys.argv[1]), load_markers(root / "data" / "overthinking_markers.json"))
    for w, i in rep.single.items():
        print(f"{i:>7}  {w!r}")
    for w, ids in rep.multi.items():
        print(f"skipped {w!r} -> {ids}")
    print(rep.summary())
