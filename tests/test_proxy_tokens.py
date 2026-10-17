import json

import torch

from adv.engines.base import GenParams, GenResult
from adv.engines.hf import ThinkingOnlyBias
from adv.proxy import RecordingProxy
from adv.tokenizer import MockTokenizer, normalize_messages, parse_tool_calls, split_completion, thinking_open

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
    "command": {"type": "string"}, "timeout": {"type": "number"}}}}}]


def test_parse_xml_tool_call_with_types():
    text = ("Let me run it.\n<tool_call>\n<function=bash>\n<parameter=command>\npython run_tests.py\n</parameter>\n"
            "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "Let me run it."
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "python run_tests.py", "timeout": 30}


def test_parse_k2_horizon_tool_call():
    text = ("Running the tests.<ifm|tool_calls>\n<ifm|tool_call>bash\n<ifm|arg_key>command</ifm|arg_key>\n"
            "<ifm|arg_value>python run_tests.py</ifm|arg_value>\n<ifm|arg_key>timeout</ifm|arg_key>\n"
            "<ifm|arg_value>30</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>")
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "Running the tests."
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "python run_tests.py", "timeout": 30}


def test_parse_json_tool_call_fallback():
    _, calls = parse_tool_calls('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>', TOOLS)
    assert calls[0]["function"]["name"] == "bash"


def test_normalize_messages():
    msgs = normalize_messages([{"role": "developer", "content": [{"type": "text", "text": "hi"}]},
                               {"role": "assistant", "content": None, "tool_calls": [
                                   {"id": "1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}])
    assert msgs[0] == {"role": "system", "content": "hi"}
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


class ScriptedEngine:
    """Returns pre-written completions, one per call."""
    max_parallel = 1

    def __init__(self, tok, texts):
        self.tok, self.texts = tok, list(texts)

    def generate(self, prompt_ids, params, think_end):
        ids = self.tok.encode(self.texts.pop(0)) + [self.tok.im_end]
        return GenResult(ids, [1] * len(ids), "stop", 3)


def test_proxy_chains_turns_into_one_sequence_with_tool_output_masked():
    tok = MockTokenizer()
    eng = ScriptedEngine(tok, [
        "abc</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
        "def</think>\n\nAll done."])
    px = RecordingProxy(tok, eng, {"solver": GenParams("solver")}, max_ctx=100000)
    rid = px.begin("solver", max_turns=4)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    r1 = px.chat(rid, msgs, TOOLS)
    assert r1["finish_reason"] == "tool_calls" and r1["message"]["reasoning_content"] == "abc"
    msgs += [{"role": "assistant", "content": r1["message"]["content"], "reasoning_content": "abc",
              "tool_calls": r1["message"]["tool_calls"]},
             {"role": "tool", "tool_call_id": "x", "content": "FILE_LISTING_OUTPUT"}]
    r2 = px.chat(rid, msgs, TOOLS)
    assert r2["message"]["content"] == "All done."
    ro = px.end(rid)
    assert [t.chained for t in ro.turns] == [False, True]
    (ids, mask), = ro.sequences()
    trained = tok.decode([i for i, m in zip(ids, mask) if m])
    untrained = tok.decode([i for i, m in zip(ids, mask) if not m])
    assert "abc" in trained and "All done." in trained and "FILE_LISTING_OUTPUT" not in trained
    assert "FILE_LISTING_OUTPUT" in untrained and "task" in untrained


def test_proxy_turn_limit():
    tok = MockTokenizer()
    px = RecordingProxy(tok, ScriptedEngine(tok, ["x</think>\n\nhi"]), {"solver": GenParams("solver")})
    rid = px.begin("solver", max_turns=1)
    px.chat(rid, [{"role": "user", "content": "q"}])
    r = px.chat(rid, [{"role": "user", "content": "q"}])
    assert "Turn limit" in r["message"]["content"] and px.end(rid).truncated


def test_thinking_helpers():
    tok = MockTokenizer()
    ids = tok.encode(tok.render_text([{"role": "user", "content": "q"}], None, True))
    assert thinking_open(tok, ids) == tok.think_end
    gen = tok.encode("reason</think>\n\nanswer") + [tok.im_end]
    assert split_completion(tok, gen, tok.think_end) == ("reason", "answer")


def test_thinking_only_bias_processor():
    vocab, think_end, marker = 10, 9, 3
    proc = ThinkingOnlyBias(prompt_len=2, think_end=think_end, bias={marker: -2.0}, budget=3, active=True, device="cpu")
    scores = torch.zeros(1, vocab)
    out = proc(torch.tensor([[1, 2, 5]]), scores)            # inside <think>: biased
    assert out[0, marker] == -2.0 and out[0, 0] == 0.0
    out = proc(torch.tensor([[1, 2, 5, 6, 7]]), torch.zeros(1, vocab))   # budget hit: force </think>
    assert out[0, think_end] == 0.0 and torch.isinf(out[0, 0]) and proc.forced_at == 3
    out = proc(torch.tensor([[1, 2, 5, 6, 7, 9]]), torch.zeros(1, vocab))  # after thinking: untouched
    assert out[0, marker] == 0.0
