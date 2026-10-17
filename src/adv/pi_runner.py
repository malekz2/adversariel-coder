"""Run the Solver inside the Pi coding agent, one process per attempt.

Pi gets a fresh temp workspace (PROBLEM.md, solution.py stub, run_tests.py) and talks to our
recording proxy as an OpenAI-compatible provider called "adv". The rollout id travels as the
API key, so the proxy knows which attempt each request belongs to. After Pi exits we grade
solution.py against the HIDDEN tests, which never exist on disk where Pi can see them.

Security note: Pi's bash tool executes whatever the model writes, as you, on the lab PC.
The workspace is a temp folder, but bash is not confined to it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = os.name == "nt"


@dataclass
class PiResult:
    exit_code: int | None
    timed_out: bool
    seconds: float
    tool_calls: int = 0
    tool_errors: int = 0
    tools_used: dict = field(default_factory=dict)
    final_text: str = ""
    stderr: str = ""


class PiRunner:
    def __init__(self, node: str, cli_js: str, agent_dir: Path, proxy_base_url: str, context_window: int,
                 max_tokens: int, shell_path: str | None = None, extra_path: list[str] | None = None,
                 tools: str = "read,write,edit,bash", timeout_s: float = 900):
        self.node, self.cli_js = node, cli_js
        self.agent_dir = Path(agent_dir)
        self.tools, self.timeout_s = tools, timeout_s
        self.extra_path = extra_path or []
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        models = {"providers": {"adv": {
            "baseUrl": proxy_base_url, "api": "openai-completions", "apiKey": "$ADV_ROLLOUT_KEY",
            "models": [{"id": "solver", "name": "adversarial solver", "reasoning": True, "input": ["text"],
                        "contextWindow": context_window, "maxTokens": max_tokens}]}}}
        (self.agent_dir / "models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
        settings = {"quietStartup": True, "httpIdleTimeoutMs": "disabled", "retry": {"enabled": False},
                    "compaction": {"enabled": False}, "enableInstallTelemetry": False,
                    "enableAnalytics": False, "defaultProjectTrust": "never",
                    "defaultProvider": "adv", "defaultModel": "solver"}
        if shell_path:
            settings["shellPath"] = shell_path
        (self.agent_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def run(self, workspace: Path, prompt: str, rollout_id: str) -> PiResult:
        env = dict(os.environ)
        env.update({"PI_CODING_AGENT_DIR": str(self.agent_dir), "ADV_ROLLOUT_KEY": rollout_id,
                    "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0",
                    "PYTHONDONTWRITEBYTECODE": "1"})
        if self.extra_path:  # so that `python run_tests.py` inside Pi's bash finds our Python
            env["PATH"] = os.pathsep.join(self.extra_path + [env.get("PATH", "")])
        cmd = [self.node, self.cli_js, "--mode", "json", "-p", "--no-session", "--provider", "adv",
               "--model", "solver", "--tools", self.tools, "--no-extensions", "--no-skills",
               "--no-prompt-templates", "--no-context-files", "--no-mcp", "--no-themes", "--offline",
               "--no-approve", "--thinking", "medium", "--", prompt]
        kwargs = dict(cwd=str(workspace), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE)
        if IS_WINDOWS:
            kwargs["creationflags"] = 0x08000000 | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, **kwargs)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(proc)
            out, err = proc.communicate()
        res = PiResult(proc.returncode, timed_out, time.monotonic() - t0,
                       stderr=err.decode("utf-8", "replace")[-3000:])
        for line in out.decode("utf-8", "replace").split("\n"):  # strict LF-delimited JSONL
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("type") if isinstance(ev, dict) else None
            if kind == "tool_execution_end":
                res.tool_calls += 1
                res.tool_errors += int(bool(ev.get("isError")))
                name = ev.get("toolName", "?")
                res.tools_used[name] = res.tools_used.get(name, 0) + 1
            elif kind == "message_end" and (ev.get("message") or {}).get("role") == "assistant":
                parts = ev["message"].get("content") or []
                texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
                if texts:
                    res.final_text = "\n".join(texts)[-2000:]
        return res

    @staticmethod
    def _kill(proc):
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(proc.pid, 9)
        except Exception:
            proc.kill()
