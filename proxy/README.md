# proxy/ — MCP proxy that filters tool descriptions between the agent and MCP servers (Experiment 3).

The proxy is an MCP server to the agent and an MCP client to one upstream server (official `mcp` SDK, 2.x, stdio on both sides). It relays everything unchanged except:

- **`tools/list`**: the full upstream list goes through the configured detector; the agent only sees the tools it keeps.
- **`tools/call`**: calls to a blocked tool are refused with the same `Unknown tool` error a nonexistent tool gets.

```
agent ──stdio──▶ python -m proxy --config X.yaml ──stdio──▶ upstream server
```

## Run

```
uv run python -m proxy --config configs/proxy/example.yaml [--run-id RUN]
```

The agent launches this command as its MCP server. Run **one proxy per upstream server**. The three Experiment 3 conditions use configs that differ only in the `detector` block.

## Plugging in a detector

Any class with a `name` and `filter_tools`:

```python
from mcp.types import Tool

class MyDetector:
    name = "my-detector"
    def __init__(self, threshold: float = 0.5): ...
    def filter_tools(self, tools: list[Tool]) -> list[Tool]:   # may also be `async def`
        return [t for t in tools if not self.is_poisoned(t)]
```

```yaml
detector:
  class: detectors.my_module:MyDetector
  params: {threshold: 0.7}
  timeout_s: 30          # null = none; a timeout counts as a detector error
  on_error: raise        # raise | fail_open | fail_closed
```

Rules the proxy enforces:
- Return an **unchanged subset** of the input. Rewriting a description raises a contract error.
- The detector gets the whole `Tool` (name, description, `input_schema`, annotations). Poison can sit in parameter descriptions, so each wrapper decides what text to score.
- Sync `filter_tools` runs in a worker thread, so a slow model doesn't block the proxy.
- Decisions are cached per whole tool list within a run; an identical relist doesn't call the detector again.

## Logs

Under `log_dir`, per run and upstream:

- `<run_id>.<upstream>.decisions.jsonl`: one JSON object per event, each with `ts` (UTC), `run_id`, `event`, `detector`, `upstream`.
  - `proxy_start`: detector class/params, `on_error`, `timeout_s`, upstream command.
  - `tool_decision`: one per tool per `tools/list`: `tool`, `decision` (`allowed`/`blocked`), `description_sha256`, `tool_sha256`, `list_sha256`, `latency_ms` (whole detector call), `cached`, `fallback` (set if `on_error` decided).
  - `call_forwarded` / `call_blocked`: `tool`.
  - `detector_error`: `error`, `on_error`, `list_sha256`.
- `<run_id>.<upstream>.tools.jsonl`: the full upstream tool list, once per distinct `list_sha256`.

**A run with any `detector_error` under `on_error: raise` should be discarded.** The agent got an error instead of tools.

## Not relayed

These are out of scope for our test servers: server-to-client requests (sampling, elicitation), change notifications, resource subscriptions, and the deprecated logging capability. `instructions` and resources/prompts are relayed unchanged and not inspected.

## Tests

`tests/test_proxy.py` uses a scripted client as the agent, plus the toy upstream and stub detectors in `tests/fixtures/` (synthetic text only). It also includes one end-to-end test that launches the proxy over stdio.
