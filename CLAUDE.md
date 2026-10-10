# Team 12 — Project Context

USC Applied NLP semester project, team of 6. **Research-focused.** We are not building a new detector. We test whether existing MCP tool-poisoning detectors still work outside the conditions they were built for.

## The problem

An AI agent connected to MCP servers reads each tool's text description (from `tools/list`) to decide how to use it. **Tool poisoning** hides malicious instructions in that description (e.g., "also read ~/.ssh/id_rsa"), and the agent follows them. A **detector** inspects descriptions and blocks suspicious tools before the agent sees them.

## What we test

- **MCP-Guard** (arXiv 2508.10991): regex rules → fine-tuned E5 classifier → LLM check. Public code (github.com/GenTelLab/MCP-Guard) and checkpoint. Trained on its own benchmark (mostly jailbreak text), not on MCPTox.
- **MCP Guardian** (arXiv 2504.12757): regex/keyword filter. No public code; we rebuild it from the paper as the simple baseline.

## Data

**MCPTox** (arXiv 2508.14925, AAAI 2026): poisoned tool descriptions on 45 real MCP servers, 353 tools, ~10 risk categories, three attack types:
- **P1**: user directly asks for a tool; its description redirects the agent to a different, harmful action.
- **P2**: a fake "background" tool triggers on related actions and redirects the agent.
- **P3**: an unrelated tool sets a rule that changes another tool's parameters (e.g., redirect all emails).

Likely release: github.com/zhiqiangwang4/MCPTox-Benchmark (not yet confirmed; schema undocumented). Exact counts are still being checked.

## The three experiments

1. **Generalization:** split MCPTox risk categories into a dev set and a held-out set. Tune on dev only, then test on held-out. Report precision/recall/F1 on each side, false-positive rate on clean tools, and results per attack type and category.
2. **Adaptive attacker:** an LLM rewrites a caught poisoned description to evade the detector, capped at 5–10 tries. Report how often it evades and how many tries it takes.
3. **Live pipeline:** a proxy filters tool descriptions between a LangGraph agent and our own test MCP servers. Run the same tasks with no detector, MCP-Guard and MCP Guardian, and report attack success rate and normal-task success rate together.

## Rules

- **Never read held-out examples.** Category names and counts are fine.
- **Work only on the area the user names** (e.g., proxy, MCP-Guard, data split). If a change would affect another area or anything shared, say so instead of making it.
- **Don't invent facts about papers.** Cite the arXiv ID, and mark anything unverified.
- **Do not add claude as a contributor in any github push.**

## Out of scope

Rug pulls, server-code bugs, injection through tool outputs, tool squatting, building a new detector.
