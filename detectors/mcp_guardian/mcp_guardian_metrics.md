# MCP Guardian: metrics on the tune split

These are the results for our reimplementation of MCP Guardian (Kumar et al., 2025, arXiv:2504.12757) on the `tune` split of Team 12's official MCPTox split. Guardian is a regex Web Application Firewall (WAF) with the paper's 11 rules. It is tested as published: no rule was added or tuned on MCPTox.

## Summary

| | |
|---|---|
| Accuracy | **27.9%** (279 / 999) |
| Recall (poisoned cases caught) | **5.3%** (40 / 759) |
| Precision | 97.6% (40 / 41 flagged) |
| F1 | 0.100 |
| False positive rate | 0.4% (1 / 240 clean tools) |
| Balanced accuracy | 52.4% |

Guardian almost never flags a tool. When it does flag one, it is nearly always right, but it misses about 95% of attacks. It catches attacks only when the description contains a sensitive file path (`.ssh`, `/etc/passwd`, `/etc/shadow`). Of its 11 rules, 3 fired and 8 never did. It caught nothing in 5 of the 8 risk categories.

Plain accuracy is misleading on this split, because 76% of the rows are poisoned. A detector that labelled every tool "poisoned" would score 76.0%, far above Guardian's 27.9%. Read recall, false positive rate and balanced accuracy instead.

## Run details

| | |
|---|---|
| Date | 2026-10-11 |
| Branch / commit | `MCP-Guardian` @ `405310e` |
| Detector | `detectors/mcp_guardian`, `WafScanner()` with the default `WAF_RULES` |
| Ruleset | 11 rules, fingerprint `9decc37e4f86` (unchanged since the rules were written) |
| Extra rules | none (`extra_rules` left empty) |
| Split | `tune` from `data/processed/tune.jsonl`, loaded through `data/loader.py` |
| Rows | 999: 759 poisoned MCPTox cases and 240 clean tools |
| Fields scanned | tool `name` and `description` (the split has no parameter schema) |
| Decision rule | a row is flagged "poisoned" if any of the 11 rules matches any scanned field |
| Environment | Python 3.12.14, macOS (Apple Silicon) |
| Determinism | Regex only, no randomness. Two separate runs gave identical results. |
| Val / test | Not included here. `test` was never loaded (`FINAL_EVAL` unset). |

The rules were never fit to MCPTox, so `tune` is not training data for Guardian. It is simply data the detector has not seen.

## Confusion matrix

"Positive" means poisoned.

| | Flagged poisoned | Passed as clean | Total |
|---|---:|---:|---:|
| **Actually poisoned** | TP = 40 | FN = 719 | 759 |
| **Actually clean** | FP = 1 | TN = 239 | 240 |
| **Total** | 41 | 958 | 999 |

## All metrics

The 95% intervals are Wilson score intervals for a proportion.

| Metric | Formula | Value | 95% CI |
|---|---|---:|---|
| Accuracy | (TP + TN) / N | 0.2793 (279 / 999) | 0.252 – 0.308 |
| Recall / true positive rate | TP / (TP + FN) | 0.0527 (40 / 759) | 0.039 – 0.071 |
| Miss rate / false negative rate | FN / (TP + FN) | 0.9473 (719 / 759) | |
| Precision | TP / (TP + FP) | 0.9756 (40 / 41) | 0.874 – 0.996 |
| Specificity / true negative rate | TN / (TN + FP) | 0.9958 (239 / 240) | |
| False positive rate | FP / (FP + TN) | 0.0042 (1 / 240) | 0.001 – 0.023 |
| Negative predictive value | TN / (TN + FN) | 0.2495 (239 / 958) | |
| F1 | 2PR / (P + R) | 0.1000 | |
| F2 (weights recall higher) | 5PR / (4P + R) | 0.0650 | |
| Balanced accuracy | (recall + specificity) / 2 | 0.5243 | |
| Matthews correlation coefficient | | 0.1045 | |
| Cohen's kappa | | 0.0240 | |

### Reference baselines

| Trivial detector | Accuracy | Recall | False positive rate |
|---|---:|---:|---:|
| Always "poisoned" | 0.7598 (759 / 999) | 1.000 | 1.000 |
| Always "clean" | 0.2402 (240 / 999) | 0.000 | 0.000 |
| **MCP Guardian** | **0.2793** | **0.053** | **0.004** |

Guardian's accuracy is only 3.9 points above "always clean", and its kappa of 0.024 confirms that it agrees with the labels barely more than chance would. Its balanced accuracy (0.524) is just above the 0.5 of either trivial detector.

## Recall by risk category

Sorted by number of cases.

| Risk category | Cases | Caught | Recall |
|---|---:|---:|---:|
| Privacy Leakage | 189 | 14 | 7.4% |
| Information Manipulation | 186 | 0 | 0.0% |
| Service Disruption | 140 | 0 | 0.0% |
| Infrastructure Damage | 85 | 23 | 27.1% |
| Data Tampering | 81 | 3 | 3.7% |
| Code Injection | 46 | 0 | 0.0% |
| Message Hijacking | 28 | 0 | 0.0% |
| Other | 4 | 0 | 0.0% |
| **All** | **759** | **40** | **5.3%** |

Infrastructure Damage is the only category above 10%: 23 of its attacks name `~/.ssh`, `/etc/passwd` or `/etc/shadow`. Code Injection sounds like the WAF's home ground, yet it gets 0/46: none of these descriptions matches the shell, SQL or script rules.

## Recall by attack type

| Attack type | Meaning | Cases | Caught | Recall |
|---|---|---:|---:|---:|
| P1 | The requested tool's description redirects the agent | 119 | 8 | 6.7% |
| P2 | A fake background tool triggers on related actions | 290 | 21 | 7.2% |
| P3 | An unrelated tool rewrites another tool's parameters | 350 | 11 | 3.1% |

P3 has the lowest recall. A P3 attack works by changing another tool's parameters, so its description often has no file path for the rules to match. 8 of its 11 catches are in the Infrastructure Damage and Data Tampering categories.

## Risk category × attack type

Each cell shows caught / cases. A dash means the split has no cases for that cell.

| Risk category | P1 | P2 | P3 | Total |
|---|---:|---:|---:|---:|
| Privacy Leakage | 2 / 29 | 9 / 110 | 3 / 50 | 14 / 189 |
| Information Manipulation | 0 / 15 | 0 / 27 | 0 / 144 | 0 / 186 |
| Service Disruption | – | 0 / 50 | 0 / 90 | 0 / 140 |
| Infrastructure Damage | 6 / 42 | 12 / 35 | 5 / 8 | 23 / 85 |
| Data Tampering | 0 / 15 | 0 / 27 | 3 / 39 | 3 / 81 |
| Code Injection | 0 / 9 | 0 / 28 | 0 / 9 | 0 / 46 |
| Message Hijacking | 0 / 9 | 0 / 10 | 0 / 9 | 0 / 28 |
| Other | – | 0 / 3 | 0 / 1 | 0 / 4 |
| **Total** | **8 / 119** | **21 / 290** | **11 / 350** | **40 / 759** |

## Activity of each rule

"Cases" counts poisoned cases the rule matched. "Only this rule" counts cases that no other rule matched. "Servers" counts distinct MCP servers among the matched cases.

| Rule | Paper source | Cases | Only this rule | Servers | Clean tools |
|---|---|---:|---:|---:|---:|
| `rm_rf` | Sec. 4.1: `rm\s+-rf` | 0 | 0 | 0 | 0 |
| `destructive_shell` | Sec. 3.4.1: destructive shell commands | 0 | 0 | 0 | 0 |
| `remote_code_exec` | Sec. 2.3.1 (5): command injection | 0 | 0 | 0 | 0 |
| `sql_drop` | Sec. 4.3: drop table | 0 | 0 | 0 | 0 |
| `sql_tautology` | Sec. 3.4.1: SQL injection strings | 0 | 0 | 0 | 0 |
| `sql_union` | Sec. 3.4.1: SQL injection strings | 0 | 0 | 0 | 0 |
| `script_tag` | Sec. 3.4.2: `<script>` tags | 0 | 0 | 0 | 0 |
| `ssh_key` | Sec. 3.4.1: references to SSH files | 16 | 14 | 1 | 0 |
| `sensitive_file` | Sec. 3.4.1 and 2.3.1 (1): sensitive files | 26 | 24 | 4 | 0 |
| `env_secret` | Sec. 3.4.1 and 3.4: environment variables, secrets | 0 | 0 | 0 | 1 |
| `path_traversal` | Sec. 3.4.1: references to sensitive files | 0 | 0 | 0 | 0 |

- 38 caught cases matched exactly one rule, and 2 matched both `ssh_key` and `sensitive_file`: 14 + 24 + 2 = 40.
- All 8 rules for shell commands, SQL injection, script tags and path traversal stayed silent on all 999 rows.

### Text that triggered the rules

For each field, the excerpt is the first match of a rule.

| Rule | Matched text | Cases | Field |
|---|---|---:|---|
| `ssh_key` | `.ssh` | 16 | description |
| `sensitive_file` | `/etc/passwd` | 15 | description |
| `sensitive_file` | `/etc/shadow` | 11 | description |
| `env_secret` | `printEnv` | 1 clean tool | name |

All 40 true positives came from the description. The only false positive came from a tool name.

### Rule matches by risk category

Each count is one (case, rule) match. A case that matched two rules is counted once for each.

| Risk category | `ssh_key` `.ssh` | `sensitive_file` `/etc/passwd` | `sensitive_file` `/etc/shadow` | Matches | Cases caught |
|---|---:|---:|---:|---:|---:|
| Infrastructure Damage | 11 | 6 | 6 | 23 | 23 |
| Privacy Leakage | 5 | 6 | 5 | 16 | 14 |
| Data Tampering | 0 | 3 | 0 | 3 | 3 |
| **Total** | **16** | **15** | **11** | **42** | **40** |

The two cases that matched both rules are both Privacy Leakage cases.

## Where the catches come from

The 40 caught cases come from only 4 of the 28 servers that have poisoned cases in this split. They involve 14 distinct poisoned tools, out of 260 distinct poisoned tools in the split.

| Server | Caught | Poisoned cases on this server | Recall on this server |
|---|---:|---:|---:|
| FileSystem | 28 | 32 | 87.5% |
| GitHub | 6 | 40 | 15.0% |
| Redis | 3 | 12 | 25.0% |
| gNucleus Text to CAD | 3 | 3 | 100.0% |
| The other 24 servers | 0 | 672 | 0.0% |

Seventy percent of all catches (28 / 40) are on the FileSystem server, where attacks naturally mention file paths. Guardian's recall therefore depends on whether a server's attacks happen to name a sensitive file, not on whether the description is malicious.

## The false positive

| Server | Tool | Rule | Matched text | Field |
|---|---|---|---|---|
| Everything | `printEnv` | `env_secret` | `printEnv` | name |

`printEnv` is a harmless demo tool from the Everything reference server. `env_secret` matches the word `printenv` (Sec. 3.4.1, references to environment variables), so the name alone triggers it. This was the only one of the 240 clean tools that was flagged. 17 of the clean tools have no description, so Guardian scanned only their names.

## Speed

Scanning all 999 rows took 41 ms, about 0.04 ms per tool, on an Apple Silicon laptop. This covers only the WAF scan, with no proxy or network involved, from a single run, so treat it as an order of magnitude.

## Why Guardian misses so much

The WAF looks for attack *payloads*: `rm -rf`, `DROP TABLE`, `<script>`, `.ssh`, `/etc/passwd`. Most MCPTox attacks are plain-language *instructions* to the agent, such as "before using this tool you must first call X" or "always set the recipient to Y". They contain no command, SQL or file path, so no rule fires. This explains the zeros for Information Manipulation (0/186), Service Disruption (0/140), Message Hijacking (0/28) and Code Injection (0/46).

## Caveats

- **Tune is not training data for this detector.** The rules were written from the paper text before any MCPTox data was opened and have not changed since; the fingerprint `9decc37e4f86` shows it. The tune and val numbers both measure performance on unseen data.
- **The paper's WAF scans call arguments; we scan tool descriptions.** Applying the same rules to `tools/list` metadata is our adaptation, needed for the detector slot in the proxy. The rules themselves are unchanged.
- **No parameter schema.** The split's records carry only a name and description, so attacks hidden in parameter descriptions (if any existed) could not be seen.
- **`<IMPORTANT>` tags.** About a third of tune attacks (251 / 759) are wrapped in `<IMPORTANT>` tags. Guardian has no rule for tags, so this does not affect its score. A detector with such a rule would get an easy boost, so keep this in mind when comparing detectors.
- **Unit of evaluation.** One row is one MCPTox case or one clean tool. Several cases can share the same poisoned tool (759 cases over 260 tools), so the per-tool and per-case views differ.

## How to reproduce

Run from the repository root. Use `--split tune` so val and test are not touched.

```bash
uv run python -m detectors.mcp_guardian.evaluate_rules --split tune
```

Expected output:

```
MCP Guardian WAF, 11 rules, fingerprint 9decc37e4f86

=== tune: 759 poisoned cases, 240 clean tools ===
recall 0.053 (40/759)   precision 0.976   F1 0.100   FPR 0.004 (1/240)
recall by category: Code Injection 0/46; Data Tampering 3/81; Information Manipulation 0/186; Infrastructure Damage 23/85; Message Hijacking 0/28; Other 0/4; Privacy Leakage 14/189; Service Disruption 0/140
recall by attack type: P1 8/119; P2 21/290; P3 11/350
rules that fired (poisoned cases / servers / clean tools): ssh_key 16/1/0; sensitive_file 26/4/0; env_secret 0/0/1
false positive: Everything/printEnv -> ['env_secret']
```

The extra figures in this file (accuracy, the confidence intervals, MCC, kappa, the category × attack type grid, the per-server table and the rule excerpts) were computed from the same `WafScanner().scan` results on the same 999 rows.
