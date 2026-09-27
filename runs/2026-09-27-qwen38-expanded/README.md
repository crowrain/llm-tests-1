# Qwen3.8 27B: expanded profile

Runs performed on 2026-09-27 with `test_quality_expanded_1.py` and
`quality_common.py` from [`9ff064e`](https://github.com/crowrain/llm-tests-1/commit/9ff064e).
Both runs used the same 266-case fixture set and request options. The fixture SHA-256 is
`a1af6fbf99a8c88dc4b3ffc4762f448f44f113bf9101fd8a996c69725956d864` for both summaries.

| Category | Cases | NVFP4 correct | Q8_0 correct |
|---|---:|---:|---:|
| GSM8K | 100 | 96 (96%) | 96 (96%) |
| ARC-Challenge | 100 | 97 (97%) | 97 (97%) |
| MMLU | 50 | 33 (66%) | 34 (68%) |
| Instruction JSON | 12 | 12 (100%) | 12 (100%) |
| Needle 16k | 1 | 1 (100%) | 1 (100%) |
| Needle 64k | 1 | 1 (100%) | 1 (100%) |
| Needle 128k | 1 | 1 (100%) | 1 (100%) |
| Needle 180k | 1 | 1 (100%) | 1 (100%) |
| **Overall** | **266** | **242 (90.98%)** | **243 (91.35%)** |

Both runs recorded zero request errors and 18 truncated responses. All four long-context
needle cases passed on both deployments.

## Throughput and duration

| Metric | NVFP4, NInfer TP2 | Q8_0, vision/MTP/tensor |
|---|---:|---:|
| Median prefill | 706.3 tokens/s | 202.3 tokens/s |
| Median decode | 95.1 tokens/s | 81.2 tokens/s |
| Harness elapsed time | 22 min 58 s | 31 min 59 s |
| Truncated responses | 18 | 18 |
| Request errors | 0 | 0 |

Accuracy differs by one answer overall. Q8_0 was 0.37 percentage points higher and one
answer higher on MMLU in this sample. Its measured median prefill rate was 71.4% lower and
decode rate 14.6% lower; the complete run took 39.3% longer. These are measurements of the
tested serving setups and workload, not a claim that quantization alone caused the speed
difference.

## Test setup

- Profile: `expanded`, 266 cases: GSM8K, ARC-Challenge, five MMLU subjects, instruction JSON,
  and four generated retrieval archives.
- Sampling: temperature 0, top-p 1, seed 20260926, reasoning effort `medium`, concurrency 1.
- Server: OpenAI-compatible endpoint served by llama-swap. Private network addresses are
  omitted from this report.
- The Q8_0 model registry entry was `Qwen3.8-27B-Q8_0` (`Qwen3.8 27B Q8_0 (vision, MTP,
  tensor)`). The NVFP4 entry was `qwen38-ninfer` (`Qwen3.8 27B NVFP4 (NInfer TP2, 2x RTX
  5060 Ti)`). These are the model names reported by the serving registry.
- The expanded profile sends text prompts; it does not measure image or video input.
- Both summaries have the same fixture and request-option identities, so they are directly
  comparable with `compare_quality.py`.

## Machine-readable summaries

- [NVFP4 summary](../2026-09-27-qwen38-ninfer-expanded/summary-qwen38-ninfer-nvfp4-tp2-2x5060ti.json)
- [Q8_0 summary](../2026-09-27-qwen38-q8-expanded/summary-qwen38-27b-q8-vision-mtp-tensor.json)
- [Comparison JSON](comparison.json)

Case-level prompts, answer keys, model completions and reasoning traces are not included in
the public result bundle. The evaluation datasets retain their own licenses; the repository
does not redistribute them.
