# Qwen3.8 27B — expanded benchmark

266 identical text cases, `temperature=0`, seed 20260926, concurrency 1. Both runs used
the same fixtures and completed with 0 request errors and 18 truncated responses.

| Model and quant | Inference and hardware | Accuracy | Prefill | Decode | Time |
|---|---|---:|---:|---:|---:|
| Qwen3.8 27B **NVFP4** | NInfer TP2, MTP; **2× RTX 5060 Ti 16 GB** | 242/266 (**90.98%**) | **706.3 tok/s** | **95.1 tok/s** | **22:58** |
| Qwen3.8 27B **Q8_0 GGUF** | llama.cpp v0.4.0, MTP, tensor split; **RTX 4080 SUPER 16 GB + 2× RTX 5060 Ti 16 GB** | 243/266 (**91.35%**) | 202.3 tok/s | 81.2 tok/s | 31:59 |

Shared host: AMD Ryzen 9 5900X, NVIDIA driver 595.91.07. The Q8_0 run used Q8_0 KV cache;
the NVFP4 run used INT8 KV cache. This compares the complete serving configurations, not
quantization in isolation.

| Category | NVFP4 | Q8_0 |
|---|---:|---:|
| GSM8K | 96% | 96% |
| ARC-Challenge | 97% | 97% |
| MMLU | 66% | 68% |
| Instruction JSON | 100% | 100% |
| Long-context needles: 16k / 64k / 128k / 180k | 4/4 | 4/4 |

Artifacts: [NVFP4 summary](../2026-09-27-qwen38-ninfer-expanded/summary-qwen38-ninfer-nvfp4-tp2-2x5060ti.json),
[Q8_0 summary](../2026-09-27-qwen38-q8-expanded/summary-qwen38-27b-q8-vision-mtp-tensor.json),
[comparison JSON](comparison.json). Harness revision: [`9ff064e`](https://github.com/crowrain/llm-tests-1/commit/9ff064e).
