# Qwen3.8 — expanded benchmark

266 identical text cases, `temperature=0`, seed 20260926, concurrency 1. All runs used
the same fixtures. Accuracy includes request errors in the denominator.

| Model and quant | Inference and hardware | Accuracy | Prefill | Decode | Time |
|---|---|---:|---:|---:|---:|
| Qwen3.8 27B **NVFP4** | NInfer TP2, MTP; **2× RTX 5060 Ti 16 GB** | 242/266 (**90.98%**) | **706.3 tok/s** | **95.1 tok/s** | **22:58** |
| Qwen3.8 27B **Q8_0 GGUF** | llama.cpp v0.4.0, MTP, tensor split; **RTX 4080 SUPER 16 GB + 2× RTX 5060 Ti 16 GB** | 243/266 (**91.35%**) | 202.3 tok/s | 81.2 tok/s | 31:59 |
| Qwen3.8 27B **GSQ-RCO IQ3_S** | llama.cpp v0.4.0, MTP; **RTX 4080 SUPER 16 GB** | 239/266 (**89.85%**) | 244.4 tok/s | 93.3 tok/s | 20:19 |
| Qwen3.8 Flash-Next 125B-A6B **Q4_K_XL** | llama.cpp PR 28243, MoE, MTP, CPU offload; **RTX 4080 SUPER 16 GB + 2× RTX 5060 Ti 16 GB** | **246/266 (92.48%)** | 39.3 tok/s | 30.6 tok/s | 1:36:49 |

Shared host: AMD Ryzen 9 5900X, NVIDIA driver 595.91.07. The GGUF runs used Q8_0 KV cache;
NVFP4 used INT8 KV cache. This compares complete serving configurations, not quantization
in isolation. IQ3_S scored 90.87% over 263 accepted requests; its 49,152-token context
rejected the 64k, 128k and 180k needles. Flash-Next scored 93.18% over 264 accepted
requests; two MMLU requests returned HTTP 502.

| Category | NVFP4 | Q8_0 | IQ3_S | Flash-Next |
|---|---:|---:|---:|---:|
| GSM8K | 96% | 96% | 96% | 97% |
| ARC-Challenge | 97% | 97% | 97% | 96% |
| MMLU | 66% | 68% | 66% | 76% |
| Instruction JSON | 100% | 100% | 100% | 91.67% |
| Long-context needles: 16k / 64k / 128k / 180k | 4/4 | 4/4 | 1/4 | 4/4 |

Artifacts: [NVFP4 summary](../2026-09-27-qwen38-ninfer-expanded/summary-qwen38-ninfer-nvfp4-tp2-2x5060ti.json),
[Q8_0 summary](../2026-09-27-qwen38-q8-expanded/summary-qwen38-27b-q8-vision-mtp-tensor.json),
[IQ3_S summary](../2026-09-27-qwen38-gsq-rco-expanded/summary-qwen38-27b-gsq-rco-iq3s-mtp-4080super.json),
[Flash-Next summary](../2026-09-27-qwen38-flashnext-expanded/summary-qwen38-flashnext-125b-q4kxl-moe-mtp-cpuoffload.json),
[NVFP4 vs Q8_0 comparison JSON](comparison.json). Harness revision: [`9ff064e`](https://github.com/crowrain/llm-tests-1/commit/9ff064e).
