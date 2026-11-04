# Qwen3.8 Flash-Next IQ3_S — expanded benchmark

Полный expanded-профиль выполнен 4 ноября 2026 года через llama-swap и Strata на Ubuntu 26.04.1. Использованы те же fixtures и system prompt, что в прогоне 2 ноября: SHA-256 fixtures `264f93d457e08fd4522d2944813f11cf66805f2c6469ca34070566f8d5f48337`, system prompt `0dbe0ad3e7e2fc9d01082ca5096345546a0074157d071e07b07aa8fb012d1563`. [Общие fixtures](../2026-11-02-basalt-split-expanded/fixtures.json) не копируются повторно.

Хост: AMD Ryzen 7 5700X, 121 ГиБ RAM, GeForce RTX 4080 SUPER 16 ГБ, NVIDIA 595.99.02. Strata 0.2.10 (CUDA engine build 0.1.56), llama-swap v269. Модель Qwen3.8-Flash-Next GSQ-RCO IQ3_S (ISTA-DASLab). MTP включён; контекст 262144, KV int8, 32768 токенов KV resident, PLE I/O в RAM; `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`.

## Качество и генерация

Использованы 266 заданий, температура 0, seed 20260926, reasoning effort medium, concurrency 1, лимит ответа 4096 токенов.

| Категория | Верно / всего |
|---|---:|
| GSM8K | 98 / 100 |
| ARC-Challenge | 98 / 100 |
| MMLU | 49 / 50 |
| JSON-инструкции | 12 / 12 |
| Needle, 16k / 64k / 128k / 180k | 4 / 4 |
| **Итого** | **261 / 266 (98,12%)** |

Ошибок запросов и усечённых ответов нет; пять завершённых ответов неверны. Медианная скорость генерации — **103.55 ток/с**. Полный прогон с 30 prefill-пробами занял **28:09**. Самый длинный вход содержал 182244 токена.

## Prefill

Для каждой длины измерены три холодных запроса и три повтора. Скорость cold указана по backend таймерам; повторы подтверждены счётчиками кэша.

| Вход | Cold prefill, ток/с | Повторно использовано токенов |
|---|---:|---:|
| 16K | 3348.3 | 16294 / 16299 |
| 64K | 3258.0 | 64850 / 64855 |
| 128K | 3162.9 | 129588 / 129593 |
| 180K | 3055.5 | 182239 / 182244 |

## Артефакты

- [Сводка JSON](summary-strata-iq3s-linux.json)
- [266 результатов и decode timings](results-strata-iq3s-linux.jsonl)
- [30 cold/cached prefill-проб](prefill-strata-iq3s-linux.jsonl)
- [Метаданные хоста и запуска](run_metadata.json)
