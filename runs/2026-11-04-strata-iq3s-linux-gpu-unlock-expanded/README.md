# Qwen3.8 Flash-Next IQ3_S — expanded benchmark after gpu-unlock

Полный expanded-профиль выполнен 4 ноября 2026 года через llama-swap и Strata на Ubuntu 26.04.1. Это повтор опубликованного Linux-прогона IQ3_S с включённым `gpu-unlock` 2.0.0 для GPU `0000:2b:00.0`. Использованы те же fixtures, system prompt и параметры теста, что в [исходном прогоне](../2026-11-04-strata-iq3s-linux-expanded/README.md); SHA-256 fixtures `264f93d457e08fd4522d2944813f11cf66805f2c6469ca34070566f8d5f48337`. [Общие fixtures](../2026-11-04-strata-iq3s-linux-expanded/fixtures.json) не копируются повторно.

Хост: AMD Ryzen 7 5700X, 121 ГиБ RAM, GeForce RTX 4080 SUPER 16 ГБ, NVIDIA 595.99.02. Strata 0.2.10 (CUDA engine build 0.1.56), llama-swap v269. MTP включён; контекст 262144, KV int8, 32768 токенов KV resident, PLE I/O в RAM; `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5`. Подробные версии и опции находятся в [metadata](run_metadata.json).

## Качество и генерация

Использованы 266 заданий, температура 0, seed 20260926, reasoning effort medium, concurrency 1, лимит ответа 4096 токенов.

| Категория | Верно / всего |
|---|---:|
| GSM8K | 98 / 100 |
| ARC-Challenge | 97 / 100 |
| MMLU | 49 / 50 |
| JSON-инструкции | 12 / 12 |
| Needle, 16k / 64k / 128k / 180k | 4 / 4 |
| **Итого** | **260 / 266 (97,74%)** |

Ошибок запросов и усечённых ответов нет; шесть завершённых ответов неверны. Медианная скорость генерации — **102,35 ток/с**. Полный прогон с 30 prefill-пробами занял **26:07**. Самый длинный вход содержал 182244 токена.

## Prefill

Для каждой длины измерены три холодных запроса и три повтора. Скорость cold указана по backend таймерам; повторы подтверждены счётчиками кэша.

| Длина | Медиана входных токенов | Повторно использовано токенов |
|---|---:|---:|
| 16K | 16299 | 16294 / 16299 |
| 64K | 64855 | 64850 / 64855 |
| 128K | 129593 | 129588 / 129593 |
| 180K | 182244 | 182239 / 182244 |

| Длина | Cold backend prefill, ток/с |
|---|---:|
| 16K | 3579,3 |
| 64K | 3484,5 |
| 128K | 3350,3 |
| 180K | 3230,4 |

В сравнении с предыдущим прогоном на тех же fixtures cold prefill вырос на 5,7–7,0%; медианный decode изменился с 103,55 до 102,35 ток/с. Результат качества — 260/266 вместо 261/266, единственное изменение пришлось на ARC-Challenge. Это сравнение двух отдельных полных прогонов; один ответ и небольшое изменение скорости сами по себе не доказывают статистически устойчивое изменение качества или decode.

## Артефакты

- [Сводка JSON](summary-strata-iq3s-linux-gpu-unlock.json)
- [266 результатов и decode timings](results-strata-iq3s-linux-gpu-unlock.jsonl)
- [30 cold/cached prefill-проб](prefill-strata-iq3s-linux-gpu-unlock.jsonl)
- [Сравнение с прогоном до gpu-unlock](comparison-vs-before-gpu-unlock.json)
- [Метаданные хоста и запуска](run_metadata.json)
- [Журнал запуска](run.log)
