# R5 performance and output checks

## Latest measured application results

9G7B_MHA, BF16, TP1, TG-V200, static cache, non-Graph. B16 was measured on
2026-10-23 and B4 on 2026-10-26. The native baseline is COREX PyTorch eager.

| Batch / input / output | Runtime | TTFT ms | TPOT ms | Request ms | Output token/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 4 / 512 / 8 | Native eager | 329.965 | 65.985 | 788.340 | 40.592 |
| 4 / 512 / 8 | Original InfiniLM | 475.162 | 51.310 | 834.530 | 38.345 |
| 4 / 512 / 8 | R5 | 284.103 | 46.575 | 610.143 | 52.447 |
| 16 / 2048 / 16 | Native eager | 6865.398 | 244.501 | 10532.960 | 24.305 |
| 16 / 2048 / 16 | Original InfiniLM | 8074.136 | 154.675 | 10394.245 | 24.629 |
| 16 / 2048 / 16 | R5 | 5827.561 | 152.226 | 8111.702 | 31.559 |

Timing starts when CPU input token IDs are ready, before input/cache preparation.
TTFT ends when the first generated token for every batch row is visible on CPU.
TPOT is the interval between first and last token availability divided by output
steps minus one; it is not divided by batch size. Loading, tokenization, text
decoding and queueing are excluded. Fixed output length is used. These rates are
fixed-batch measurements, not online serving capacity.

Each comparison uses three rotated rounds across the three runtimes. Each process
performs two warmups and three measured requests. Values use the median within
each process and then the median across the three processes. Per-request output
rate is aggregated independently, so it is not calculated from a rounded table
latency. The numbers describe the combined profile, not isolated component gains.

Original evidence archive SHA256:

- B4: `4e8d5e99bb4e01725ce6fa82eb47d336126b304ea0fe02b61132c24f851074be`
- B16: `0cc02475d26b3bd8ba9e5aa1e5c113ef6a5315b47afd7176e6700e687c951812`

## Quality scope

The earlier six-question first-model screen scored 6/6 for original InfiniLM and
R5, and 5/6 for native eager. On the second model, `9g_8b_thinking_llama`, each of
the three runtimes scored MMLU 2/2, C-Eval 2/2 and GSM8K 0/2. Both GSM8K outputs
were truncated at the unchanged 512-token budget. Original InfiniLM and R5 matched
all 1,028 generated token IDs and the four multiple-choice scoring results.
This small screen is not a full benchmark accuracy claim.

Second-model B4 performance and real online capacity were not measured. Embedding
gather and vector-copy experiments are separate candidates and are not included
in the table or this frozen R5 profile. The historical 48-value discrepancy is
not claimed resolved by these later successful checks.

## This publication build

The fresh source build was compared with the frozen R5 on two models and three
configurations. See [VALIDATION.md](VALIDATION.md). Its timings are smoke checks
under a different timing boundary; they do not replace the formal table above.
