# Publication validation — 2026-10-26

Fresh build and device regression completed on TG-V200 / COREX 4.3.
The test used a new container temporary directory and did not replace the installed runtime.

- Compiled 15 model/API translation units from source and eight support libraries.
- All 31 build/link/inspection commands passed.
- Reused the pinned InfiniCore and operator libraries listed in DEPENDENCIES.md.
- Compared frozen R5 and rebuilt R5 in separate processes for each case.
- Each process ran two warmups and one measured request; all three outputs were retained.
- Verified that the intended model, dependency and preload libraries were mapped.
- All request outputs matched across repeats and between the two builds.

| Model | Batch / input / output | Frozen R5 ms | Rebuilt R5 ms | Output comparison |
| --- | --- | ---: | ---: | --- |
| 9G7B_MHA | 4 / 512 / 8 | 615.212 | 613.057 | All three requests identical |
| 9G7B_MHA | 16 / 2048 / 16 | 8160.463 | 8156.991 | All three requests identical |
| 9g_8b_thinking_llama | 1 / 128 / 8 | 337.592 | 327.237 | All three requests identical |

These are publication smoke timings: input construction/cache reset precedes the timer,
and the table shows one measured request per process. They are not a repeated statistical
performance study, a new optimization gain, a full accuracy benchmark or online capacity.
The formal combined-profile numbers and their different timing boundary are in BENCHMARKS.md.

The checks cover 18 request executions: three configurations × two runtimes × three repeats.
The EOS optimization is present, but these fixed-length requests use `stop_on_eos=False`;
they do not independently retest early EOS termination.

Source manifest SHA256: `9c0016b096e475dd9412ad8cc21ad676789fe4942b2614846cc94b92931dec29`.
Rebuilt model SHA256: `7188ed4eb16595b7b8aa4f652217dd5df9633955058d9b7acced40bad8ce3231`.

Machine-readable summary: [validation.json](validation.json).
Current-main compilation, current-main benchmark/service tests, other accelerators, and
a complete InfiniCore source rebuild were not performed. The upstream PR remains a draft.

## Repeat the model check

Edit model locations in `cases.example.json` and save it as `cases.json` in a test directory.
After building, run the check in a fresh process for each case:

```sh
python optimizations/r5/runtime.py --build /tmp/r5-new-build --gpu 0 -- \
  python optimizations/r5/check_model.py --root /path/to/test-dir --case mha_b4 --output /path/to/check.json
```

Repeat for `mha_b16` and `second_b1`. Compare the retained `requests[].tokens` with the
frozen R5 results in the same environment. A matching checksum alone is insufficient.
