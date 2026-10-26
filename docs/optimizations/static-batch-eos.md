# Static-batch EOS early stopping

## Change

`InferEngine.generate()` previously stopped on EOS only for a single sequence.
A static batch therefore continued to `max_new_tokens` after every row had
already produced an EOS token. This change adds the opt-in
`GenerationConfig.stop_batch_on_eos` flag. Each row's first EOS is remembered;
the loop stops once every row has finished. The last step remains in the output
and in the optional timing report.

The batch is not compacted, KV slots are not moved, and unfinished rows are not
modified. Early-finished rows still run until the last row ends. Consumers should
continue to interpret each answer through its first EOS. No post-EOS tokens are
masked or rewritten.

```python
from infinilm.infer_engine import GenerationConfig

config = GenerationConfig(
    max_new_tokens=512,
    temperature=1.0,
    top_k=1,
    top_p=1.0,
    stop_on_eos=True,
    stop_batch_on_eos=True,
)
outputs = engine.generate(input_ids, config)
```

The new flag defaults to `False`. The optimized path requires batch > 1,
`stop_on_eos=True`, a nonempty engine EOS list, static attention, no Mamba cache,
and no `pixel_values`. Single-sequence, paged, Mamba, multimodal and fixed-budget
generation keep their existing paths. With any unfinished row, generation still
uses the original budget.

## Historical measurements motivating this port

These are measured results from the isolated legacy InfiniLM implementation on
2026-09-17, not measurements of this new-main port. Hardware: Iluvatar TG-V200;
model: 9G7B_MHA, BF16. Twenty original GSM8K prompts of 72 tokens were selected
before inference, without padding or truncation, and split into batches of 4
and 16. The greedy configuration, EOS IDs, 512-token budget and strict answer
scoring were unchanged. All tested implementations scored 12/20; effective
output tokens matched per sample through first EOS.

| Batch | Baseline generation ms | EOS-enabled ms | Time reduction | Forward calls |
| --- | ---: | ---: | ---: | --- |
| 4 | 22784.281531 | 11644.036553 | 48.8944% | 512 -> 265 |
| 16 | 29800.565531 | 17995.784120 | 39.6126% | 512 -> 335 |

Each batch used three AB/BA/AB pairs of independent processes, one warmup and
one measured request per process. The table is the median of three measured
requests per arm. Timing includes `generate()` and completion synchronization;
it excludes model loading, tokenization, input construction, cache reset and
answer decoding. No profiler or resource sampler was loaded. The benefit is
avoided finished-batch work, not faster GPU kernels. It depends on the EOS
distribution and must not be added to other optimization percentages.

Historical evidence identities:

- Batch 4 independent check SHA256:
  `f97d64c920c57e24674975b1028fb83c308ae11574fbd68f712818fea6af8af9`.
- Batch 16 independent check SHA256:
  `73e82be19e75ea7cfbad190d7652ca28495f10086aae8f5548176735724a1d84`.
- Source reports: `batch_eos_v48_20260917.md` and
  `batch_eos_perf_v49_20260917.md` in the project's retained experiment archive.
  Model weights, dataset payloads and internal infrastructure logs are not
  distributed in this branch.

## Current branch validation

Port base: InfiniTensor/InfiniLM `eb97954b518cf33fbf3ee2dc22f13ecf2141400b`.

```shell
python -m unittest discover -s test/runtime -p test_static_batch_eos.py -v
python scripts/format.py --check --path python/infinilm/infer_engine.py test/runtime/test_static_batch_eos.py
git diff --check
```

Fourteen CPU tests exercise the actual `generate()` method with tensor fixtures:
default-off behavior, asynchronous EOS arrival, unfinished rows, multiple EOS
IDs, fixed-budget generation, batch 1/16, paged/multimodal fallback, malformed
output cardinality, effective answer preservation and terminal-step timing.
They do not load a native extension or establish GPU quality/performance.

The legacy implementation reused an existing host-token read. Current main
feeds the output tensor directly into the next iteration; enabling this option
adds a `to_numpy()` read at each step. Its cost and the net benefit must be
measured on this exact branch. No new-main GPU build, model, benchmark or
service result is claimed. Before merge, run original/disabled/enabled on the
same inputs, compare each row through first EOS, and report repeated timings
both when all rows finish early and when at least one row reaches the budget.

## 中文说明

本改动处理静态批次已经全部生成结束、仍继续计算到最大长度的情况。
为每条序列记录第一次出现结束标记的位置，全部结束后退出生成循环。
不改变模型算子，不压缩批次，不移动缓存，默认关闭，通过配置显式启用。

历史天数测试中，批量4和16的生成耗时分别降低48.89%和39.61%；这些数据
属于旧版隔离实验。当前官方主分支接口已经变化，本分支完成了代码迁移及
CPU行为测试，正式合并前需要在目标设备重新测量。独立的功能性能验证工具
单独整理，后续按官方目录和组件归属逐项提交。
