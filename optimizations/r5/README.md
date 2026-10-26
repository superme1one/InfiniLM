# TG-V200 R5 validated combination

This branch contains the R5 source combination used for the project measurements
through 2026-10-26. It preserves the tested legacy InfiniCore API. It is a source
handoff for upstream review, not a port to the reorganized current InfiniLM main.
The new standalone build was checked on TG-V200 against the frozen R5 runtime.
See [VALIDATION.md](VALIDATION.md) and [BENCHMARKS.md](BENCHMARKS.md).

## Included optimizations

| Component | Change | Source |
| --- | --- | --- |
| Add | Contiguous BF16 fast path with guarded fallback | `src/add_contiguous_override.cu` |
| SwiGLU | Regular-stride access and reduced indexing overhead | `src/swiglu_stride_override.cu` |
| Add + RMSNorm | Fuse the layer-local residual add and normalization | `src/add_rms_fusion.cu`, decoder layer |
| Length information | Reuse host length metadata instead of repeated device readbacks | `src/length_mirror.cpp` |
| Vocabulary projection | Project the final prompt position; use the tested short-row policy | `src/last_logits_short_scope.cpp`, rank worker and LM head |
| Softmax | Select a thread configuration by row length | `src/softmax_row_bucket_override.cu` |
| RoPE | Tested backend thread selection and paired Q/K path | `src/rope_nvidia.cu`, `src/rope_pair*` |
| Batch EOS | Exit once all rows have reached EOS when enabled by the generation request | `python/infinilm/infer_engine.py` |

`runtime.py` selects the validated combination. M1, cross-layer residual chaining
and decode-layout candidates are disabled. Embedding experiments are kept in the
separate validation repository and are not added to R5. The original source tree
also contains earlier, disabled experiments; their presence is not evidence of
an accepted optimization. This snapshot deliberately preserves source identity
before upstream components are split and adapted.

## Pinned environment

| Dependency | Revision / environment |
| --- | --- |
| InfiniLM base | `d21a4f59ba599f0ca67ebe7025f008c4162fc0ee` |
| InfiniCore | `5fdc80ddc2241fded2eb500665db9adce6d8cc63` |
| InfiniOps checkout | `4e55e6df614638f03b2bd8ae73c396639657a443` |
| Device / SDK | Iluvatar TG-V200, COREX 4.3 |
| Compiler / Python | GCC 9.4, COREX clang targeting `ivcore20`, Python 3.10.18 |
| Request mode | BF16, tensor parallel 1, static cache, non-Graph |

The revisions identify the installed source bases; R5 also uses the recorded
logging-isolated InfiniCore library and patched RoPE operator library. A stock
checkout at those revisions alone is not the complete R5 runtime. Runtime hashes
are recorded in [DEPENDENCIES.md](DEPENDENCIES.md); matching dependency libraries
must be supplied separately. No SDK, model weights or shared binaries are committed.

## Build into a separate directory

Use `build.py` for this snapshot. The legacy default `xmake` target does not link
the R5 support libraries. Supply the matching SDK headers, InfiniCore source,
spdlog and pybind11 headers, and a JSON file describing the two pinned libraries:

```json
{
  "core": "/path/to/libinfinicore_cpp_api.so",
  "operator_library": "/path/to/libinfiniop.so",
  "sha256": {
    "/path/to/libinfinicore_cpp_api.so": "ACTUAL_SHA256",
    "/path/to/libinfiniop.so": "ACTUAL_SHA256"
  }
}
```

```sh
python optimizations/r5/build.py --output /tmp/r5-new-build \
  --infini-root /path/to/installed/infini --core-source /path/to/InfiniCore \
  --corex /usr/local/corex --spdlog-include /path/to/spdlog/include \
  --pybind-include /path/to/pybind11/include --runtime-config /path/to/runtime-input.json
python optimizations/r5/runtime.py --build /tmp/r5-new-build --gpu 0 -- python your_inference_script.py
```

The builder verifies `SOURCE_MANIFEST.json`, rebuilds the model translation units
and eight support libraries, and writes its commands, logs and runtime hashes.
It does not rebuild all of InfiniCore or alter the installed runtime. It needs a
new output directory and bounds each compiler process to 45 seconds. The launcher
uses a fresh process because the selected libraries and optimization switches
must be set before imports.

## Upstream integration

The submitted branch preserves the complete working R5 combination. The upstream
PR is a draft because main has since replaced the old InfiniCore execution APIs.
Model-side changes belong in InfiniLM; backend Add, SwiGLU, Softmax and RoPE changes
need coordination with InfiniOps/InfiniCore. Current-main migration, formatting,
other backends and current-main service tests remain required before merging.
