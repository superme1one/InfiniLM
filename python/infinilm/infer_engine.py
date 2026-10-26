import time
from dataclasses import dataclass

import infinicore

from infinilm.auto_config import AutoConfig
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.lib import _infinilm


@dataclass
class GenerationConfig:
    max_new_tokens: int | None = None

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    eos_token_id: list[int] | None = None
    stop_on_eos: bool = True


class InferEngine(_infinilm.InferEngine):
    def __init__(
        self,
        model_path,
        device=None,
        distributed_config=DistConfig(1),
        cache_config=None,
    ):
        # INFINILM_V_CACHE_DECODE_LAYOUT_DISPATCH_POLICY
        # INFINILM_V_CACHE_DECODE_LAYOUT_POLICY_REVISION_2
        # INFINILM_V_CACHE_DECODE_LAYOUT_POLICY_REVISION_3
        self._infinilm_model_path = str(model_path)
        self.config = AutoConfig.from_pretrained(model_path)

        if device is None:
            device = infinicore.device()

        super().__init__(
            self.config,
            distributed_config._underlying,
            device._underlying.type,
            cache_config,
        )

        self.use_cache = False

        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input_ids,
        *,
        position_ids=None,
        past_kv_lengths=None,
        total_kv_lengths=None,
        input_offsets=None,
        block_tables=None,
        slot_mapping=None,
        temperature=None,
        top_k=None,
        top_p=None,
    ):
        # TODO: Remove `_underlying` and simplify the corresponding code.
        input_ids = input_ids._underlying if input_ids is not None else None
        position_ids = position_ids._underlying if position_ids is not None else None
        past_kv_lengths = (
            past_kv_lengths._underlying if past_kv_lengths is not None else None
        )
        total_kv_lengths = (
            total_kv_lengths._underlying if past_kv_lengths is not None else None
        )
        input_offsets = input_offsets._underlying if input_offsets is not None else None
        block_tables = block_tables._underlying if block_tables is not None else None
        slot_mapping = slot_mapping._underlying if slot_mapping is not None else None

        return infinicore.Tensor(
            super()
            .forward(
                super().Input(
                    input_ids,
                    position_ids=position_ids,
                    past_sequence_lengths=past_kv_lengths,
                    total_sequence_lengths=total_kv_lengths,
                    input_offsets=input_offsets,
                    block_tables=block_tables,
                    slot_mapping=slot_mapping,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
            .output_ids
        )

    def generate(
        self,
        input_ids,
        generation_config,
        *,
        _measure_and_log_time=False,
        paged_block_size=16,
    ):
        if generation_config.eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        else:
            eos_token_id = generation_config.eos_token_id

        past_seq_len = 0
        output_ids = []
        initial_batch_size, initial_seqlen = input_ids.shape[:2]
        seq_len = initial_seqlen
        batch_size = initial_batch_size

        import os as _iv_eos_os
        # Experimental static-batch completion: no row compaction or KV move.
        # Reuse the host token read already needed for the next model input.
        batch_eos_enabled = (
            _iv_eos_os.environ.get("IV_OPT_BATCH_EOS") == "1"
            and initial_batch_size > 1
            and generation_config.stop_on_eos
            and not self.enable_paged_attn
        )
        batch_finished = [False] * initial_batch_size

        if batch_size != 1 and generation_config.max_new_tokens is None:
            raise ValueError(
                "When `batch_size > 1`, `max_new_tokens` must be specified."
            )

        if _measure_and_log_time:
            time_measurements = []

        for iter in range(0, generation_config.max_new_tokens):
            if _measure_and_log_time:
                start_time = time.perf_counter()

            batch_size, seq_len = input_ids.shape[:2]

            if self.enable_paged_attn:
                input_ids = input_ids.view([1, batch_size * seq_len])
                position_ids = infinicore.from_list(
                    list(range(past_seq_len, past_seq_len + seq_len)) * batch_size,
                    dtype=infinicore.int64,
                )
                block_tables_list = [
                    [
                        i * batch_size + b
                        for i in range(
                            (past_seq_len + seq_len + paged_block_size - 1)
                            // paged_block_size
                        )
                    ]
                    for b in range(batch_size)
                ]
                slot_mapping_list = [
                    (((past_seq_len + i) // paged_block_size) * batch_size + b)
                    * paged_block_size
                    + (past_seq_len + i) % paged_block_size
                    for b in range(batch_size)
                    for i in range(seq_len)
                ]

                block_tables = infinicore.from_list(
                    block_tables_list,
                    dtype=infinicore.int64,
                )
                slot_mapping = infinicore.from_list(
                    slot_mapping_list,
                    dtype=infinicore.int64,
                )
            else:
                position_ids = infinicore.from_list(
                    [
                        list(range(past_seq_len, past_seq_len + seq_len))
                        for _ in range(batch_size)
                    ],
                    dtype=infinicore.int64,
                )

                block_tables = None
                slot_mapping = None

            past_kv_lengths = infinicore.from_list(
                [past_seq_len] * batch_size, dtype=infinicore.int64
            )
            total_kv_lengths = infinicore.from_list(
                [past_seq_len + seq_len] * batch_size, dtype=infinicore.int64
            )

            input_offsets = infinicore.from_list(
                [seq_len * i for i in range(batch_size + 1)], dtype=infinicore.int64
            )

            output_id = self(
                input_ids=input_ids,
                position_ids=position_ids,
                past_kv_lengths=past_kv_lengths,
                total_kv_lengths=total_kv_lengths,
                input_offsets=input_offsets,
                block_tables=block_tables,
                slot_mapping=slot_mapping,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            output_ids.append(output_id)

            if (
                initial_batch_size == 1
                and generation_config.stop_on_eos
                and generation_config.max_new_tokens is not None
                and output_id.to_numpy()[0] in eos_token_id
            ):
                break

            next_host_tokens = output_id.to_numpy().tolist()
            if batch_eos_enabled:
                if len(next_host_tokens) != initial_batch_size:
                    raise ValueError("Static batch output cardinality changed")
                batch_finished = [
                    done or token in eos_token_id
                    for done, token in zip(batch_finished, next_host_tokens)
                ]
                if all(batch_finished):
                    if _measure_and_log_time:
                        time_measurements.append(time.perf_counter() - start_time)
                    break
            input_ids = infinicore.from_list(
                [[token] for token in next_host_tokens]
            )
            past_seq_len = past_seq_len + seq_len

            if _measure_and_log_time:
                end_time = time.perf_counter()

                time_measurements.append((end_time - start_time))

        if _measure_and_log_time:
            print(
                f"\n\n\n Generation completed in {round(sum(time_measurements) * 1000, 2)} ms"
            )
            print(
                f" Batchsize={initial_batch_size}  Per_Batch_Input_Len={initial_seqlen}  Per_Batch_New_Tokens={len(time_measurements)}\n"
            )
            print(
                f" Prefill TTFT: {round(time_measurements[0] * 1000, 2)} ms  Throughput: {round((initial_batch_size * initial_seqlen) / time_measurements[0], 2)} tok/s\n",
            )
            if len(time_measurements) > 1:
                print(
                    f" Decode  Avg ITL: {round(sum(time_measurements[1:]) * 1000 / (len(time_measurements) - 1), 2)} ms   Throughput: {round((initial_batch_size * (len(time_measurements) - 1)) / sum(time_measurements[1:]), 2)} tok/s\n",
                )

        return output_ids

    def configure_v_cache_decode_layout_policy(
        self,
        *,
        batch_size,
        input_len,
        max_new_tokens,
        enabled=True,
    ):
        """Configure the validated StaticKV V-layout policy before reset_cache."""
        import json
        import os
        import sys

        policy_keys = (
            "INFINILM_KV_PAD_MODE",
            "INFINICORE_QK_FIXED_KERNEL",
            "INFINILM_KV_PAD_ALIGN",
            "INFINILM_KV_PAD_DEBUG",
            "INFINICORE_V_FIXED_KERNEL",
            "INFINICORE_V_FIXED_MIN_K",
            "INFINICORE_V_FIXED_BLOCK_ROWS",
            "INFINICORE_V_FIXED_TRACE",
            "INFINILM_V_CACHE_DECODE_LAYOUT",
            "INFINILM_V_CACHE_DECODE_LAYOUT_TRACE",
            "INFINILM_V_CACHE_DECODE_LAYOUT_FALLBACK_LAYERS",
        )
        for key in policy_keys:
            os.environ.pop(key, None)

        batch_size = int(batch_size)
        input_len = int(input_len)
        max_new_tokens = int(max_new_tokens)
        expected_model = os.getenv(
            "INFINILM_V_CACHE_DECODE_LAYOUT_POLICY_MODEL",
            "9G7B_MHA",
        )
        model_name = os.path.basename(
            os.path.normpath(self._infinilm_model_path)
        )
        dtype_value = getattr(self.config, "dtype", None)
        dtype_text = str(dtype_value).lower()
        dtype_ok = (
            dtype_value == getattr(infinicore, "bfloat16", None)
            or "bf16" in dtype_text
            or "bfloat16" in dtype_text
        )
        attention_heads = int(
            getattr(self.config, "num_attention_heads", 0) or 0
        )
        kv_heads = int(
            getattr(
                self.config,
                "num_key_value_heads",
                attention_heads,
            )
            or attention_heads
        )
        head_dim = int(getattr(self.config, "head_dim", 0) or 0)
        if head_dim == 0 and attention_heads > 0:
            hidden_size = int(
                getattr(self.config, "hidden_size", 0) or 0
            )
            if hidden_size % attention_heads == 0:
                head_dim = hidden_size // attention_heads

        reasons = []
        experimental = os.getenv(
            "INFINILM_V_CACHE_DECODE_LAYOUT_EXPERIMENTAL",
            "",
        )
        if experimental != "1":
            reasons.append("policy_revoked_prompt_sensitive")
        if not enabled:
            reasons.append("policy_disabled")
        if model_name != expected_model:
            reasons.append("model_not_validated")
        if self.enable_paged_attn:
            reasons.append("paged_cache")
        if not dtype_ok:
            reasons.append("dtype_not_bf16")
        if head_dim != 128:
            reasons.append("head_dim_not_128")
        if attention_heads <= 0 or kv_heads != attention_heads:
            reasons.append("not_mha")
        if batch_size != 16:
            reasons.append("batch_not_validated")
        if input_len != 2048:
            reasons.append("input_len_not_validated")
        if max_new_tokens <= 0 or max_new_tokens > 64:
            reasons.append("output_len_not_validated")

        action = "legacy"
        if not reasons:
            action = "v_cache_layout"
            os.environ["INFINILM_KV_PAD_MODE"] = "qk_fixed"
            os.environ[
                "INFINICORE_QK_FIXED_KERNEL"
            ] = "warp_fma_contiguous_down"
            os.environ["INFINILM_KV_PAD_ALIGN"] = "32"
            os.environ[
                "INFINILM_V_CACHE_DECODE_LAYOUT"
            ] = "page16_fp32_warp"
            os.environ[
                "INFINILM_V_CACHE_DECODE_LAYOUT_FALLBACK_LAYERS"
            ] = "0,11"

        decision = {
            "action": action,
            "model": model_name,
            "batch_size": batch_size,
            "input_len": input_len,
            "max_new_tokens": max_new_tokens,
            "reasons": reasons,
        }
        trace = os.getenv(
            "INFINILM_V_CACHE_DECODE_LAYOUT_POLICY_TRACE",
            "",
        )
        if trace and trace != "0":
            print(
                "INFINILM_V_CACHE_DECODE_LAYOUT_POLICY_ROUTE "
                + json.dumps(decision, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
        return decision

    def reset_cache(self, cache_config):
        infinicore.sync_device()
        self.enable_paged_attn = isinstance(cache_config, PagedKVCacheConfig)
        super().reset_cache(cache_config)

    def state_dict_keyname(self):
        return super().state_dict()[0].keys()

    def load_state_dict(self, state_dict, strict=None):
        for name, param in state_dict.items():
            super().load_param(name, param._underlying)
