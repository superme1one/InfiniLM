"""Compare a repackaged R5 build with the frozen R5 using real model requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser()
p.add_argument('--case',required=True)
p.add_argument('--output',required=True)
p.add_argument('--root',required=True)
a=p.parse_args()
root=Path(a.root)
case=json.loads((root/'cases.json').read_text())[a.case]
import infinicore as api
from infinilm.infer_engine import InferEngine,GenerationConfig
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.modeling_utils import load_model_state_dict_by_file
import infinilm.infer_engine as engine_source
model=InferEngine(case['model'],device=api.device('cuda',0),distributed_config=DistConfig(1),
    cache_config=StaticKVCacheConfig(max_batch_size=case['batch'],max_cache_len=case['capacity']))
assert model.config.dtype==api.bfloat16 and not model.enable_paged_attn
load_model_state_dict_by_file(model,case['model'],dtype=model.config.dtype)
print('MODEL_LOADED',a.case,flush=True)
requests=[]
for repeat in range(3):
    model.reset_cache(StaticKVCacheConfig(max_batch_size=case['batch'],max_cache_len=case['capacity']))
    ids=api.from_list([case['ids'] for _ in range(case['batch'])])
    api.sync_device();start=time.perf_counter()
    outputs=model.generate(ids,GenerationConfig(max_new_tokens=case['output_len'],temperature=1.0,
        top_k=1,top_p=1.0,stop_on_eos=False))
    api.sync_device();ms=(time.perf_counter()-start)*1000
    tokens=[[int(t) for t in output.to_numpy().reshape(-1)] for output in outputs]
    assert len(tokens)==case['output_len'] and all(len(row)==case['batch'] for row in tokens)
    assert not requests or tokens==requests[0]['tokens'],'Output changed across repeated requests.'
    requests.append(dict(repeat=repeat,kind='warmup' if repeat<2 else 'measured',elapsed_ms=ms,tokens=tokens))
    print('REQUEST_COMPLETE',repeat,round(ms,3),flush=True)
modules=sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if '/' in s})
record=dict(case=a.case,model=case['model'],batch=case['batch'],input_len=len(case['ids']),output_len=case['output_len'],
    requests=requests,source=str(Path(engine_source.__file__).resolve()),
    source_sha256=hashlib.sha256(Path(engine_source.__file__).read_bytes()).hexdigest(),
    modules=modules,flags={k:v for k,v in os.environ.items() if k.startswith('IV_OPT_')},
    preloads=os.environ['LD_PRELOAD'].split(':'),full_task_quality_test=False,publication_regression=True)
Path(a.output).write_text(json.dumps(record,indent=2))
print('MODEL_CHECK_COMPLETE',a.case,flush=True)
