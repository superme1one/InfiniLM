"""Launch a new process with the exact R5 profile and pinned runtime libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def environment(runtime, gpu):
    for path, wanted in runtime['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != wanted:
            raise ValueError('Runtime dependency changed: ' + path)
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(('IV_', 'INFINILM_', 'INFINICORE_')) or key in ('LD_PRELOAD','LD_AUDIT'):
            env.pop(key)
    env.update({('IV_OPT_' + name):'1' for name in ('ADD','SWIGLU','FUSION','LENGTH','LAST_LOGITS',
                    'SHORT_LOGITS','SOFTMAX','ROPE_BACKEND','ROPE_PAIR','BATCH_EOS')})
    env.update(IV_OPT_CHAIN='0',IV_OPT_DECODE_LAYOUT='0',IV_MLP_M1_V83_MODE='off',IV_LOGITS_PAD_ROWS='64',
               CUDA_VISIBLE_DEVICES=str(gpu),IX_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    corex = Path(runtime['corex'])
    env['PATH'] = str(corex / 'bin') + ':/usr/local/bin:/usr/bin:/bin'
    env['PYTHONPATH'] = runtime['root'] + '/python:' + str(corex / 'lib64/python3/dist-packages')
    env['LD_LIBRARY_PATH'] = ':'.join([str(Path(runtime['operator_library']).parent),str(Path(runtime['core']).parent),
        runtime['root'] + '/lib',str(corex / 'lib64'),str(corex / 'lib'),runtime['infini_root'] + '/lib','/usr/local/lib'])
    env['LD_PRELOAD'] = ':'.join(runtime['preloads'])
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Provide a command after --.')
    runtime = json.loads((args.build / 'runtime.json').read_text())
    os.execvpe(command[0],command,environment(runtime,args.gpu))


if __name__ == '__main__':
    main()
