"""Build the measured R5 model and support libraries against its pinned SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--infini-root', type=Path, required=True)
    parser.add_argument('--core-source', type=Path, required=True)
    parser.add_argument('--corex', type=Path, required=True)
    parser.add_argument('--spdlog-include', type=Path, required=True)
    parser.add_argument('--pybind-include', type=Path, required=True)
    parser.add_argument('--runtime-config', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Use a new output directory.')
    manifest = json.loads((HERE / 'SOURCE_MANIFEST.json').read_text())
    for row in manifest['files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Source differs from R5: ' + row['path'])
    runtime = json.loads(args.runtime_config.read_text())
    for path, digest in runtime['sha256'].items():
        if sha(path) != digest:
            raise ValueError('Pinned dependency changed: ' + path)
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    lib = out / 'lib'
    lib.mkdir()
    includes = out / 'include'
    shutil.copytree(args.infini_root / 'include', includes)
    shutil.copy2(HERE / 'include/infinicore/nn/rope.hpp', includes / 'infinicore/nn/rope.hpp')
    shutil.copytree(ROOT / 'python', out / 'python', ignore=shutil.ignore_patterns('*.so', '__pycache__', '*.pyc'))
    src = HERE / 'src'
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(('IV_', 'INFINILM_', 'INFINICORE_')) or key in ('LD_PRELOAD', 'LD_AUDIT'):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES='', IX_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    env['PATH'] = str(args.corex / 'bin') + ':/usr/local/bin:/usr/bin:/bin'
    env['LD_LIBRARY_PATH'] = ':'.join(map(str, (args.corex / 'lib64', args.infini_root / 'lib', Path('/usr/local/lib'))))
    common = ['-m64', '-fPIC', '-std=c++17', '-I' + str(args.spdlog_include),
              '-I' + str(includes), '-I' + str(ROOT / 'include'), '-I' + str(src),
              '-isystem', str(args.pybind_include), '-isystem', sysconfig.get_path('include')]
    cuda = [str(args.corex / 'bin/clang++'), '-O3', '-x', 'ivcore', '-std=c++17',
            '--cuda-gpu-arch=ivcore20', '-fPIC', '-c', '-I' + str(args.infini_root / 'include'),
            '-I' + str(args.core_source / 'include'), '-I' + str(args.core_source / 'src'),
            '-I' + str(args.spdlog_include), '-I' + str(src)]
    links = ['-L' + str(args.infini_root / 'lib'), '-linfinicore_cpp_api', '-linfinirt',
             '-L' + str(args.corex / 'lib64'), '-lcudart']
    report = dict(status='building',source_manifest_sha256=sha(HERE / 'SOURCE_MANIFEST.json'),
                  full_model_sources_compiled=True,full_infinicore_rebuild=False,
                  dependency_runtime=runtime,stages=[],started=time.time())

    def save():
        (out / 'build.json').write_text(json.dumps(report, indent=2))

    def run(name, command):
        row = dict(name=name,command=list(map(str,command)),started=time.time())
        report['stages'].append(row)
        save()
        with (out / 'logs' / (name + '.log')).open('x') as log:
            process = subprocess.Popen(row['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid'] = process.pid
            save()
            try:
                rc = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                rc = 124
        row.update(exit_code=rc,ended=time.time())
        save()
        if rc:
            raise RuntimeError(name + ' failed; see its build log.')
        print('R5_BUILD_STAGE',name,flush=True)

    try:
        specs = [('add','add_contiguous_override.cu','libadd_contiguous.so',False),
                 ('swiglu','swiglu_stride_override.cu','libswiglu_guarded.so',False),
                 ('fusion','add_rms_fusion.cu','libadd_rms_fusion.so',True),
                 ('softmax','softmax_row_bucket_override.cu','libsoftmax_row_bucket.so',False)]
        for name, source, library, define in specs:
            obj = out / (name + '.o')
            run(name + '_compile',cuda + (['-DENABLE_ILUVATAR_API'] if define else []) + [src / source,'-o',obj])
            run(name + '_link',['g++','-shared',obj,*links,'-linfiniop','-ldl','-Wl,-z,defs','-o',lib / library])
        for name, source, library in [('length','length_mirror.cpp','liblength_mirror.so'),
                                      ('chain','chain_fusion_support.cpp','libchain_fusion.so'),
                                      ('last','last_logits_short_scope.cpp','liblast_logits.so')]:
            run(name + '_link',['g++','-O2',*common,'-shared',src / source,
                '-L' + str(args.infini_root / 'lib'),'-linfinicore_cpp_api',
                '-Wl,--version-script=' + str(src / 'logging.map'),'-o',lib / library])
        run('pair_kernel',cuda + ['-DENABLE_ILUVATAR_API',src / 'rope_pair.cu','-o',out / 'pair.o'])
        run('pair_host',['g++','-O3',*common,'-I' + str(args.corex / 'include'),'-c',src / 'rope_pair_host.cpp','-o',out / 'pair_host.o'])
        run('pair_link',['g++','-shared',out / 'pair.o',out / 'pair_host.o',*links,'-o',lib / 'librope_pair.so'])
        objects = []
        for source in sorted(list((ROOT / 'csrc').rglob('*.cpp')) + list((ROOT / 'csrc').rglob('*.cc'))):
            name = source.relative_to(ROOT).as_posix().replace('/','_')
            obj = out / (name + '.o')
            run(name,['g++',*common,'-c',source,'-o',obj])
            objects.append(obj)
        api = out / 'rope_pair_api.o'
        run('pair_api',['g++',*common,'-c',src / 'rope_pair_model_api.cpp','-o',api])
        objects.append(api)
        model = out / 'python/infinilm/lib' / ('_infinilm' + sysconfig.get_config_var('EXT_SUFFIX'))
        model.parent.mkdir(exist_ok=True)
        run('model_link',['g++','-Wl,--version-script=' + str(src / 'logging.map'),*objects,
            lib / 'libadd_rms_fusion.so',lib / 'liblength_mirror.so',lib / 'libchain_fusion.so',
            lib / 'liblast_logits.so',lib / 'librope_pair.so','-shared','-m64','-fPIC',
            '-L' + str(args.infini_root / 'lib'),'-L/usr/local/lib','-linfinicore_cpp_api',
            '-linfiniop','-linfinirt','-linfiniccl','-lpython' + sysconfig.get_python_version(),
            '-lssl','-lcrypto','-o',model])
        run('model_dynamic',['readelf','-d',model])
        runtime.update(root=str(out),model=str(model),infini_root=str(args.infini_root),corex=str(args.corex),
                       preloads=[str(lib / x) for x in ('libsoftmax_row_bucket.so','liblast_logits.so','libchain_fusion.so',
                           'liblength_mirror.so','libadd_rms_fusion.so','libadd_contiguous.so','libswiglu_guarded.so')])
        runtime['sha256'].update({str(p):sha(p) for p in [model,*lib.glob('*.so')]})
        (out / 'runtime.json').write_text(json.dumps(runtime,indent=2))
        report.update(status='passed',model=str(model),model_sha256=sha(model),model_translation_units=len(objects),ended=time.time())
    except BaseException as error:
        report.update(status='failed',error=str(error),ended=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
