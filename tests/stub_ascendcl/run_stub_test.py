#!/usr/bin/env python3
"""Exercise the graph runner without an NPU; the stub copies golden to output."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import run_fused_multi_conv2d as runner


def select_cases(out_root, names, explicit_dirs):
    if explicit_dirs:
        if names:
            raise ValueError('Use either case names or --case-dir, not both')
        directories = [Path(path).resolve() for path in explicit_dirs]
    else:
        candidates = sorted({path.parent.resolve() for path in out_root.rglob('case.bin')})
        if names:
            directories = []
            for name in names:
                matches = [path for path in candidates if path.name == name]
                if len(matches) != 1:
                    raise ValueError('%s: expected one case directory, found %d; use --case-dir' %
                                     (name, len(matches)))
                directories.append(matches[0])
        else:
            directories = candidates
    if not directories:
        raise ValueError('No case directories found; run --steps case,om first')
    selected = []
    for directory in directories:
        case = directory / 'case.bin'
        models = sorted((directory / 'om').glob('*.om'))
        if not case.is_file() or len(models) != 1:
            raise ValueError('%s: requires case.bin and exactly one om/*.om (found %d)' %
                             (directory, len(models)))
        runner.validate_model_binding(str(case), str(models[0]))
        selected.append((directory, case, models[0]))
    return selected


def run_mode(command, env, work, log):
    result = subprocess.run(command, env=env, cwd=str(work), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    return result.returncode == 0 and any(line.startswith('[PASS]')
                                         for line in result.stdout.splitlines())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('names', nargs='*', help='Unambiguous case directory names')
    ap.add_argument('--out-root', type=Path, default=ROOT / 'out')
    ap.add_argument('--case-dir', action='append', default=[], help='Exact case directory; repeatable')
    ap.add_argument('--work-dir', type=Path, default=HERE / '.work')
    args = ap.parse_args(argv)
    try:
        cases = select_cases(args.out_root, args.names, args.case_dir)
        compiler = shutil.which(os.environ.get('CC', 'gcc'))
        if not compiler:
            raise ValueError('A C compiler is required for the ACL stub')
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run([compiler, '-shared', '-fPIC', '-O1', str(HERE / 'stub.c'),
                        '-o', str(work / 'libascendcl.so')], check=True)
        failed = False
        for directory, case, model in cases:
            label = directory.name + '_' + hashlib.sha256(str(directory).encode()).hexdigest()[:10]
            case_work = work / label
            case_work.mkdir(exist_ok=True)
            with (case_work / 'plan.txt').open('w') as stream:
                subprocess.run([sys.executable, str(HERE / 'make_conf.py'), str(case), str(case_work)],
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            for mode in ('name', 'fallback'):
                env = dict(os.environ, FC2D_STUB_CONF=str(case_work / 'stub.conf'),
                           FC2D_STUB_GOLDEN=str(case_work / 'golden.bin'), REPEAT='1', WARMUP='0')
                # Only the test child sees this fake library. The product loader is unchanged.
                for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
                    env[key] = str(work) + os.pathsep + env.get(key, '')
                env.pop('FC2D_STUB_NONAME', None)
                if mode == 'fallback':
                    env['FC2D_STUB_NONAME'] = '1'
                log = case_work / ('run_' + mode + '.log')
                ok = run_mode([sys.executable, str(ROOT / 'run_fused_multi_conv2d.py'), str(case),
                               'FusedMultiConv2d', '0', str(model)], env, work, log)
                print('%s %s %s; log: %s' % (directory, mode, 'PASS' if ok else 'FAIL', log))
                failed |= not ok
        return 1 if failed else 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print('[X] %s' % exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
