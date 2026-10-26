#!/usr/bin/env python3
"""Compile one fresh OM and check the actual native tiling bytes (requires ATC)."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import run_fused_multi_conv2d as runner

TAIL_FIELDS = ('blockDim', 'fixedShift1', 'fixedShift2', 'hasBias1', 'hasBias2', 'relu1', 'relu2')
# The second-layer L0C stride adds one uint32 after l0cStride; keep frozen models readable.
TILING_FIELD_COUNTS = {312: 78, 316: 79}


def check_tiling(spec, record):
    if record.get('optype') != 'FusedMultiConv2d':
        raise ValueError('capture does not describe FusedMultiConv2d')
    attrs = record.get('attrs') or []
    names = [attr.get('name') for attr in attrs]
    if len(names) < 13 or names[12] != 'ascendc_op_para_size':
        raise ValueError('unsupported SDK attribute layout: missing reserved slot 12')
    public = ['bias_mask', 'dtype', 'strides', 'pads', 'dilations', 'groups',
              'data_format', 'offset_x', 'round_mode', 'pad_mode', 'enable_hf32', 'enable_relu']
    if names[:12] != public or (len(names) > 13 and names[13] != 'fixed_shift_value'):
        raise ValueError('unsupported operator attribute layout: rebuild the package for the current prototype')
    for attr in attrs:
        if attr['name'] in ('data_format', 'round_mode', 'pad_mode'):
            if attr.get('dtype') not in ('list_int', 'list_int64') or attr.get('value') != [0, 0]:
                raise ValueError('unsupported enum attribute contract for %s: expected ListInt [0,0]; rebuild package and OM' % attr['name'])
    # The capture records Python-side inputs. GE appends registered private defaults
    # inside native tiling, so an omitted fixed_shift_value need not appear here.
    result = record.get('result')
    if not isinstance(result, dict):
        raise ValueError('native tiling did not return a result object')
    try:
        data = bytes.fromhex(result['tiling_data'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('native tiling_data must be a hexadecimal byte string') from exc
    if len(data) not in TILING_FIELD_COUNTS:
        raise ValueError('unsupported tiling ABI: expected 312 or 316 bytes, received %d' % len(data))
    words = struct.unpack('<%dI' % TILING_FIELD_COUNTS[len(data)], data)
    geometry = tuple(spec[key] for key in ('n', 'ci', 'hi', 'wi', 'cout1', 'cout2'))
    if words[:6] != geometry:
        raise ValueError('native tiling belongs to a different case geometry')
    geometry_fields = ('kh','kw','kh2','kw2','s1','s2','ph1','pw1','ph2','pw2')
    if all(key in spec for key in geometry_fields) and words[6:16] != tuple(spec[key] for key in geometry_fields):
        raise ValueError('native tiling geometry differs from case kernel/stride/padding')
    tail = dict(zip(TAIL_FIELDS, words[-7:]))
    stage_offset = 54 if len(words) == 79 else 53
    tail.update(l0cStride=words[52], l0cStride2=words[53] if len(words) == 79 else 0,
                w1Stage=words[stage_offset], w2Stage=words[stage_offset + 1])
    expected = {'hasBias1': int(bool(spec['bias'])),
                'hasBias2': int(bool(spec['bias'])), 'relu1': int(bool(spec['relu1'])),
                'relu2': int(bool(spec['relu2']))}
    if spec['dtype_mode'] == 0:
        expected.update(fixedShift1=spec['shift1'], fixedShift2=spec['shift2'])
    elif spec['dtype_mode'] == 3:
        expected.update(fixedShift1=spec['shift1'])
    for key, value in expected.items():
        if tail[key] != value:
            raise ValueError('native tiling %s=%d, expected %d; rebuild/install the corrected operator package'
                             % (key, tail[key], value))
    if tail['blockDim'] <= 0 or tail['blockDim'] != result.get('block_dim'):
        raise ValueError('native tiling blockDim differs from launch block_dim')
    return {'fields': tail, 'tiling_data_sha256': hashlib.sha256(data).hexdigest(),
            'tiling_data_bytes': len(data)}


def check_capture(spec, directory):
    reports = []
    for path in sorted(Path(directory).glob('tiling.*.jsonl')):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get('event') == 'tiling' and record.get('optype') == 'FusedMultiConv2d':
                reports.append(check_tiling(spec, record))
    if not reports:
        raise ValueError('no native FusedMultiConv2d tiling was captured; verification is incomplete')
    if len({item['tiling_data_sha256'] for item in reports}) != 1:
        raise ValueError('multiple different tiling results captured; use one fresh case per verification')
    return reports[0]


@contextlib.contextmanager
def compilation_environment(capture, cache):
    values = {'FC2D_ATC_TILING_CAPTURE': str(capture), 'ASCEND_CACHE_PATH': str(cache),
              'PYTHONPATH': os.pathsep.join([str(ROOT / 'tools/atc_tiling_capture'), str(ROOT),
                                           os.environ.get('PYTHONPATH', '')])}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compile_and_verify(case, output, soc):
    source = Path(case).resolve()
    spec = fc2d.read_spec(str(source))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / 'case.bin')
    paths = {'bin': str(output / 'case.bin'), 'onnx': str(output / 'model.onnx'),
             'omdir': str(output / 'om')}
    capture, cache = output / 'tiling_capture', output / 'cache'
    capture.mkdir()
    cache.mkdir()
    try:
        with compilation_environment(capture, cache):
            fc2d.do_om({'name': 'FusedMultiConv2d'}, paths, soc)
        report = check_capture(spec, capture)
        models = list((output / 'om').glob('*.om'))
        if len(models) != 1:
            raise ValueError('expected exactly one newly compiled OM')
        binding = runner.validate_model_binding(paths['bin'], str(models[0]))
        report.update(verified=True, case_sha256=binding['case_sha256'],
                      model_sha256=binding['model_sha256'], soc=soc)
        (output / 'tiling_verification.json').write_text(json.dumps(report, indent=2) + '\n')
        return report
    except BaseException:
        # This directory belongs to this fresh compilation. A known-bad or
        # unverified model must not retain a binding accepted by push/runner.
        for binding in (output / 'om').glob('*.om.fc2d.json'):
            binding.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', help='an ABI 8 case.bin; copied into the new output directory')
    parser.add_argument('--soc', required=True, help='the actual ATC target SoC')
    parser.add_argument('--out-dir', required=True, help='a new directory; existing directories are rejected')
    args = parser.parse_args()
    try:
        report = compile_and_verify(args.case, args.out_dir, args.soc)
    except (OSError, ValueError) as exc:
        print('[FAIL] %s' % exc)
        return 1
    print('[PASS] Native ATC tiling verified: ' + json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
