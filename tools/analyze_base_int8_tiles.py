#!/usr/bin/env python3
"""Compare captured base_int8 output by conv2 M tile; standard library only."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def read_reference(path):
    raw = Path(path).read_bytes()
    if len(raw) < 176:
        raise ValueError('case file is shorter than its header')
    magic, version, count, _, _, elements = struct.unpack_from('<8sIIQQQ', raw)
    if magic != b'FC2DCASE' or version != 6:
        raise ValueError('expected FC2DCASE version 6')
    spec = struct.unpack_from('<32i', raw, 40)
    expected = (1, 32, 288, 112, 64, 96, 3, 3, 1, 2, 1, 1, 1, 1)
    if spec[:14] != expected or spec[14:18] != (1, 0, 1, 1):
        raise ValueError('this probe requires base_int8: int8, no bias, both ReLUs')
    if elements != 96 * 144 * 56:
        raise ValueError('unexpected output element count')
    offset = 176
    reference = None
    for _ in range(count):
        if offset + 64 > len(raw):
            raise ValueError('truncated tensor descriptor')
        name, dtype, ndim, *tail = struct.unpack_from('<16sII4qQ', raw, offset)
        offset += 64
        data = raw[offset:offset + tail[4]]
        if len(data) != tail[4]:
            raise ValueError('truncated tensor data')
        offset += tail[4]
        if name.rstrip(b'\0') == b'y_expect':
            if dtype != 2 or ndim != 4 or tuple(tail[:4]) != (1, 96, 144, 56):
                raise ValueError('unexpected reference dtype or shape')
            reference = data
    if offset != len(raw) or reference is None or len(reference) != elements:
        raise ValueError('invalid case payload or missing y_expect')
    return reference


def analyze(reference, actual, second=None):
    if len(reference) != 96 * 144 * 56:
        raise ValueError('reference must contain exactly 774144 bytes')
    if len(actual) != len(reference):
        raise ValueError('actual output must contain exactly 774144 bytes')
    if second is not None and len(second) != len(actual):
        raise ValueError('second output has a different size')
    channels, height, width = 96, 144, 56
    plane, band_size, tile_size = height * width, 18 * width, 9 * width
    mismatches = sum(a != b for a, b in zip(actual, reference))
    row_mismatches = [0] * height
    first_mismatch = None
    for i, (got, want) in enumerate(zip(actual, reference)):
        if got != want:
            row_mismatches[(i % plane) // width] += 1
            if first_mismatch is None:
                first_mismatch = i
    hypotheses = []
    for target_q in range(2):
        for source_q in range(2):
            total = matches = informative = informative_matches = 0
            for channel in range(channels):
                for band in range(8):
                    origin = channel * plane + band * band_size
                    target = origin + target_q * tile_size
                    candidate = origin + source_q * tile_size
                    for i in range(tile_size):
                        got = actual[target + i]
                        match = got == reference[candidate + i]
                        total += 1
                        matches += match
                        # Fix the mask to the target tile's intended nonzero points.
                        # This prevents shared ReLU zeros from dominating the comparison.
                        if reference[target + i] != 0:
                            informative += 1
                            informative_matches += match
            hypotheses.append({
                'actual_tile': target_q,
                'reference_tile': source_q,
                'matches': matches,
                'elements': total,
                'match_percent': round(100 * matches / total, 6),
                'target_reference_nonzero_elements': informative,
                'matches_on_target_reference_nonzero': informative_matches,
                'nonzero_mask_match_percent': round(100 * informative_matches / informative, 6) if informative else None,
            })
    copied_tiles = [
        {'actual_tile': h['actual_tile'], 'matches_reference_tile': h['reference_tile']}
        for h in hypotheses
        if mismatches and h['actual_tile'] != h['reference_tile'] and h['matches'] == h['elements']
    ]
    result = {
        'geometry': {'channels': channels, 'height': height, 'width': width,
                     'band_rows': 18, 'conv2_tile_elements': [504, 504]},
        'elements': len(actual), 'mismatches': mismatches,
        'first_mismatch_index': first_mismatch,
        'row_mismatches': row_mismatches,
        'actual_sha256': hashlib.sha256(actual).hexdigest(),
        'reference_sha256': hashlib.sha256(reference).hexdigest(),
        'tile_comparisons': hypotheses,
        'exact_other_tile_matches': copied_tiles,
        'interpretation_limit': 'An exact match to another tile supports overwrite/misrouting; absence of that match does not exclude a pipeline hazard.',
    }
    if second is not None:
        result['repeat_comparison'] = {
            'different_elements': sum(a != b for a, b in zip(actual, second)),
            'second_sha256': hashlib.sha256(second).hexdigest(),
            'interpretation_limit': 'Identical runs do not exclude a deterministic pipeline hazard.',
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', help='the case.bin used to generate and run the OM')
    parser.add_argument('actual', help='raw int8 NCHW output captured with FC2D_DUMP')
    parser.add_argument('--second', help='optional second raw output from the same OM and case')
    args = parser.parse_args()
    try:
        result = analyze(read_reference(args.case), Path(args.actual).read_bytes(),
                         Path(args.second).read_bytes() if args.second else None)
    except (OSError, ValueError, struct.error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
