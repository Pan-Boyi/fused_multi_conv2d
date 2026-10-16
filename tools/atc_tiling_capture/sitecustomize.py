"""Opt-in ATC diagnostics; leave the SDK files and tiling results unchanged."""
import importlib
import json
import os
from pathlib import Path
import sys


def install():
    directory = os.environ.get('FC2D_ATC_TILING_CAPTURE')
    if not directory:
        return
    destination = Path(directory) / ('tiling.%d.jsonl' % os.getpid())

    def write(record):
        with destination.open('a') as stream:
            stream.write(json.dumps(record, default=lambda value:
                                   value.hex() if isinstance(value, (bytes, bytearray))
                                   else str(value)) + '\n')

    for name in ('asc_op_compile_base.asc_op_compiler.op_tiling', 'tbe.common.utils.op_tiling'):
        try:
            module = importlib.import_module(name)
            original = module.do_op_tiling

            def capture(*args, _original=original, _module=name, **kwargs):
                optype = args[0] if args else kwargs.get('optype')
                result = _original(*args, **kwargs)
                if optype == 'FusedMultiConv2d':
                    write({'event': 'tiling', 'module': _module, 'optype': optype,
                           'attrs': kwargs.get('attrs', args[6] if len(args) > 6 else None),
                           'result': result})
                return result

            # Some SDK adapters import this function by name before site startup.
            for loaded in list(sys.modules.values()):
                if loaded is not None and getattr(loaded, 'do_op_tiling', None) is original:
                    loaded.do_op_tiling = capture
            module.do_op_tiling = capture
        except Exception as exc:
            write({'event': 'capture-unavailable', 'module': name, 'error': str(exc)})


install()
