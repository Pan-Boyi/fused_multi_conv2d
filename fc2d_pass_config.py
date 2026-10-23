"""Temporarily unregister stage 2 while compiling its ExtendConv2D reference."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import tempfile
import threading

from fc2d_fusion import PASS_NAME

_LOCK_FD = None
_CONFIG_SUFFIX = Path('lib64/plugin/opskernel/fusion_pass/config/support_fusion_pass.json')


def current_lock_fd():
    """Keep the registry lock alive in ATC if the parent process is killed."""
    return _LOCK_FD


def resolve_config(path=None, required=False):
    """Resolve one active CANN registry; SDK aliases must identify the same file."""
    explicit = path or os.environ.get('FC2D_SUPPORT_FUSION_PASS_JSON')
    if explicit:
        result = Path(explicit).expanduser().resolve(strict=True)
        if not result.is_file():
            raise ValueError('fusion pass configuration is not a file: %s' % result)
        return result
    roots = [Path(value) for key in ('ASCEND_OPP_PATH', 'ASCEND_HOME_PATH')
             if (value := os.environ.get(key))]
    atc = shutil.which('atc')
    if atc:
        roots.append(Path(atc).resolve().parent.parent)
    candidates = {candidate.resolve() for root in roots
                  for candidate in (root / _CONFIG_SUFFIX, root / 'opp' / _CONFIG_SUFFIX)
                  if candidate.is_file()}
    if len(candidates) > 1:
        raise ValueError('multiple CANN fusion registries found; select --fusion-pass-config explicitly: %s'
                         % ', '.join(map(str, sorted(candidates))))
    if candidates:
        return candidates.pop()
    if required:
        raise ValueError('support_fusion_pass.json not found in the active CANN; use --fusion-pass-config PATH')
    return None


def _disabled_payload(original):
    """Remove only the exact stage-2 token from architecture pass lists."""
    document = json.loads(original)
    if not isinstance(document, dict) or any(not isinstance(value, str) for value in document.values()):
        raise ValueError('expected architecture -> comma-separated pass lists in support_fusion_pass.json')
    names = {token.strip() for value in document.values() for token in value.split(',')
             if token.strip().casefold() == PASS_NAME.casefold()}
    if len(names) != 1:
        raise ValueError('expected one spelling of %s in support_fusion_pass.json, found %s' % (PASS_NAME, sorted(names)))
    removed = names.pop()
    for architecture, value in document.items():
        document[architecture] = ','.join(token for token in value.split(',') if token.strip() != removed)
    return (json.dumps(document, indent=2, ensure_ascii=False) + '\n').encode(), removed


def _digest(data):
    return hashlib.sha256(data).hexdigest()


@contextlib.contextmanager
def _blocked_signals():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path, data, mode, owner=None):
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            if owner is not None:
                current = os.fstat(stream.fileno())
                if (current.st_uid, current.st_gid) != owner:
                    os.fchown(stream.fileno(), *owner)
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FusionPassRegistry:
    """Serialize compilation and restore a durable, byte-exact registry backup."""

    def __init__(self, path, state_dir=None):
        self.path = Path(path).resolve(strict=True)
        root = (state_dir or os.environ.get('FC2D_FUSION_REGISTRY_STATE') or
                Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) /
                'fused_multi_conv2d/fusion_registry')
        self.directory = Path(root) / _digest(os.fsencode(self.path))
        self.journal = self.directory / 'transaction.json'
        self.backup = self.directory / 'original.json'
        self.lock = None
        self.locked = False
        self.handlers = {}

    def __enter__(self):
        global _LOCK_FD
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.directory / 'compile.lock').open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX)
            self.locked = True
            _LOCK_FD = self.lock.fileno()
            if threading.current_thread() is threading.main_thread():
                for number in (signal.SIGINT, signal.SIGTERM):
                    self.handlers[number] = signal.signal(number, self._interrupt)
            self._recover()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @staticmethod
    def _interrupt(number, frame):
        raise KeyboardInterrupt('compilation interrupted by signal %d' % number)

    def __exit__(self, exception_type, exception, traceback):
        global _LOCK_FD
        try:
            if self.locked:
                with _blocked_signals():
                    self._recover()
        finally:
            for number, handler in self.handlers.items():
                signal.signal(number, handler)
            self.handlers.clear()
            if self.locked:
                _LOCK_FD = None
            if self.lock is not None:
                self.lock.close()
                self.lock = None
            self.locked = False

    def _restore_directory_mode(self, record):
        expected = record['directory_mode']
        actual = stat.S_IMODE(self.path.parent.stat().st_mode)
        if actual not in (expected, expected | stat.S_IWUSR):
            raise RuntimeError('CANN directory permissions changed externally; backup retained: %s' % self.directory)
        if actual != expected:
            os.chmod(self.path.parent, expected)

    def _replace_config(self, payload, record):
        self._restore_directory_mode(record)
        mode = record['directory_mode']
        try:
            if not os.access(self.path.parent, os.W_OK):
                os.chmod(self.path.parent, mode | stat.S_IWUSR)
            _atomic_write(self.path, payload, record['file_mode'], (record['uid'], record['gid']))
        finally:
            self._restore_directory_mode(record)

    def _recover(self):
        if not self.journal.exists():
            return
        record = json.loads(self.journal.read_text())
        original = self.backup.read_bytes()
        if record['path'] != str(self.path) or _digest(original) != record['original_sha256']:
            raise RuntimeError('invalid fusion registry recovery data: %s' % self.directory)
        current = _digest(self.path.read_bytes())
        if current not in (record['original_sha256'], record['disabled_sha256']):
            raise RuntimeError('fusion registry changed externally; refusing to overwrite it; backup: %s' % self.backup)
        try:
            if current != record['original_sha256']:
                self._replace_config(original, record)
            self._restore_directory_mode(record)
        except OSError as exc:
            raise RuntimeError('failed to restore CANN fusion registry; backup retained at %s: %s' % (self.backup, exc)) from exc
        self.journal.unlink()
        _sync_directory(self.directory)

    @contextlib.contextmanager
    def stage2_disabled(self):
        original = self.path.read_bytes()
        disabled, name = _disabled_payload(original)
        metadata = self.path.stat()
        record = dict(path=str(self.path), original_sha256=_digest(original), disabled_sha256=_digest(disabled),
                      file_mode=stat.S_IMODE(metadata.st_mode), directory_mode=stat.S_IMODE(self.path.parent.stat().st_mode),
                      uid=metadata.st_uid, gid=metadata.st_gid)
        proof = dict(config_file=str(self.path), removed_pass=name, original_sha256=record['original_sha256'],
                     disabled_sha256=record['disabled_sha256'], backup_file=str(self.backup), restored=False)
        try:
            with _blocked_signals():
                _atomic_write(self.backup, original, 0o600)
                _atomic_write(self.journal, json.dumps(record, indent=2).encode(), 0o600)
                self._replace_config(disabled, record)
            print('  stage-2 registry entry temporarily removed: %s (%s)' % (name, self.path), flush=True)
            yield proof
        finally:
            with _blocked_signals():
                self._recover()
                proof['restored'] = True
                proof['restored_sha256'] = _digest(self.path.read_bytes())
            print('  CANN fusion registry restored: %s' % self.path, flush=True)


@contextlib.contextmanager
def registry_session(path=None, required=False, state_dir=None):
    """All harness ATC calls share the lock; recover before starting a new case."""
    configuration = resolve_config(path, required)
    if configuration is None:
        yield None
        return
    with FusionPassRegistry(configuration, state_dir) as registry:
        if required:
            _disabled_payload(registry.path.read_bytes())
        yield registry
