# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exercise temporary installed-pass changes using isolated CANN fixtures."""
import contextlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d_fusion as fusion
import fc2d_pass_config as config


class PassRegistry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.sdk = self.work / 'sdk'
        self.suffix = Path('lib64/plugin/opskernel/fusion_pass/config/support_fusion_pass.json')
        self.registry = self.sdk / 'opp' / self.suffix
        self.registry.parent.mkdir(parents=True)
        self.original = ('{\n  "3510": "%s,OtherPass",\n'
                         '  "5102": "%s,%s,OtherPass,%sExtra"\n}\n' %
                         (fusion.FIRST_PASS_NAME, fusion.FIRST_PASS_NAME,
                          fusion.PASS_NAME, fusion.PASS_NAME)).encode()
        self.registry.write_bytes(self.original)
        self.registry.chmod(0o640)
        self.atc = self.sdk / 'compiler/bin/atc'
        self.atc.parent.mkdir(parents=True)
        self.atc.write_text('#!/bin/sh\nexit 0\n')
        self.atc.chmod(0o755)
        self.env = dict(os.environ, FC2D_SUPPORT_FUSION_PASS_JSON=str(self.registry),
                        FC2D_FUSION_REGISTRY_STATE=str(self.work / 'state'),
                        ASCEND_HOME_PATH=str(self.sdk),
                        PATH=str(self.atc.parent) + os.pathsep + os.environ['PATH'],
                        PYTHONPATH=str(ROOT))

    @contextlib.contextmanager
    def session(self):
        with mock.patch.dict(os.environ, self.env, clear=True):
            with config.registry_session(required=True) as registry:
                yield registry

    def assert_restored(self, mode=0o640):
        self.assertEqual(self.registry.read_bytes(), self.original)
        self.assertEqual(stat.S_IMODE(self.registry.stat().st_mode), mode)

    def assert_disabled(self):
        document = json.loads(self.registry.read_bytes())
        self.assertEqual(document['3510'], fusion.FIRST_PASS_NAME + ',OtherPass')
        self.assertEqual(document['5102'].split(','),
                         [fusion.FIRST_PASS_NAME, 'OtherPass', fusion.PASS_NAME + 'Extra'])

    def child(self, body):
        body = body.replace('print("READY", flush=True)',
                            'print("READY", flush=True, file=sys.__stdout__)')
        script = ('import contextlib, sys, time\nfrom fc2d_pass_config import registry_session\n'
                  'with contextlib.redirect_stdout(sys.stderr):\n'
                  '    with registry_session(required=True) as registry:\n' +
                  '\n'.join('        ' + line for line in body.splitlines()) + '\n')
        process = subprocess.Popen([sys.executable, '-c', script], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop_child, process)
        return process

    @staticmethod
    def stop_child(process):
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    def wait_ready(self, process):
        ready, _, _ = select.select([process.stdout], [], [], 5)
        if not ready:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            self.fail('registry child did not enter context: %s %s' % (stdout, stderr))
        self.assertEqual(process.stdout.readline().strip(), 'READY')

    def compiler_driver(self):
        pid_file = self.work / 'compiler.pid'
        finished = self.work / 'compiler-finished.txt'
        compiler = self.work / 'compiler.py'
        compiler.write_text(
            'import json, os, signal, sys, time\nfrom pathlib import Path\n'
            'signal.signal(signal.SIGTERM, lambda *_: sys.exit(7))\n'
            'Path(%r).write_text(str(os.getpid()))\n'
            'try:\n    time.sleep(60)\n'
            'finally:\n'
            '    passes = json.loads(Path(%r).read_bytes())["5102"].split(",")\n'
            '    Path(%r).write_text("disabled" if %r not in passes else "restored too early")\n'
            % (str(pid_file), str(self.registry), str(finished), fusion.PASS_NAME))
        driver = self.child('import fc2d\nwith registry.stage2_disabled():\n'
                            '    fc2d.run([sys.executable, %r], log_path=%r)' %
                            (str(compiler), str(self.work / 'atc.log')))
        deadline = time.monotonic() + 5
        while not pid_file.exists() and driver.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pid_file.exists(), 'compiler did not start')
        pid = int(pid_file.read_text())
        self.addCleanup(self.terminate_pid, pid)
        return driver, pid, finished

    @staticmethod
    def terminate_pid(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.01)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(0.05)

    def test_exact_pass_removal_restores_bytes_permissions_and_symlink_target(self):
        alias = self.work / 'support_fusion_pass.json'
        alias.symlink_to(self.registry)
        self.env['FC2D_SUPPORT_FUSION_PASS_JSON'] = str(alias)
        with self.session() as registry:
            self.assert_restored()
            with registry.stage2_disabled():
                self.assert_disabled()
                self.assertTrue(alias.is_symlink())
                self.assertEqual(stat.S_IMODE(self.registry.stat().st_mode), 0o640)
            self.assert_restored()
        self.assert_restored()

    def test_owned_read_only_sdk_file_and_directory_modes_are_restored(self):
        self.registry.chmod(0o440)
        self.registry.parent.chmod(0o550)
        with self.session() as registry:
            with registry.stage2_disabled():
                self.assert_disabled()
                self.assertEqual(stat.S_IMODE(self.registry.stat().st_mode), 0o440)
            self.assert_restored(0o440)
        self.assert_restored(0o440)
        self.assertEqual(stat.S_IMODE(self.registry.parent.stat().st_mode), 0o550)

    def test_all_architecture_entries_remove_the_pass_with_case_insensitive_spelling(self):
        document = json.loads(self.original)
        alias = fusion.PASS_NAME.replace('2D', '2d')
        document['5102'] = document['5102'].replace(fusion.PASS_NAME + ',', alias + ',')
        document['950'] = 'OtherPass,' + alias
        self.original = (json.dumps(document, indent=3) + '\n').encode()
        self.registry.write_bytes(self.original)
        with self.session() as registry:
            with registry.stage2_disabled():
                modified = json.loads(self.registry.read_bytes())
                self.assertEqual(modified['950'], 'OtherPass')
                self.assert_disabled()
        self.assert_restored()

    def test_exception_keyboard_interrupt_and_system_exit_restore_original(self):
        for failure in (RuntimeError('ATC failed'), KeyboardInterrupt(), SystemExit(9)):
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)):
                    with self.session() as registry:
                        with registry.stage2_disabled():
                            self.assert_disabled()
                            raise failure
                self.assert_restored()

    def test_invalid_json_and_missing_pass_do_not_modify_registry(self):
        for raw in (b'{invalid JSON', b'{"5102":"OtherPass"}\n', b'[]\n'):
            with self.subTest(raw=raw):
                self.registry.write_bytes(raw)
                with self.assertRaises((ValueError, RuntimeError)):
                    with self.session() as registry:
                        with registry.stage2_disabled():
                            self.fail('invalid registry accepted')
                self.assertEqual(self.registry.read_bytes(), raw)
                self.assertEqual(stat.S_IMODE(self.registry.stat().st_mode), 0o640)

    def test_explicit_missing_config_does_not_fall_back_to_another_sdk(self):
        self.env['FC2D_SUPPORT_FUSION_PASS_JSON'] = str(self.work / 'missing.json')
        with self.assertRaises((ValueError, RuntimeError, OSError)):
            with self.session():
                self.fail('explicit missing registry accepted')
        self.assert_restored()

    def test_active_sdk_aliases_are_deduplicated_and_distinct_sdks_require_selection(self):
        target = self.sdk / 'aarch64-linux' / self.suffix
        target.parent.mkdir(parents=True)
        self.registry.rename(target)
        self.registry.symlink_to(target)
        actual_atc = self.sdk / 'aarch64-linux/bin/atc'
        actual_atc.parent.mkdir(parents=True)
        self.atc.rename(actual_atc)
        self.atc.symlink_to(actual_atc)
        self.env.pop('FC2D_SUPPORT_FUSION_PASS_JSON')
        self.env['ASCEND_OPP_PATH'] = str(self.sdk / 'opp')
        with self.session() as registry:
            self.assertEqual(registry.path, self.registry.resolve())
        other = self.work / 'other-opp' / self.suffix
        other.parent.mkdir(parents=True)
        other.write_bytes(self.original)
        self.env['ASCEND_OPP_PATH'] = str(self.work / 'other-opp')
        with self.assertRaisesRegex(ValueError, 'multiple'):
            with self.session():
                self.fail('ambiguous active CANN registries accepted')
        self.env['FC2D_SUPPORT_FUSION_PASS_JSON'] = str(self.registry)
        with self.session() as registry:
            self.assertEqual(registry.path, self.registry.resolve())
        self.assert_restored()
        self.assertEqual(other.read_bytes(), self.original)

    def test_sigint_and_sigterm_restore_before_process_exits(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                process = self.child('with registry.stage2_disabled():\n'
                                     '    print("READY", flush=True)\n    time.sleep(60)')
                self.wait_ready(process)
                self.assert_disabled()
                process.send_signal(signum)
                process.communicate(timeout=5)
                self.assertNotEqual(process.returncode, 0)
                self.assert_restored()

    def test_killed_process_is_recovered_before_next_fused_session(self):
        process = self.child('with registry.stage2_disabled():\n'
                             '    print("READY", flush=True)\n    time.sleep(60)')
        self.wait_ready(process)
        process.kill()
        process.communicate(timeout=5)
        self.assert_disabled()
        with self.session():
            self.assert_restored()
        self.assert_restored()

    def test_recovery_does_not_overwrite_unrelated_external_edit(self):
        process = self.child('with registry.stage2_disabled():\n'
                             '    print("READY", flush=True)\n    time.sleep(60)')
        self.wait_ready(process)
        process.kill()
        process.communicate(timeout=5)
        external = self.registry.read_bytes().replace(b'OtherPass', b'ExternalPass')
        self.registry.write_bytes(external)
        with self.assertRaises((ValueError, RuntimeError)):
            with self.session():
                self.fail('conflicting interrupted transaction accepted')
        self.assertEqual(self.registry.read_bytes(), external)

    def test_fused_session_cannot_overlap_reference_compilation(self):
        reference = self.child('with registry.stage2_disabled():\n'
                               '    print("READY", flush=True)\n    time.sleep(60)')
        self.wait_ready(reference)
        fused = self.child('print("READY", flush=True)')
        ready, _, _ = select.select([fused.stdout], [], [], 0.2)
        if ready:
            stdout, stderr = fused.communicate(timeout=5)
            self.assertNotEqual(fused.returncode, 0, stdout + stderr)
            self.assertNotIn('READY', stdout, 'fused compilation entered while Stage2 was disabled')
        self.assert_disabled()
        reference.send_signal(signal.SIGTERM)
        reference.communicate(timeout=5)
        if not ready:
            self.wait_ready(fused)
            stdout, stderr = fused.communicate(timeout=5)
            self.assertEqual(fused.returncode, 0, stdout + stderr)
        self.assert_restored()

    def test_killed_driver_keeps_compiler_lock_until_compiler_exits(self):
        driver, pid, finished = self.compiler_driver()
        driver.kill()
        driver.communicate(timeout=5)
        self.assert_disabled()
        contender = self.child('print("READY", flush=True)')
        ready, _, _ = select.select([contender.stdout], [], [], 0.2)
        self.assertFalse(ready, 'next compilation entered before surviving ATC exited')
        self.assert_disabled()
        os.kill(pid, signal.SIGTERM)
        self.wait_ready(contender)
        stdout, stderr = contender.communicate(timeout=5)
        self.assertEqual(contender.returncode, 0, stdout + stderr)
        self.assert_restored()
        self.assertEqual(finished.read_text(), 'disabled')

    def test_sigterm_stops_and_waits_for_compiler_before_restoring_registry(self):
        driver, pid, finished = self.compiler_driver()
        self.assert_disabled()
        driver.send_signal(signal.SIGTERM)
        driver.communicate(timeout=5)
        self.assertNotEqual(driver.returncode, 0)
        self.assertEqual(finished.read_text(), 'disabled')
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assert_restored()


if __name__ == '__main__':
    unittest.main()
