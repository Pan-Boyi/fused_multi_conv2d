"""Graph filter ABI checks; no CANN device is needed."""
import contextlib
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fc2d
import fc2d_onnx
import run_fused_multi_conv2d as runner


class FilterContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.tmp.name)
        cls.generator = cls.work / "gen_case"
        subprocess.run(["g++", "-std=c++17", "-O2", str(ROOT / "gen_case.cpp"),
                        "-o", str(cls.generator)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def make_case(self, mode):
        path = self.work / (mode + ".bin")
        subprocess.run([str(self.generator), str(path), "--dtype", mode,
                        "--ci", "32", "--cout1", "64", "--cout2", "2",
                        "--hi", "20", "--wi", "30", "--kh", "1", "--kw", "3",
                        "--kh2", "3", "--kw2", "1", "--s1", "1", "--s2", "2",
                        "--ph1", "0", "--pw1", "1", "--ph2", "1", "--pw2", "0",
                        "--bias", "1", "--shift1", "29" if mode == "a16w8" else "42", "--shift2", "42"], check=True, stdout=subprocess.DEVNULL)
        return path

    def test_four_dtype_paths_use_logical_filter_shapes_and_bytes(self):
        for mode in ("fp16", "int8", "s8f16", "a16w8"):
            with self.subTest(mode=mode):
                path = self.make_case(mode)
                self.assertEqual(struct.unpack_from("<I", path.read_bytes(), 8)[0], 8)
                spec = fc2d.read_spec(str(path))
                tensors, *unused = runner.load_case(str(path))
                plan = fc2d_onnx.tensor_plan(spec)
                self.assertNotIn("kernel_size", plan["attrs"])
                self.assertEqual(list(plan["attrs"]), ['bias_mask', 'dtype', 'strides', 'pads', 'dilations', 'groups', 'data_format', 'offset_x', 'round_mode', 'pad_mode', 'enable_hf32', 'enable_relu', 'fixed_shift_value'])
                slots = {name: (dt, shape) for name, dt, shape in plan["slots"]}
                width = 2 if mode == "fp16" else 1
                for name, dims in (("filter1", [64, 32, 1, 3]),
                                   ("filter2", [2, 64, 3, 1])):
                    self.assertEqual(slots[name][1], dims)
                    self.assertEqual(tensors[name][1], dims)
                    self.assertEqual(len(tensors[name][2]), width * dims[0] * dims[1] * dims[2] * dims[3])
                self.assertEqual(tensors["filter1"][2], tensors["w1_nchw"][2])
                model = fc2d_onnx.build_model(plan, tensors)
                self.assertEqual(len(model.graph.input), 3)
                self.assertEqual(sum(node.op_type == "Conv" for node in model.graph.node), 2)
                initializers = {t.name: t for t in model.graph.initializer}
                scales=[] if mode=='fp16' else ['quant_scale1','quant_scale2' if mode=='int8' else 'dequant_scale2']
                self.assertEqual(set(initializers), {'filter1','filter2',*scales})
                for name in scales:
                    self.assertEqual(initializers[name].data_type,fc2d_onnx.TensorProto.UINT64)
                    self.assertEqual(list(initializers[name].dims),tensors[name][1])
                    self.assertEqual(initializers[name].raw_data,tensors[name][2])
                    self.assertNotIn(name,runner.model_input_slots(spec))
                    self.assertNotIn(name+':',fc2d_onnx.input_shape_arg(plan))
                self.assertEqual(list(initializers["filter2"].dims), [2, 64, 3, 1])
                self.assertEqual(initializers["filter2"].raw_data, tensors["filter2"][2])
                shapes = {value.name: [d.dim_value for d in value.type.tensor_type.shape.dim]
                          for value in model.graph.input}
                self.assertNotIn("filter2", shapes)
                self.assertNotIn("filter2:", fc2d_onnx.input_shape_arg(plan))
                self.assertNotIn("filter1", runner.model_input_slots(spec))
                self.assertEqual(plan["y"][1], [1, 2, 10, 15])

    def test_consumed_scale_constants_require_original_dtype_shape_and_bytes(self):
        for mode in ('int8','s8f16','a16w8'):
            path=self.make_case(mode)
            spec=fc2d.read_spec(str(path));tensors,*_=runner.load_case(str(path))
            plan=fc2d_onnx.tensor_plan(spec)
            for name in ('quant_scale1','quant_scale2' if mode=='int8' else 'dequant_scale2'):
                actual_dt,dims,raw=tensors[name]
                for fault in ('missing','dtype','shape','bytes'):
                    with self.subTest(mode=mode,name=name,fault=fault):
                        changed=dict(tensors)
                        if fault=='missing':del changed[name]
                        elif fault=='dtype':changed[name]=(1,dims,raw)
                        elif fault=='shape':changed[name]=(actual_dt,[dims[0],1],raw)
                        else:changed[name]=(actual_dt,dims,raw[:-1])
                        with self.assertRaises(ValueError):fc2d_onnx.build_model(plan,changed)

    def test_old_packed_case_abi_is_rejected_before_model_compilation_or_execution(self):
        raw = bytearray(self.make_case("int8").read_bytes())
        struct.pack_into("<I", raw, 8, 6)
        path = self.work / "old.bin"
        path.write_bytes(raw)
        for reader in (fc2d.read_spec, runner.load_case):
            with self.subTest(reader=reader.__name__), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    reader(str(path))


if __name__ == "__main__":
    unittest.main()
