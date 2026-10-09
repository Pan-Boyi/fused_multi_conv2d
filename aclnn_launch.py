"""aclnn（两段式）下发路径 —— FusedConv2d

**为什么需要这条路。** `atc --singleop` 的 json **表达不了实例数 > 1 的 DYNAMIC 输入**
（2026-10-09 实测，见下）。要把 filter / bias 做成 list（为了以后扩展到更多层卷积融合），
就必须走 aclnn —— 那一侧有 `aclCreateTensorList`。这个模块是那条路的下发实现。

实测依据（用 ScatterList 和 ForeachNonFiniteCheckAndUnscale 两个现成算子交叉验证）：
  - singleop json 的 desc 键只有 format / shape / type / origin_shape / shape_range，
    `libge_compiler.so` 里没有任何 dynamic 相关的键；
  - 嵌套数组被 json 解析器直接拒：
    `E10032 ... [json.exception.type_error.304] cannot use at() with array`
    —— 解析器对每个 input_desc 元素调 `.at("format")`，元素必须是对象；
  - 平铺 N 项时 FE 按位置拿节点输入去比 op-info 的清单，个数对不上就报
    `EZ3002 ... the inputs in op desc and inputs in op information library are not matched`；
    1 个实例能过只是因为平铺个数恰好等于 IR 输入个数；
  - `dynamic_input_name` 这个键在 atc 侧**被静默忽略**（加与不加，生成的图名逐字节相同）。

**算子侧不需要改一行代码。** `conv/fused_conv2d/op_host/CMakeLists.txt` 本来就写了
`ACLNNTYPE aclnn`，所以 opbuild 会从 OpDef **自动生成** aclnn 两段式接口
（build/autogen/aclnn_fused_conv2d.{h,cpp}）。生成的接口：
  - 参数顺序 = 输入按 IR 序 → 属性按 Attr() 声明序 → out → workspaceSize → executor
  - 只对 4 个必需参数判空，缺席的可选输入**直接传 nullptr**
  - 三个 ListInt 属性传 NULL 时用 OpDef 的缺省值（kernel_size {3,3} / strides {1,2} / pads {1,1}）
  - 四条 dtype 通路作为 support list 编进去，soc 名单是 {"mc62"}
  - 不插 Contiguous / FlattenDims，张量原样透传（这对本算子很关键：filter 是
    FRACTAL_Z 裸字节，摊平会让 host tiling 的逐维校验全不成立）
手写一份 op_api 只会和生成的那份**重名冲突**（实测 ld 报
`multiple definition of aclnnFusedConv2dGetWorkspaceSize`），所以不要手写。

**这条路和原来的 aclopExecuteV2 并存**，用 `--launcher` 选。输入数据、golden、比对逻辑
全部共用，只有「怎么把张量和属性交给算子」这一段不同：

    aclopExecuteV2（原有）                     aclnnFusedConv2d（本模块）
    ---------------------------------------    ----------------------------------------
    需要 .om（atc --singleop 编出来）          **不需要 .om**，直接从算子包找 kernel
    aclCreateTensorDesc + aclCreateDataBuffer  aclCreateTensor（带 strides/offset/storage）
    aclopAttr + aclopSetAttrInt/Bool/ListInt   属性就是函数参数（int64 / bool / aclIntArray*）
    缺席的 optional 传 UNDEFINED desc 占位     缺席的 optional **直接传 nullptr**
    符号在 libascendcl.so                      张量构造在 libnnopbase.so，
                                               两段式接口在算子包的 libcust_opapi.so

**不需要 .om 是实际的简化**：走 aclnn 时 fc2d.py 的 `om` 步骤整步可以跳过，连带
singleop.json 的生成、以及「属性值必须和 .om 完全一致否则报 100024」那一整类坑都不存在
—— aclnn 是按符号直接调的，不做签名匹配。**代价**是 aclnn 没有「板上没装算子包也能靠
.om 跑」这条退路：板子上必须装好带 FusedConv2d 的 mc62 算子包。
"""

import ctypes
import os

ACL_SUCCESS = 0
ACL_FORMAT_ND = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

# 自定义算子包里两段式接口所在的 so。build.sh 编出来的名字就是这个
# （build 日志里的链接目标 libcust_opapi.so）。
OPAPI_SO_NAME = "libcust_opapi.so"
# aclCreateTensor / aclCreateTensorList / aclCreateIntArray 都在这里，
# **libascendcl.so 里一个都没有**（nm -D 实测：libascendcl=0 libnnopbase=1）。
NNOPBASE_SO_NAME = "libnnopbase.so"


class AclnnUnavailable(RuntimeError):
    """aclnn 这条路跑不起来（缺库 / 缺符号 / 算子包没装 op_api）。

    单独一个异常类型，好让调用方能干净地退回 --launcher singleop。
    """


def _die(msg):
    raise AclnnUnavailable(msg)


# ---------------------------------------------------------------- 找库
def find_nnopbase_so(explicit=None):
    if explicit:
        if not os.path.isfile(explicit):
            _die("--nnopbase-so 指的文件不存在: %s" % explicit)
        return explicit
    roots = []
    for env in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        v = os.environ.get(env, "")
        if v:
            roots.append(v)
    cands = [os.path.join(r, "lib64", NNOPBASE_SO_NAME) for r in roots]
    cands.append(NNOPBASE_SO_NAME)   # 交给 ld.so 按 LD_LIBRARY_PATH 找
    for c in cands:
        if c == NNOPBASE_SO_NAME or os.path.isfile(c):
            return c
    _die("找不到 %s。先 source CANN 的 set_env.sh。" % NNOPBASE_SO_NAME)


def find_opapi_so(explicit=None, vendor=None):
    """定位自定义算子包的 opapi so。

    找不到时要说清楚**该怎么办** —— 这是走 aclnn 最容易卡住的一步：算子包装了
    但没装 op_api 那一半，或者装了但 ASCEND_CUSTOM_OPP_PATH 没指对。
    """
    if explicit:
        if not os.path.isfile(explicit):
            _die("--opapi-so 指的文件不存在: %s" % explicit)
        return explicit

    # 自定义算子包装的是 op_api/**lib**/，不是 lib64 —— cmake/variables.cmake:77 用的是
    # `lib`，只有 built-in 那一侧才是 aarch64-linux/lib64（variables.cmake:101）。
    # 两个都找，顺序无关紧要。
    LIBDIRS = ("lib", "lib64")

    cands = []
    env = os.environ.get("FC2D_OPAPI_SO", "")
    if env:
        cands.append(env)
    for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if root:
            cands += [os.path.join(root, "op_api", d, OPAPI_SO_NAME) for d in LIBDIRS]
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if opp:
        vendors = os.path.join(opp, "vendors")
        # vendor 目录名带 `_nn` 后缀：--vendor_name=customize 装到 vendors/customize_nn
        # （cmake/variables.cmake:23 VENDOR_PACKAGE_NAME = ${VENDOR_NAME}_nn）。
        names = []
        if vendor:
            names += [vendor, vendor + "_nn"]
        if os.path.isdir(vendors):
            names += sorted(n for n in os.listdir(vendors) if n != "config.ini")
        for n in names:
            if n:
                cands += [os.path.join(vendors, n, "op_api", d, OPAPI_SO_NAME) for d in LIBDIRS]

    for c in cands:
        if c and os.path.isfile(c):
            return c

    _die("找不到 %s。走 aclnn 必须装**带 op_api 的**自定义算子包。\n"
         "        1. 编包: bash build.sh --pkg --soc=mc62 --ops=fused_conv2d\n"
         "           （算子侧不用改代码：op_host/CMakeLists.txt 已经是 ACLNNTYPE aclnn，\n"
         "            aclnn 接口由 opbuild 从 OpDef 自动生成）\n"
         "        2. 装包，再 source 算子包的 set_env.sh，确认 ASCEND_CUSTOM_OPP_PATH 指到 vendors/<name>\n"
         "        3. 或者直接 --opapi-so <path> / 环境变量 FC2D_OPAPI_SO 指到那个文件\n"
         "        确认符号在不在: nm -D --defined-only <so> | grep aclnnFusedConv2d\n"
         "        找过这些位置:\n          %s"
         % (OPAPI_SO_NAME, "\n          ".join(c for c in cands if c) or "(无)"))


# ---------------------------------------------------------------- 工具
def row_major_strides(dims):
    """行主序连续张量的 strides。

    **单位是元素，不是字节。**写成字节累乘不会报错，会静默算错地址
    （int8 那条通路 elem=1 时甚至看不出来，fp16 会错一倍）。

    aclCreateTensor 要显式给 strides —— 它表达的是**视图**，不像
    aclCreateTensorDesc 那样只给 shape 就默认连续。
    """
    strides = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        strides[i] = strides[i + 1] * int(dims[i + 1])
    return strides


class AclnnLauncher:
    """持有 aclnn 下发所需的库句柄和 acl 对象。

    用法：
        L = AclnnLauncher(acl)
        tensors = {name: L.tensor(dtype, dims, dev_ptr) for ...}
        attrs   = L.make_attrs(info, shift1, shift2)
        out_t   = L.tensor(y_dtype, y_dims, y_dev)
        for _ in range(warmup + repeat):
            L.run_once(tensors, attrs, out_t, stream)   # 每轮自带第一段
        L.close()
    """

    def __init__(self, acl, nnopbase_so=None, opapi_so=None, vendor=None):
        self.acl = acl
        self.nnopbase_path = find_nnopbase_so(nnopbase_so)
        self.opapi_path = find_opapi_so(opapi_so, vendor)
        # RTLD_GLOBAL：生成的 aclnn_fused_conv2d.cpp 里那一堆 Nnopbase* 符号是
        # extern 的，要靠全局符号表从 libnnopbase 解析。
        self.nnop = ctypes.CDLL(self.nnopbase_path, mode=ctypes.RTLD_GLOBAL)
        self.opapi = ctypes.CDLL(self.opapi_path, mode=ctypes.RTLD_GLOBAL)
        self._bind()
        self._tensors = []
        self._arrays = []
        self._keepalive = []   # strides/dims 的 ctypes 数组在张量活着时不能被 GC
        self._workspaces = []

    # -- 符号签名 --------------------------------------------------------
    def _bind(self):
        """注册 ctypes 签名。

        **必须注册**：不注册时 ctypes 把 Python int 当 32 位 c_int 传，
        fixedShift1 / fixedShift2 / a16w8Shift1 这三个 int64_t 参数在 aarch64 上
        会拿到未定义的高 32 位；bool 不注册直接报
        "Don't know how to convert parameter N"。
        """
        c_vp, c_i, c_i64, c_u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64
        p_i64 = ctypes.POINTER(ctypes.c_int64)

        # acl_meta.h:37-50，都在 libnnopbase.so 里
        nnop_sig = [
            ("aclCreateTensor", [p_i64, c_u64, c_i, p_i64, c_i64, c_i, p_i64, c_u64, c_vp], c_vp),
            ("aclDestroyTensor", [c_vp], c_i),
            ("aclCreateIntArray", [p_i64, c_u64], c_vp),
            ("aclDestroyIntArray", [c_vp], c_i),
            ("aclCreateTensorList", [ctypes.POINTER(c_vp), c_u64], c_vp),
            ("aclDestroyTensorList", [c_vp], c_i),
        ]
        for name, argtypes, restype in nnop_sig:
            fn = getattr(self.nnop, name, None)
            if fn is None:
                _die("%s 里没有 %s —— 这个 CANN 的 aclnn 接口不完整，\n"
                     "        走不了 aclnn 路径，用 --launcher singleop。"
                     % (NNOPBASE_SO_NAME, name))
            fn.argtypes = argtypes
            fn.restype = restype
            setattr(self, name, fn)

        gws = getattr(self.opapi, "aclnnFusedConv2dGetWorkspaceSize", None)
        run = getattr(self.opapi, "aclnnFusedConv2d", None)
        if gws is None or run is None:
            _die("算子包的 %s 里没有 aclnnFusedConv2d[GetWorkspaceSize] 符号。\n"
                 "        这个包大概是在 aclnn 生成打开之前编的 —— 重新编包。\n"
                 "        确认用: nm -D --defined-only %s | grep aclnnFusedConv2d"
                 % (OPAPI_SO_NAME, self.opapi_path))
        # 签名和 build/autogen/aclnn_fused_conv2d.h 逐位置一致。
        gws.argtypes = [
            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,   # x f1 b1 f2 b2 dq2 q1 q2
            c_i64, c_i64,                                      # fixedShift1/2
            ctypes.c_bool, ctypes.c_bool,                      # relu1/2
            c_vp, c_vp, c_vp,                                  # kernelSize strides pads
            c_i64,                                             # a16w8Shift1
            c_vp,                                              # out
            ctypes.POINTER(c_u64), ctypes.POINTER(c_vp),       # workspaceSize*, executor**
        ]
        gws.restype = c_i
        run.argtypes = [c_vp, c_u64, c_vp, c_vp]
        run.restype = c_i
        self.gws, self.run = gws, run

    # -- 建对象 ----------------------------------------------------------
    def tensor(self, dtype, dims, dev_ptr):
        """按 FORMAT_ND 建一个连续张量。dev_ptr 是已经 H2D 好的 device 地址。

        注意参数顺序和 aclCreateTensorDesc **没有一点关系**：
          aclCreateTensorDesc(dtype, ndim, dims, format)
          aclCreateTensor(viewDims, viewDimsNum, dtype, stride, offset, format,
                          storageDims, storageDimsNum, data)
        dtype 从第 1 位挪到了第 3 位，照抄旧调用会把 dims 指针当 dtype 传。
        """
        dims = [int(d) for d in dims]
        strides = row_major_strides(dims)
        c_dims = (ctypes.c_int64 * len(dims))(*dims)
        c_strides = (ctypes.c_int64 * len(strides))(*strides)
        self._keepalive += [c_dims, c_strides]
        t = self.aclCreateTensor(c_dims, len(dims), dtype, c_strides, 0, ACL_FORMAT_ND,
                                 c_dims, len(dims), dev_ptr)
        if not t:
            _die("aclCreateTensor 返回 null（dtype=%d dims=%s）" % (dtype, dims))
        self._tensors.append(t)
        return t

    def int_array(self, vals):
        vals = [int(v) for v in vals]
        c_vals = (ctypes.c_int64 * len(vals))(*vals)
        self._keepalive.append(c_vals)
        a = self.aclCreateIntArray(c_vals, len(vals))
        if not a:
            _die("aclCreateIntArray 返回 null（%s）" % vals)
        self._arrays.append(a)
        return a

    def tensor_list(self, tensors):
        """给改动 1 用：把若干张量拼成一个 aclTensorList。

        这就是走 aclnn 的全部目的 —— atc --singleop 表达不了这个。
        缺席的那一层（比如某一层不带 bias）按 fused_sgd 的做法放一个 shape 为 [0]
        的空张量占位：`optim/fused_sgd/op_host/fused_sgd_tiling.cpp:127-151` 的判据是
        「dim_num < 1 或任一维 == 0 ⇒ 视为缺席」，而且是**逐实例**判的，
        所以 list 里每一层的有无可以独立表达。
        """
        arr = (ctypes.c_void_p * len(tensors))(*tensors)
        self._keepalive.append(arr)
        tl = self.aclCreateTensorList(arr, len(tensors))
        if not tl:
            _die("aclCreateTensorList 返回 null（%d 个张量）" % len(tensors))
        return tl

    def make_attrs(self, info, shift1, shift2, a16w8_shift1=29):
        """把 case 的超参做成 aclnn 要的属性值。

        两层同核时 kernel_size 发长度 2，异核时发长度 4 —— aclnn 下 aclIntArray
        的长度是运行时参数、不参与签名匹配（不像 .om），所以这里不再有「失配」
        风险；但 tiling 侧仍然按长度区分两层异核，不能图省事固定发 4。
        """
        kh2 = info.get("kh2") or info["kh"]
        kw2 = info.get("kw2") or info["kw"]
        ksize = ([info["kh"], info["kw"]] if (kh2, kw2) == (info["kh"], info["kw"])
                 else [info["kh"], info["kw"], kh2, kw2])
        return {
            "fixed_shift1": shift1,
            "fixed_shift2": shift2,
            "relu1": bool(info["relu1"]),
            "relu2": bool(info["relu2"]),
            "kernel_size": self.int_array(ksize),
            "strides": self.int_array([info["s1"], info["s2"]]),
            "pads": self.int_array([info["ph1"], info["pw1"], info["ph2"], info["pw2"]]),
            "a16w8_shift1": a16w8_shift1,
            "_ksize": ksize,   # 只为打印
        }

    # -- 下发 ------------------------------------------------------------
    def run_once(self, inputs, attrs, out, stream):
        """走完两段。

        **每轮都必须重新走第一段。** executor 是一次性的：第二段调用之后就失效了
        （生成代码走 NnopbaseRunWithWorkspace，框架在那之后回收 executor）。
        拿同一个 executor 下发第二次是未定义行为 —— 所以 warmup/repeat 的循环
        调这个函数，不要把第一段提到循环外面。
        """
        def t(name):
            return inputs.get(name) or None

        ws_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p()
        rc = self.gws(
            t("x"), t("filter1"), t("bias1"), t("filter2"), t("bias2"),
            t("dequant_scale2"), t("quant_scale1"), t("quant_scale2"),
            int(attrs["fixed_shift1"]), int(attrs["fixed_shift2"]),
            bool(attrs["relu1"]), bool(attrs["relu2"]),
            attrs["kernel_size"], attrs["strides"], attrs["pads"],
            int(attrs["a16w8_shift1"]),
            out, ctypes.byref(ws_size), ctypes.byref(executor))
        if rc != ACL_SUCCESS:
            _die("aclnnFusedConv2dGetWorkspaceSize = %d\n"
                 "        生成的第一段只判 4 个必需参数的空指针，然后把输入/属性排进 executor，\n"
                 "        最后 NnopbaseRunForWorkspace 跑 host tiling —— 形状被拒就是在那里报的。\n"
                 "        dtype 组合必须是这四条之一（support list 里编死的）：\n"
                 "          全fp16无scale / 全int8+q1+q2 / int8进fp16出+q1+dq2 / fp16x+int8权重+q1+dq2\n"
                 "        另外：板上必须装了带 FusedConv2d 的 mc62 算子包，aclnn 没有 .om 退路。" % rc)

        ws = ctypes.c_void_p()
        n = int(ws_size.value)
        if n > 0:
            rc = self.acl.aclrtMalloc(ctypes.byref(ws), n, ACL_MEM_MALLOC_HUGE_FIRST)
            if rc != ACL_SUCCESS:
                _die("workspace aclrtMalloc(%d) = %d" % (n, rc))
            self._workspaces.append(ws)
        # n == 0 时 workspace 传 nullptr —— aclrtMalloc(0) 不一定成功，别无条件申请。
        return self.run(ws, n, executor, stream), n

    # -- 清理 ------------------------------------------------------------
    def close(self):
        for a in self._arrays:
            self.aclDestroyIntArray(a)
        for t in self._tensors:
            self.aclDestroyTensor(t)
        self._arrays, self._tensors = [], []
        for ws in self._workspaces:
            self.acl.aclrtFree(ws)
        self._workspaces = []
        self._keepalive = []
