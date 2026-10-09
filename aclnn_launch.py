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
                                               两段式接口在 opapi 库里（哪一个见下）

**不需要 .om 是实际的简化**：走 aclnn 时 fc2d.py 的 `om` 步骤整步可以跳过，连带
singleop.json 的生成、以及「属性值必须和 .om 完全一致否则报 100024」那一整类坑都不存在
—— aclnn 是按符号直接调的，不做签名匹配。**代价**是 aclnn 没有「板上没装算子包也能靠
.om 跑」这条退路：板子上必须装好带 FusedConv2d 的 mc62 算子包。

**两段式接口落在哪个 so，取决于用什么命令编，不取决于算子将来归属哪个库。**
同一份 autogen 源码只因为一个 cmake 开关 `ENABLE_CUSTOM`（ops-nn 的
CMakeLists.txt:55 `option(ENABLE_CUSTOM ... OFF)`，**默认 OFF = 内置**）分叉：

    ENABLE_CUSTOM=OFF（内置，默认）           ENABLE_CUSTOM=ON（vendor 自定义包）
    --------------------------------------    ----------------------------------------
    gen_norm_symbol() → target opapi_nn       gen_cust_symbol() → target cust_opapi
    libopapi_nn.so                            libcust_opapi.so
    <arch>-linux/lib64/                       vendors/<name>_nn/op_api/lib/
    bash build.sh --opapi -f <改动清单>       bash build.sh --pkg --ops=fused_conv2d

`--ops=` 会**无条件**把开关拨到 vendor 侧（build.sh:865-867 的 `ops=*)` 分支里直接
`ENABLE_CUSTOM=TRUE`），而 `-f` 走的 set_ci_mode（build.sh:1432-1452）只注入
`-DASCEND_OP_NAME` / `-DASCEND_COMPILE_OPS`、**不碰这个开关** —— 所以「内置 flavor
+ 只编一个算子」必须用 `-f`（`--ops` 和 `--opapi` 还互斥，build.sh:477）。

2026-10-09 实测：`bash build.sh --opapi -f <清单>` 在 ENABLE_CUSTOM=FALSE 下编出
`build/libopapi_nn.so`（67,880 字节），`nm -D` 里正好两个 aclnn 符号
—— aclnnFusedConv2d + aclnnFusedConv2dGetWorkspaceSize，NEEDED 只有
libnnopbase.so。所以算子合进主线后两段式接口就在 libopapi_nn.so 里。
（注意 62 个未定义符号中 `ge::TypeUtils::*` / `error_message::*` 来自
libgraph.so，而它**不在** NEEDED 里 —— dlopen 前得先把它拉进全局符号表，
下面 _preload_soft_deps() 干这事。）

但**开发期上板自验仍然该走 vendor 包**，那不是走偏：上游开发指南
（docs/zh/develop/aicore_develop_guide.md:405-449 的标准第 3、4 步，:802-806 的
aclnn 验证就是 export LD_LIBRARY_PATH=.../opp/vendors/<name>_nn/op_api/lib）和
PR 门禁（scripts/ci/check_pkg.sh:102 `build.sh --pkg --vendor_name=$n --ops=$n`）
规定的就是它，门禁对 conv2d_v2 / conv3d_v2 / mat_mul_v3 这些**早已在主线内置库**
的算子也是同样跑法。而且 vendor 包是增量安装、不覆写 toolkit 的 opp/built-in，
还能把 kernel / tiling / op-info 和 aclnn 一起带上板。

结论：**两种 flavor 都得能找到**。所以下面 open_opapi_so() 按**符号**探测，
不按文件名猜 —— 文件名是构建模式的产物，符号才是判据。
"""

import ctypes
import os

ACL_SUCCESS = 0
ACL_FORMAT_ND = 2
ACL_MEM_MALLOC_HUGE_FIRST = 0

# 两段式接口可能住在这两个 so 的任意一个，取决于编包时 ENABLE_CUSTOM 的取值
# （见模块 docstring）。列表顺序 = 探测优先级：内置在前，因为算子最终归属内置库，
# 而手边的 vendor 包往往是旧的。但**真正的判据是符号在不在**，不是名字。
BUILTIN_OPAPI_SO_NAME = "libopapi_nn.so"    # ENABLE_CUSTOM=OFF → <arch>-linux/lib64
VENDOR_OPAPI_SO_NAME = "libcust_opapi.so"   # ENABLE_CUSTOM=ON  → vendors/<n>_nn/op_api/lib
# 探测用的符号：两段式的第一段，autogen 必然导出。
PROBE_SYMBOL = "aclnnFusedConv2dGetWorkspaceSize"
# libopapi_nn.so 的 ge::TypeUtils / error_message 符号在这里，而它不在 NEEDED 里。
# 缺了它 dlopen 直接报 undefined symbol，所以探测前先尽力拉进来。
SOFT_DEP_SO_NAMES = ("libgraph.so", "libascendalog.so", "libalog.so")
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


def _preload_soft_deps():
    """把 opapi 库的隐式依赖拉进全局符号表，尽力而为。

    内置的 libopapi_nn.so 里 `ge::TypeUtils::FormatToSerialString` /
    `error_message::ReportInnerErrMsg` / `DlogRecord` 这些是未定义符号，而
    libgraph.so 和 alog **不在它的 NEEDED 里**（readelf -d 实测只有
    libnnopbase.so + libstdc++/libgcc/libc）。不先加载就直接 dlopen 会报
    undefined symbol。失败不致命 —— 很多场景 libascendcl 已经把它们带进来了。
    """
    loaded = []
    for name in SOFT_DEP_SO_NAMES:
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            loaded.append(name)
        except OSError:
            pass
    return loaded


def _opapi_candidates(explicit=None, vendor=None):
    """按优先级列出所有可能的 opapi so 路径。

    vendor 包装的是 op_api/**lib**/，不是 lib64（cmake/variables.cmake:78 用 `lib`）；
    内置那侧是 <arch>-linux/lib64（variables.cmake:102），而装好的 toolkit 里
    lib64 通常是指向它的 symlink。两边的目录名都列上，便宜。
    """
    LIBDIRS = ("lib", "lib64")
    cands = []

    def add(p):
        if p and p not in cands:
            cands.append(p)

    # -- 0. 显式指定最高优先：用户知道自己在干什么
    add(explicit)
    add(os.environ.get("FC2D_OPAPI_SO", ""))

    # -- 1. 内置 flavor：装好的 toolkit，以及本地 build 目录里刚编出来的那个
    for env in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH"):
        root = os.environ.get(env, "")
        if not root:
            continue
        # ASCEND_OPP_PATH 指到 <root>/opp，往上一层才是 root
        roots = [root, os.path.dirname(root.rstrip("/"))]
        for r in roots:
            add(os.path.join(r, "lib64", BUILTIN_OPAPI_SO_NAME))
            for arch in ("aarch64-linux", "x86_64-linux"):
                add(os.path.join(r, arch, "lib64", BUILTIN_OPAPI_SO_NAME))
    # 本地 ops-nn 工作树里 `build.sh --opapi -f <清单>` 的产物就落在 build/ 根下
    for root in os.environ.get("FC2D_OPSNN_DIR", "").split(":"):
        if root:
            add(os.path.join(root, "build", BUILTIN_OPAPI_SO_NAME))

    # -- 2. vendor flavor：自定义算子包
    for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if root:
            for d in LIBDIRS:
                add(os.path.join(root, "op_api", d, VENDOR_OPAPI_SO_NAME))
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
            if not n:
                continue
            for d in LIBDIRS:
                add(os.path.join(vendors, n, "op_api", d, VENDOR_OPAPI_SO_NAME))

    # -- 3. 兜底：交给 ld.so 按 LD_LIBRARY_PATH 找（两个名字都试）
    add(BUILTIN_OPAPI_SO_NAME)
    add(VENDOR_OPAPI_SO_NAME)
    return cands


def open_opapi_so(explicit=None, vendor=None):
    """找到并加载带 aclnnFusedConv2d 的 opapi so，返回 (path, CDLL handle)。

    **按符号探测，不按文件名判定。** 名字只决定去哪儿找；一个存在但里面没有
    aclnnFusedConv2d 的库（比如 CANN 自带的那个 18MB libopapi_nn.so —— 算子还没
    合进主线时它当然没有我们的符号）必须被跳过而不是被当成答案。所以逐个
    dlopen + getattr，第一个带符号的赢，并把每个候选的淘汰原因记下来。
    """
    if explicit and not os.path.isfile(explicit):
        _die("--opapi-so 指的文件不存在: %s" % explicit)

    _preload_soft_deps()
    cands = _opapi_candidates(explicit, vendor)
    rejected = []
    for c in cands:
        bare = os.sep not in c
        if not bare and not os.path.isfile(c):
            rejected.append((c, "文件不存在"))
            continue
        try:
            h = ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            rejected.append((c, "dlopen 失败: %s" % e))
            continue
        if getattr(h, PROBE_SYMBOL, None) is None:
            rejected.append((c, "加载成功但没有 %s 符号" % PROBE_SYMBOL))
            continue
        return c, h

    _die("所有候选的 opapi 库里都没有 %s。\n"
         "        两种 flavor 都可以，关键是**那个 so 里得有这个符号**：\n"
         "        A) 内置（算子最终归属，产出 %s）:\n"
         "             git diff --name-only upstream/master...HEAD | sed \"s|^|$PWD/|\" > /tmp/fl.txt\n"
         "             bash build.sh --opapi -f /tmp/fl.txt\n"
         "           产物在 <ops-nn>/build/%s，用 --opapi-so 直接指过去，\n"
         "           或者 export FC2D_OPSNN_DIR=<ops-nn 路径>。\n"
         "           注意不能用 --ops=（它会把 ENABLE_CUSTOM 拨到 vendor 侧）。\n"
         "        B) vendor 自定义包（上板自验走这条，产出 %s）:\n"
         "             bash build.sh --pkg --soc=mc62 --ops=fused_conv2d\n"
         "           装包后 source 算子包的 set_env.sh，确认 ASCEND_CUSTOM_OPP_PATH\n"
         "           指到 vendors/<name>_nn。\n"
         "        C) 或者直接 --opapi-so <path> / export FC2D_OPAPI_SO=<path>。\n"
         "        自己确认符号: nm -D --defined-only <so> | grep aclnnFusedConv2d\n"
         "        探测记录（候选 → 淘汰原因）:\n          %s"
         % (PROBE_SYMBOL, BUILTIN_OPAPI_SO_NAME, BUILTIN_OPAPI_SO_NAME,
            VENDOR_OPAPI_SO_NAME,
            "\n          ".join("%s  →  %s" % (p, why) for p, why in rejected) or "(无候选)"))


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
        # RTLD_GLOBAL：生成的 aclnn_fused_conv2d.cpp 里那一堆 Nnopbase* 符号是
        # extern 的，要靠全局符号表从 libnnopbase 解析。
        # **必须先加载 nnopbase**：open_opapi_so 是靠真的 dlopen 候选库来探测符号的，
        # nnopbase 不在全局表里时那些候选会以 undefined symbol 失败、被误判成不可用。
        self.nnop = ctypes.CDLL(self.nnopbase_path, mode=ctypes.RTLD_GLOBAL)
        self.opapi_path, self.opapi = open_opapi_so(opapi_so, vendor)
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
            _die("%s 里缺 aclnnFusedConv2d 第二段的符号。\n"
                 "        open_opapi_so 探的是 %s（第一段），所以能走到这儿说明\n"
                 "        那个库只导出了一半 —— 不是「没编 aclnn」，更像是链接\n"
                 "        时漏了 autogen 的 object，或者库被手工裁过。\n"
                 "        确认用: nm -D --defined-only %s | grep aclnnFusedConv2d"
                 % (self.opapi_path, PROBE_SYMBOL, self.opapi_path))
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
