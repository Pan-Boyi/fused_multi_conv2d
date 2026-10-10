/* libascendcl.so 的桩：让 run_fused_conv2d.py 的整条模型 API 下发路径能在
 * 没有 NPU 的机器上跑完。
 *
 * 为什么需要：真机上 aclInit 就会失败（chipType=0），dry-run 又刻意绕开 ACL 分支，
 * 于是那一整段（模型加载、按名字取下标、dataset、execute、读回、比对、诊断）
 * 在本地**一行都没被执行过**。y_bytes 那个 UnboundLocalError 就是这么漏出去的。
 *
 * 行为由环境变量控制：
 *   FC2D_STUB_CONF    每行 "<名字> <字节数>"，顺序就是模型报的输入顺序；
 *                     最后一行 "OUT <字节数>"
 *   FC2D_STUB_GOLDEN  aclmdlExecute 往输出缓冲里拷的内容（走通比对那段）
 *   FC2D_STUB_NONAME  非空 -> aclmdlGetInputIndexByName 一律失败，
 *                     用来逼运行器走「退回 IR 顺序」那条分支
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAXIN 16
static char  g_name[MAXIN][64];
static size_t g_size[MAXIN];
static int    g_nin = 0;
static size_t g_out = 0;
static int    g_loaded = 0;

typedef struct { void* p; size_t n; } Buf;
typedef struct { Buf* b[MAXIN]; int n; } Dataset;

static void load_conf(void)
{
    if (g_loaded) return;
    g_loaded = 1;
    const char* c = getenv("FC2D_STUB_CONF");
    if (!c) { fprintf(stderr, "[stub] 没设 FC2D_STUB_CONF\n"); return; }
    FILE* f = fopen(c, "r");
    if (!f) { fprintf(stderr, "[stub] 打不开 %s\n", c); return; }
    char nm[64]; unsigned long long sz;
    while (fscanf(f, "%63s %llu", nm, &sz) == 2) {
        if (!strcmp(nm, "OUT")) { g_out = (size_t)sz; continue; }
        if (g_nin < MAXIN) { snprintf(g_name[g_nin], 64, "%s", nm); g_size[g_nin] = (size_t)sz; g_nin++; }
    }
    fclose(f);
    fprintf(stderr, "[stub] 配置: %d 个输入, 输出 %zu 字节\n", g_nin, g_out);
}

int aclInit(const char* p) { (void)p; load_conf(); return 0; }
int aclFinalize(void) { return 0; }
int aclrtSetDevice(int32_t d) { (void)d; return 0; }
int aclrtResetDevice(int32_t d) { (void)d; return 0; }
int aclrtCreateStream(void** s) { *s = malloc(8); return 0; }
int aclrtDestroyStream(void* s) { free(s); return 0; }
int aclrtSynchronizeStream(void* s) { (void)s; return 0; }
int aclrtMalloc(void** p, size_t n, int kind) { (void)kind; *p = malloc(n ? n : 1); return *p ? 0 : 1; }
int aclrtFree(void* p) { free(p); return 0; }
int aclrtMemcpy(void* d, size_t dn, const void* s, size_t sn, int kind)
{ (void)kind; memcpy(d, s, dn < sn ? dn : sn); return 0; }

/* 单算子那套：符号还被 ctypes 绑着（sig 表里），但新路径不调。给个桩免得加载失败 */
void* aclCreateTensorDesc(int dt, int nd, const int64_t* dims, int fmt)
{ (void)dt; (void)nd; (void)dims; (void)fmt; return malloc(8); }
void aclDestroyTensorDesc(void* d) { free(d); }
void* aclopCreateAttr(void) { return malloc(8); }
void aclopDestroyAttr(void* a) { free(a); }
int aclopSetAttrInt(void* a, const char* k, int64_t v) { (void)a;(void)k;(void)v; return 0; }
int aclopSetAttrFloat(void* a, const char* k, float v) { (void)a;(void)k;(void)v; return 0; }
int aclopSetAttrBool(void* a, const char* k, uint8_t v) { (void)a;(void)k;(void)v; return 0; }
int aclopSetAttrListInt(void* a, const char* k, int n, const int64_t* v)
{ (void)a;(void)k;(void)n;(void)v; return 0; }
const char* aclGetRecentErrMsg(void) { return ""; }

void* aclCreateDataBuffer(void* p, size_t n)
{ Buf* b = (Buf*)malloc(sizeof(Buf)); b->p = p; b->n = n; return b; }
int aclDestroyDataBuffer(void* b) { free(b); return 0; }

int aclmdlLoadFromFile(const char* path, uint32_t* id)
{
    load_conf();
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[stub] om 打不开: %s\n", path); return 1; }
    fclose(f);
    *id = 1;
    return 0;
}
int aclmdlUnload(uint32_t id) { (void)id; return 0; }
void* aclmdlCreateDesc(void) { load_conf(); return malloc(8); }
int aclmdlDestroyDesc(void* d) { free(d); return 0; }
int aclmdlGetDesc(void* d, uint32_t id) { (void)d; (void)id; return 0; }
size_t aclmdlGetNumInputs(void* d) { (void)d; load_conf(); return (size_t)g_nin; }
size_t aclmdlGetNumOutputs(void* d) { (void)d; return 1; }
size_t aclmdlGetInputSizeByIndex(void* d, size_t i)
{ (void)d; return (i < (size_t)g_nin) ? g_size[i] : 0; }
size_t aclmdlGetOutputSizeByIndex(void* d, size_t i) { (void)d; (void)i; return g_out; }
const char* aclmdlGetInputNameByIndex(void* d, size_t i)
{ (void)d; return (i < (size_t)g_nin) ? g_name[i] : ""; }
const char* aclmdlGetOutputNameByIndex(void* d, size_t i) { (void)d; (void)i; return "y"; }
int aclmdlGetInputIndexByName(void* d, const char* name, size_t* idx)
{
    (void)d;
    load_conf();
    if (getenv("FC2D_STUB_NONAME")) return 1;   /* 逼它退回 IR 顺序 */
    for (int i = 0; i < g_nin; i++)
        if (!strcmp(g_name[i], name)) { *idx = (size_t)i; return 0; }
    return 1;
}
void* aclmdlCreateDataset(void) { Dataset* s = (Dataset*)calloc(1, sizeof(Dataset)); return s; }
int aclmdlDestroyDataset(void* s) { free(s); return 0; }
int aclmdlAddDatasetBuffer(void* s, void* b)
{ Dataset* ds = (Dataset*)s; if (ds->n < MAXIN) ds->b[ds->n++] = (Buf*)b; return 0; }

int aclmdlExecute(uint32_t id, void* in, void* out)
{
    (void)id; (void)in;
    Dataset* o = (Dataset*)out;
    if (!o || o->n < 1) return 1;
    const char* g = getenv("FC2D_STUB_GOLDEN");
    if (!g) return 0;                       /* 不给 golden 就只验路径、不验数值 */
    FILE* f = fopen(g, "rb");
    if (!f) { fprintf(stderr, "[stub] golden 打不开: %s\n", g); return 1; }
    size_t got = fread(o->b[0]->p, 1, o->b[0]->n, f);
    fclose(f);
    if (got != o->b[0]->n)
        fprintf(stderr, "[stub] golden 只有 %zu 字节，输出缓冲 %zu\n", got, o->b[0]->n);
    return 0;
}
