# 分词器

`h3_tokenizer.m`（521 行）是工程中仅有的三个 Objective-C 文件之一（另两个是 `h3_gpu.m` 与 `h3_metal.m`）。它用 Foundation 直接解析 HuggingFace `tokenizer.json`，**不引入任何第三方依赖**。

## 选择 Objective-C 的原因

`tokenizer.json` 的解析需要成熟的 JSON 处理与 Unicode 正规化。Foundation 的 `NSJSONSerialization` 与 `NSString` 提供这两者，而纯 C 就得自己写。代价是这个文件编译时要加 `-fobjc-arc`。

Sources: [Makefile](Makefile#L5-L5), [Makefile](Makefile#L16-L16)

## 契约

```c
#define H3_PAD_TOKEN_ID UINT32_C(151643)

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size);
void h3_tokenizer_free(h3_tokenizer *tokenizer);

int  h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                         int pad_empty, uint32_t **ids, size_t *count,
                         char *error, size_t error_size);
void h3_tokenizer_ids_free(uint32_t *ids);

char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size);
```

所有权规则很明确：`encode` 产出的 `*ids` 归调用方，用 `h3_tokenizer_ids_free` 释放；`decode` 产出的 UTF-8 串归调用方，用 `free` 释放。

Sources: [h3_tokenizer.h](h3_tokenizer.h#L7-L25)

## `pad_empty` 参数

`h3_tokenizer_encode` 的第三个参数 `pad_empty` 控制**空串的处理**：为真时给空输入补上 `H3_PAD_TOKEN_ID`（151643）而不是返回空数组。H3 的下游（Qwen 文本编码器）不接受零长度序列。

Sources: [h3_tokenizer.h](h3_tokenizer.h#L15-L18)

## 在流水线中的位置

分词器有两个来源，按 [模型加载与检查点布局](model-loading) 里的路径选择：

```
FL2VA/tokenizer/tokenizer.json     ← 纯文本与锚点路径
Ref2VA/tokenizer/tokenizer.json    ← 有序参考路径
```

在 `h3_generate` 里，分词是**条件构建的第一步**（phase 名 `"tokenizer"`，0/1 → 1/1）；纯文本路径下紧接着就调 `h3_tokenizer_encode`。

Sources: [h3.c](h3.c#L993-L996), [h3.c](h3.c#L1090-L1096), [h3.c](h3.c#L1485-L1489)

## 生命周期

分词器在每次 `h3_generate` 内**加载、用完即释放**（`h3.c:1795`），不跨调用缓存——它相对权重而言极小，重入成本远低于缓存复杂度。

Sources: [h3.c](h3.c#L1795-L1795)

## 测试

`tests/test_tokenizer.c` 用真实发布的 tokenizer 做往返验证：

```sh
./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json
```

`make test` 里该用例在 tokenizer.json 不存在时自动跳过。

Sources: [Makefile](Makefile#L130-L134)
