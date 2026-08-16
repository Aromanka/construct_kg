可以。结合你上传的目录结构、Release Notes，以及 OpenAlex 当前官方文档，我建议把整个流程设计成 **“先在压缩的 Works 元数据上筛选 → 再关联 Source → 最后只给命中的文章获取正文”**。核心结论是：**不要事先把整个 OpenAlex snapshot 全量解压到磁盘。**

你这份目录中真正应该作为主数据使用的是 `openalex-snapshot/data/`，其中包含 `works`、`sources`、`keywords`、`topics` 等实体。 `legacy-data` 则是 2025-11-04 创建的旧快照，不应该和当前 `data/` 混在一起处理。 Release Notes 也说明 2025-11-12 后主数据已经切换到 Walden 数据集，而且当时完整 Works 数量达到约 4.63 亿。

### 最推荐的数据处理架构

| 阶段            | 数据                                            |      是否事先解压 | 做什么                                  |
| ------------- | --------------------------------------------- | ----------: | ------------------------------------ |
| 1. 全量文章筛选     | `data/works/**/*.gz`                          |       **否** | gzip 流式读取，提取 title + abstract        |
| 2. 保存候选文章     | 自己生成 Parquet/DB                               |           — | 保存命中的 Work ID、标题、摘要、source ID、全文可用性等 |
| 3. 获取 Sources | `works.locations[].source`，必要时 `data/sources` |       **否** | 得到 journal/repository/source         |
| 4. 获取正文       | OpenAlex 独立 PDF / TEI XML 内容库                 | **只处理命中文章** | 下载/读取正文                              |
| 5. 后续重复关键词检索  | 自己建立的 title+abstract 索引                       |           — | 不再扫描 4 亿多 Works                      |

也就是说，处理顺序最好是：

**`.gz Works → 流式恢复摘要 → title/abstract 关键词筛选 → 保存少量 Work ID → 获取 source → 获取候选文章正文`**

而不是：

**全量解压 → 全量正文下载 → 再筛选。**

---

## 1. 你这个目录到底是什么结构

你当前这份数据大致可以理解为：

```text
openalex-snapshot/
├── data/
│   ├── works/
│   │   ├── manifest
│   │   ├── updated_date=2016-06-24/
│   │   │   └── part_0000.gz
│   │   ├── ...
│   │   └── updated_date=2026-02-25/
│   │       ├── part_0000.gz
│   │       ├── ...
│   │       └── part_0009.gz
│   │
│   ├── sources/
│   │   ├── manifest
│   │   └── updated_date=.../
│   │       └── part_0000.gz
│   │
│   ├── keywords/
│   ├── topics/
│   ├── authors/
│   └── ...
│
├── legacy-data/
├── README.txt
├── RELEASE_NOTES.txt
└── browse.html
```

其中最重要的是：

**`data/works` 是你的主表。**

官方说明 snapshot 中的 JSONL 文件采用 **gzip 压缩的 JSON Lines**，也就是：

```text
一行 = 一个 Work JSON 对象
```

因此完全可以逐行读取，不需要先生成巨大的 `.jsonl` 文件。当前官方 snapshot 的 gzip JSONL 总体量约 330 GB，完全解压后约 1.6 TB，这正是为什么不建议物理解压整个数据集。([developers.openalex.org][1])

你的本地 `works` 从 2016 年一直有 `updated_date` 分区。

这里一个**非常容易踩坑的地方**是：

> `updated_date=2026-02-25` 不是“2026-02-25 的完整快照”。

它表示：

> **最后一次更新日期为 2026-02-25 的那些 Work。**

官方明确说明，完整 snapshot 是所有 `updated_date` 分区的并集；记录更新以后，会从旧日期分区移动到新日期分区。([developers.openalex.org][1])

所以：

```text
错误：
只读取 works/updated_date=2026-02-25/

正确：
读取 works/updated_date=*/part_*.gz
```

如果你的目标是完整文章库，就必须扫描 `works` 下**全部日期分区**。

---

## 2. 特别注意 manifest 和那些奇怪文件

你的目录里出现了一些这种名字：

```text
part_0005.gz
part_0005.gz.BbCc89ae
```

例如你提供的目录中确实存在这种随机后缀文件。

因此我**不建议直接使用**：

```python
glob("part_*")
```

否则有可能把非标准对象也纳入。

优先方案是读取对应实体的：

```text
data/works/manifest
data/sources/manifest
```

把 manifest 中列出的文件当成 authoritative file list。

当前 OpenAlex 官方也推荐把 manifest 作为完整性和文件列表依据，并指出 manifest 是数据文件全部写入以后才生成的。([developers.openalex.org][1])

如果你暂时不处理 manifest，至少严格匹配：

```text
part_XXXX.gz
```

而不是所有 `part_*`。

---

# 3. 任务一：每篇文章的“摘要”怎么获取

摘要在 `Work` 中不是普通字符串，而主要是：

```json
{
    "abstract_inverted_index": {
        "machine": [0, 15],
        "learning": [1, 16],
        ...
    }
}
```

OpenAlex 官方说明，`abstract_inverted_index` 是摘要的**倒排索引形式**；由于法律方面的限制，它不直接提供普通 plaintext abstract。([OpenAlex][2])

你可以根据 position 恢复：

```python
def restore_abstract(inv):
    if not inv:
        return None

    n = max(
        p
        for positions in inv.values()
        for p in positions
    ) + 1

    words = [""] * n

    for word, positions in inv.items():
        for p in positions:
            words[p] = word

    return " ".join(words)
```

因此你的第一遍处理实际上只需要：

```python
import gzip
import json

with gzip.open("part_0000.gz", "rt", encoding="utf-8") as f:
    for line in f:
        work = json.loads(line)

        work_id = work["id"]
        title = work.get("title") or work.get("display_name")
        abstract = restore_abstract(
            work.get("abstract_inverted_index")
        )

        # 在这里直接做关键词筛选
```

注意这里实际上发生了 gzip **解压**，但属于：

> **流式解压到内存。**

不是：

> gzip → 写出完整 `.jsonl` → 再读取。

这是两个完全不同的概念。

所以对你的问题“什么时候解压”，答案是：

**读取某个 `.gz` 文件时即时解压即可，不需要提前解压。**

另外必须接受一个数据事实：**不是每一个 Work 都有摘要。** OpenAlex 本身提供 `has_abstract` 条件来区分是否有摘要。([OpenAlex][2])

因此你的最终数据模型最好允许：

```text
abstract = NULL
```

而不要假设所有文章都有摘要。

---

# 4. 任务一中的“正文”必须单独处理

这一点尤其重要：

## `data/works/*.gz` 里不是论文全文库。

它主要是 Work metadata。

也就是说：

```text
title                  有
abstract_inverted_index 部分有
authors                有
locations              有
sources                有
DOI                    常见
references             有
topics/keywords         有
PDF 正文                没有直接塞在这里
```

当前 OpenAlex 已经建立了一个**独立的全文内容库**，官方称大约缓存：

* 约 6000 万份 PDF
* 约 4300 万份 GROBID TEI XML

TEI XML 是从 PDF 解析出的结构化正文。([OpenAlex][3])

所以如果你的要求真的是：

> “每篇 OpenAlex 文章都必须有正文”

那么**仅靠 OpenAlex 无法保证做到**。

因为全文只覆盖部分 Works，而且有些论文只有 publisher landing page、有些受版权/订阅限制、有些没有被 OpenAlex 缓存。当前全文文档也明确说明 PDF 保留原始版权，OpenAlex 并不因为提供元数据而重新授予全文版权。([OpenAlex][3])

你这份 snapshot 自身附带的是 CC0 数据许可文本， **但这不能推导出每篇论文 PDF 也是 CC0。**

因此应该把：

```text
OpenAlex metadata
```

和：

```text
article full text
```

作为两个数据层分别管理。

---

# 5. 正文应该什么时候下载？

**关键词筛选以后。**

这是整个方案中最值得优化的一步。

假设：

```text
OpenAlex Works
≈ 数亿条

关键词筛选后
≈ 100 万条

真正最终使用
≈ 10 万条
```

那么应该：

```text
4亿 Works metadata
        ↓
title/abstract 过滤
        ↓
100万 Work IDs
        ↓
检查全文可用性
        ↓
只下载这些文章的 PDF/XML
```

不要：

```text
4亿 Works
 ↓
先下载几千万 PDF
 ↓
PDF 解析
 ↓
然后才做关键词筛选
```

后者对网络、磁盘、PDF parsing CPU 都极其浪费。

---

# 6. 获取正文的优先方式

筛选出 Work ID 后，我建议优先级如下：

```text
首选：
OpenAlex GROBID TEI XML
        ↓
结构化正文，适合 NLP / LLM / RAG

其次：
OpenAlex PDF
        ↓
自己 PDF → text / Markdown

再次：
Work.locations[].pdf_url
        ↓
publisher/repository 原始 PDF

最后：
landing_page_url
        ↓
自己进一步处理
```

当前 OpenAlex content API 可以按 Work ID 获取：

```text
https://content.openalex.org/works/{WORK_ID}.pdf
```

或者：

```text
https://content.openalex.org/works/{WORK_ID}.grobid-xml
```

并且当前 Work 数据还可以通过 `has_content.pdf` / `has_content.grobid_xml` 判断 OpenAlex 是否已有缓存内容；`content_urls` 则属于 API-only 字段，不在 snapshot 中。([OpenAlex][3])

但这里对你手头的 **2026-02-25 snapshot** 要加一个限定：

**不要直接假设这一旧版 snapshot 一定具有当前文档中的全部新字段。**

你给我的只是目录副本，没有实际 `works/*.gz` 样本，所以目前不能从附件确认 `has_content` 是否已经存在于你这一版 Work JSON 中。

最安全的程序写法是：

```python
has_content = work.get("has_content")
```

字段不存在时就 fallback 到：

```python
locations
best_oa_location
```

或者后续根据 Work ID 请求当前 API。

---

# 7. 任务二：每篇文章的 Sources 从哪里拿？

这一步其实比你想象中简单。

你**不需要为了知道每篇论文发表在哪儿，先读完整 `data/sources`。**

一个 Work 本身的 location 中已经包含 source。

例如逻辑上类似：

```json
{
  "primary_location": {
    "source": {
      "id": "https://openalex.org/S123...",
      "display_name": "Nature",
      "type": "journal"
    }
  },

  "locations": [
    {
      "source": {...}
    },
    {
      "source": {...}
    }
  ]
}
```

官方说明，一篇 Work 可以存在于多个 Location，例如：

```text
出版社期刊
机构 repository
PubMed Central
arXiv
bioRxiv
……
```

其中：

`primary_location.source`
= 最接近 version of record 的主要 source。

`locations[].source`
= **所有承载该 Work 的 sources。**

`best_oa_location.source`
= 最佳 Open Access location 的 source。([GitHub][4])

所以如果你说的“每篇文章的 sources”是 OpenAlex 的 Source 概念，我建议保存：

```text
work_id
primary_source_id
primary_source_name

all_source_ids[]
all_source_names[]
```

---

## 那么 `data/sources/` 有什么作用？

它相当于 **Source 主数据表 / dimension table**。

你的目录中确实有一个独立 Sources 实体，而且分为多个日期 partition。

Work 内部的：

```text
locations[].source
```

是一个相对轻量的 Source 对象。

如果你只需要：

```text
source ID
source name
journal/repository 类型
ISSN
```

Work 内通常就已经足够。

如果后续还需要完整 Source 属性，那么做：

```text
works.locations[].source.id
              ↓
         source_id
              ↓ JOIN
data/sources/
```

即可。

建议自己建一个：

```text
source_id → source metadata
```

的小表。

`Sources` 的数据规模远小于 Works，所以这个表可以一次性流式导入 SQLite / DuckDB / PostgreSQL。

仍然：

**不用先解压整个 `sources`。**

---

# 8. 如果你说的 sources 实际上是“参考文献”

这里需要特别区分。

OpenAlex 的术语：

```text
Source
```

是：

> journal / conference / repository 等承载论文的来源。([OpenAlex][5])

而论文引用了哪些文章则是：

```python
work["referenced_works"]
```

官方 Work schema 将其定义为这篇 Work 引用的其他 OpenAlex Work IDs。([OpenAlex][2])

所以如果你实际上说的 Sources 是：

> “这篇论文引用了哪些文献？”

那就不需要 `data/sources`，而应该读取：

```text
referenced_works
```

然后再通过 Work ID join 回 `data/works`。

---

# 9. 任务三：根据标题 / 摘要做关键词筛选

这个任务**完全应该在 `data/works` 上做**。

不要把：

```text
data/keywords/
```

误认为“论文全文关键词倒排索引”。

你的目录里确实有一个独立 `keywords` entity。

但它不是：

```text
所有文章 title/abstract 的全文搜索索引
```

你需要的是：

```python
title = work["title"]

abstract = restore_abstract(
    work["abstract_inverted_index"]
)
```

然后：

```python
text = f"{title} {abstract or ''}".lower()

if any(keyword in text for keyword in keywords):
    # 保留
```

当然真实科研筛选通常建议做得比简单 `in` 更严谨，比如：

```text
大小写归一化
Unicode normalization
词形变化
AND / OR / NOT
完整词匹配
短语匹配
同义词
语言检测
正则表达式
```

---

# 10. 一次筛选 vs 经常筛选，处理方法不同

这是决定你是否需要建数据库/索引的关键。

### 情况 A：只需要跑一次关键词

例如：

```text
"large language model"
AND
healthcare
NOT
review
```

最简单：

```text
.gz
 ↓ gzip.open()
JSONL 一行一行
 ↓
恢复 abstract
 ↓
匹配 title + abstract
 ↓
只输出命中的文章
```

**根本不需要建立完整数据库。**

---

### 情况 B：未来会反复换关键词

比如你之后会反复执行：

```text
query 1
LLM + medicine

query 2
GPT + education

query 3
transformer + finance

query 4
foundation model + robotics
```

那就不要每次重新扫描几百 GB gzip。

最好第一次做：

```text
OpenAlex .gz
       ↓
流式解析一次
       ↓
id
title
abstract
publication_year
type
source_id
has_content
       ↓
本地检索库
```

然后建立 title + abstract 全文索引。

比如：

```text
SQLite FTS5         小/中型项目
PostgreSQL FTS      已经有 PG 环境
OpenSearch          大规模检索服务
Elasticsearch       大规模检索服务
Tantivy/Lucene 类   高性能本地倒排索引
```

或者至少把精简字段生成：

```text
works_search.parquet
```

以后再查询。

这比把原始 OpenAlex JSON 全部永久解压出来合理得多。

---

# 11. 推荐你保存一个“候选文章表”

第一遍扫描 `works/*.gz` 的输出不要只是 Work ID，建议直接保存成类似：

```text
work_id
doi
title
abstract
publication_year
language
type

primary_source_id
primary_source_name

all_source_ids

best_oa_pdf_url
landing_page_url

has_content_pdf
has_content_xml

matched_keywords
```

后续的数据流程就变成：

```text
             OpenAlex works/*.gz
                     │
                     │ 流式 gzip
                     ▼
              title + abstract
                     │
                     │ keyword filter
                     ▼
            candidate_works.parquet
                  /          \
                 /            \
                ▼              ▼
       Source enrichment    Full text
       data/sources         PDF / TEI
```

这就是我最推荐的整体架构。

---

# 12. 到底“什么时候应该真正落盘解压”？

我的建议是：**几乎没有必要把原始 `.gz` 解压成同等结构的 `.jsonl` 长期存储。**

如果后面的程序不支持 gzip，或者你确实需要极高频率重复扫描同一批原始文件，才有物理解压的理由。

但即使这样，我也不会推荐：

```text
.gz
↓
.jsonl
```

而是建议：

```text
.gz JSONL
↓ 一次流式转换
Parquet / database / search index
```

因为 raw JSONL 解压后磁盘占用很大，同时也没有增加真正有价值的索引能力。当前 OpenAlex 官方实际上已经开始同时发布 gzip JSONL 和 Snappy Parquet 两种完整格式；当前版本同样强调 Parquet 可以按列直接读取。([developers.openalex.org][1])

你的 2026-02-25 本地副本还是：

```text
data/works/...
```

这种布局，而当前官方文档已经显示新的：

```text
data/jsonl/works/
data/parquet/works/
```

所以**不要机械按照当前网页的新目录重命名你已有的数据**；你的现有 snapshot 按本地 `data/works` 处理即可。([developers.openalex.org][1])

---

# 13. 对你三个需求逐项给最终结论

1. **文章摘要：可以从 `data/works/*.gz` 获取。**读取 `abstract_inverted_index` 并恢复成文本即可；但不是所有文章都有摘要。不要提前解压，扫描 `.gz` 时即时解压。

2. **文章正文：不能从这份 metadata snapshot 直接保证获得。**应该先完成 title/abstract 筛选，再根据候选 Work ID 获取 OpenAlex 的 PDF / GROBID XML，或者使用 `locations[].pdf_url` 等原始全文位置。OpenAlex 当前 API 的普通 `search` 会搜索 title、abstract 和 fulltext，([OpenAlex][6]) 但如果你严格要求“**只按标题+摘要决定是否入选**”，本地筛选反而更可控。

3. **文章 Sources：一般直接从 Work 的 `primary_location.source` 和 `locations[].source` 获取。**如果只需要 source 名称/ID/type，不必预先读取 `data/sources`；需要完整 Source metadata 时，再用 Source ID join `data/sources`。

4. **关键词筛选：第一步就做，而且只需要扫 `works`。**一次性任务就流式扫描 `.gz`；需要经常换关键词，就第一次扫描时把 `id + title + abstract + source` 建成本地搜索索引。

5. **解压策略：不做“全量预解压”。**Metadata `.gz` 永久保持压缩；处理时流式解压。全文则只对筛选后的候选文章下载/解析；如果拿到的是 `.xml.gz`，也只在读取该 XML 时解压。

如果从工程实施角度压缩成一句话，就是：

> **先用 `works/*.gz` 在压缩状态下流式完成 title/abstract 筛选，把数亿条缩减到你的候选集；然后才处理 sources 和正文。绝对不要先把整个 OpenAlex 或全部 PDF 解压/下载后再筛。**

[1]: https://developers.openalex.org/download/snapshot-format "Snapshot data format - OpenAlex Developers"
[2]: https://developers.openalex.org/api-reference/works/get-a-single-work?utm_source=chatgpt.com "Get a single work - OpenAlex Developers"
[3]: https://developers.openalex.org/download/full-text-pdfs "Full-text PDFs - OpenAlex Developers"
[4]: https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/location-object.md?utm_source=chatgpt.com "openalex-docs/api-entities/works/work-object/location-object.md at main · ourresearch/openalex-docs · GitHub"
[5]: https://developers.openalex.org/api-reference/sources?utm_source=chatgpt.com "Sources Overview - OpenAlex Developers"
[6]: https://developers.openalex.org/guides/searching?utm_source=chatgpt.com "Search - OpenAlex Developers"
