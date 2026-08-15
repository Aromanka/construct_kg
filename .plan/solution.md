当前的知识图谱构建仓库位于：https://github.com/Aromanka/construct_kg
我选择了10篇文档，用于测试其性能。其并行速度和知识抽取成功率都很高，但是可视化抽取的知识后，我发现大部分知识关系都没有联系（独立的一条边）。这可能是由于我选择的知识基础比较小导致的，但也可能是知识图谱prompt设计/实体消融设计不够好导致的。
请你给出可能的改进方案：

**你现在看到“大量互不连接的单边”，主要不是因为只有 10 篇文档，也暂时不能据此判断 extraction prompt 不好。当前最大的原因是：你可视化的实际上还是 Bronze mention-level graph，而真正的实体 canonicalization 尚未实现。**

仓库 README 自己也将当前实现定义为 Phase I Bronze pipeline，Silver/Gold canonicalization 仍处于 gated 状态。 更直接的是，`python -m medical_kg canonicalize` 目前只会打印“canonicalization is scaffolded...”的信息，并不会真的执行实体融合。

### 1. 当前“独立边”现象首先是数据模型造成的

这是我认为目前最重要的问题。

`complete_extraction()` 对**每一条 assertion 都重新创建两个新的 `EntityMention`**：

```text
assertion 1:
  "type 2 diabetes" -> EntityMention UUID-A
  "metformin"       -> EntityMention UUID-B

assertion 2:
  "type 2 diabetes" -> EntityMention UUID-C
  "HbA1c"           -> EntityMention UUID-D
```

即使 UUID-A 和 UUID-C 的文本都完全是 `"type 2 diabetes"`，它们也是两个 Neo4j 节点。

而你的 Neo4j `RAW_ASSERTION` 投影正是：

```text
EntityMention --RAW_ASSERTION--> EntityMention
```

并没有经过 canonical entity。

所以现在理论上：

```text
(type 2 diabetes #1) -- treated_by --> (metformin #1)

(type 2 diabetes #2) -- associated_with --> (obesity #1)

(type 2 diabetes #3) -- increases --> (HbA1c #1)
```

而不是预期的：

```text
                         -> metformin
                        /
type 2 diabetes ---------> obesity
                        \
                         -> HbA1c
```

因此，**即使换成 1000 篇论文，只要继续画 `RAW_ASSERTION` mention graph，同一个概念仍然会大量复制。**

---

## 我建议的改造优先级

| 优先级    | 改造                                       | 对连通性的预期影响 |
| ------ | ---------------------------------------- | --------- |
| **P0** | 不再使用 RAW_ASSERTION 判断最终 KG 质量            | 极高        |
| **P0** | 实现真正的 Silver Entity Resolution           | 极高        |
| **P1** | 加强 canonicalization candidate generation | 极高        |
| **P1** | 优化 extraction prompt 的 entity boundary   | 高         |
| **P1** | 实现 relation canonicalization             | 中高        |
| **P2** | 跨 pass / 跨 chunk assertion deduplication | 中         |
| **P2** | 外部 ontology / identifier mapping         | 中高        |
| **P3** | ontology hierarchy enrichment            | 中         |
| **P3** | 扩展到更多 diabetes 文档                        | 中         |

其中前两项完成之前，我甚至**不建议先大改 extraction prompt**。

---

# 2. 第一优先级：真正实现 Entity Resolution

实际上你的数据库 schema 已经设计得很好。现在已经有：

`Entity`、`EntityAlias`、`EntityExternalId` 和 canonical `Assertion` 等表。

问题只是这条 pipeline 还没有真正跑起来。

你现在的 `ConservativeEntityResolver` 实际只做：

```python
normalized_alias(mention)
```

然后检查是不是**完全相同的 normalized string**。

这只能处理：

```text
Type 2 Diabetes
type-2 diabetes
TYPE 2 DIABETES
```

但处理不了医学文本里最常见的：

```text
type 2 diabetes mellitus
T2DM
T2D
type II diabetes
patients with T2D
diabetic subjects
```

更不用说：

```text
diabetic kidney disease
DKD
diabetic nephropathy
```

所以建议将 Silver resolver 做成一个 **candidate retrieval → semantic resolution** 两阶段系统：

```text
Entity mention
     │
     ├─ exact normalized alias
     │
     ├─ known abbreviation
     │
     ├─ lexical candidate retrieval
     │
     ├─ embedding top-k retrieval
     │
     └─ external ontology alias
              │
              ▼
        Candidate entities
              │
              ▼
          LLM resolver
       MATCH existing entity
              or
           CREATE NEW
```

这里有一个很重要的原则：

**Embedding 只负责 candidate retrieval，不要直接用 embedding similarity 做 merge。**

医学实体中：

```text
T1D
T2D
LADA
diabetes mellitus
prediabetes
```

embedding 都可能非常近，但绝对不能因此合并。

---

# 3. 你的 entity canonicalization prompt 也需要明显增强

现在 prompt 是：

```text
Mention: {mention}
Type: {entity_type}
Candidates: {candidates}
```

信息明显不足。

例如：

```text
Mention: IR
Type: PHYSIOLOGICAL_PROCESS
```

LLM 根本不知道这里是 insulin resistance 还是其他缩写。

建议改成至少给：

```text
Mention
Entity type
Evidence sentence
Local context
Document title
Candidate canonical name
Candidate aliases
Candidate entity type
Candidate external IDs, if available
```

输出不要让模型自由生成 entity，而应是：

```json
{
  "decision": "MATCH",
  "entity_id": "ENT_xxx",
  "confidence": 0.97
}
```

或者：

```json
{
  "decision": "NEW",
  "canonical_name": "insulin resistance",
  "confidence": 0.94
}
```

这样 semantic identity 才能真正建立。

你当前 prompt 中：

> Prefer leaving it unresolved over an incorrect merge.

这个思想我建议**保留**。医学 KG 中 **false merge 比 false split 更危险**。

---

# 4. Extraction prompt 确实也有问题，但不是“不要 preserve exact wording”

当前 extraction prompt 强调：

> Preserve exact entity wording

这其实作为 **Bronze layer 是正确的**。我不建议直接把它改成：

> Always normalize entities.

否则会损失 evidence provenance。

正确设计应当是：

```text
Bronze
paper surface form
"patients with poorly controlled type 2 diabetes"

        ↓

Silver entity resolution

"type 2 diabetes mellitus"
```

真正需要修改的是 **entity boundary**。

例如目前 LLM 很可能生成：

```text
[patients with poorly controlled type 2 diabetes]
    --showed improved response to-->
[metformin treatment for 12 weeks]
```

这样两个节点都非常特殊，当然很难和其他论文连接。

更合理的是：

```text
type 2 diabetes
    --treated_by-->
metformin
```

同时：

```json
{
  "population": "patients with poorly controlled type 2 diabetes",
  "duration": "12 weeks"
}
```

或者另一条：

```text
metformin
    --decreases-->
HbA1c
```

把：

```text
population
dose
duration
disease_state
timepoint
study_type
```

放进 qualifier，而不是塞进 entity。

你目前 schema 本来就已经有这些 qualifier 字段。

所以 extraction prompt 应增加一个非常关键的规则：

> **Extract the minimal reusable biomedical concept as each entity. Do not include population, dose, duration, disease state, statistical result, or study context inside an entity when these can be represented as qualifiers.**

这条规则可能会显著改善最终节点复用率。

---

# 5. 三个 extraction pass 会进一步放大实体碎片化

现在默认同时运行：

```yaml
general
molecular
clinical
```

这本身没问题。

但可能得到：

```text
general:
T2DM -> associated with -> obesity

clinical:
type 2 diabetes mellitus -> positively associated with -> obesity

molecular:
type 2 diabetes -> linked to -> obesity
```

三个 assertion 都可能正确，但当前 Bronze 会产生六个甚至更多 mention node。

而且 `_unique_assertions()` 目前的 dedup 条件实际上是**整个 JSON 完全相同**。

只要 evidence、relation wording 或 mention 有一点差异，就不会 dedup。

因此 Gold 阶段应该采用：

```text
canonical subject
+
canonical relation
+
canonical object
+
semantically relevant qualifiers
```

作为事实 identity。

例如：

```text
T2D -- associated_with --> obesity
```

被三篇文章、三个 pass 抽到，不应该生成三个概念关系，而应该是：

```text
T2D
 │
 │ associated_with
 ▼
Obesity
 │
 ├ evidence: paper 1
 ├ evidence: paper 2
 ├ evidence: paper 3
 └ support_count: 3
```

这会让最终图干净很多。

---

# 6. Relation canonicalization 目前也基本没有真正工作

你的 relation vocabulary 其实已经比较合理，包括：

```text
associated_with
increases_risk_of
causes
contributes_to
treats
activates
inhibits
upregulates
expressed_in
...
```

但当前 `ExactRelationNormalizer` 只有在：

```python
detailed_relation.strip().lower().replace(" ", "_")
```

**正好等于 vocabulary 中某一个词时才匹配**，否则直接：

```text
OTHER
```

而真实 LLM extraction 极有可能输出：

```text
is associated with
was significantly associated with
showed positive correlation with
was linked to
increased the likelihood of
enhanced
promoted the expression of
```

所以必须启用你已经写好的 relation canonicalization prompt 思路。

建议保留两层：

```text
detailed_relation:
"was independently associated with an increased risk of"

canonical_relation:
"increases_risk_of"
```

这样**不需要牺牲你现在丰富的语义信息**。

甚至可以进一步利用数据库已经预留的：

```text
parent_relation_id
inverse_relation_id
```

形成：

```text
upregulates
      │
      └── parent → regulates

increases_risk_of
      │
      └── parent → associated_with
```

这对后续 KG reasoning 很有价值。

---

# 7. 不建议为了“提高连通度”强行减少实体类型

你现在的 entity schema 包括：

```text
DISEASE
PHENOTYPE
DRUG
GENE
PROTEIN
METABOLITE
PATHWAY
BIOMARKER
LAB_MEASUREMENT
...
```

整体没必要因为图稀疏就大幅压缩。

相反应该建立 **type compatibility rules**。

例如：

```text
GENE != PROTEIN
DISEASE != PHENOTYPE
DRUG != TREATMENT
```

即使名字相同，也不要默认 merge。

尤其是：

```text
INS
insulin
```

可能分别对应 gene、protein、drug/intervention。

Entity resolution 时应把：

```text
name similarity
+
entity type
+
context
+
external ID
```

一起判断。

---

# 8. 一个非常便宜但很有价值的实验

在完整实现 Silver 之前，我建议你先做一次 **temporary normalized graph**。

临时把所有 Bronze mentions 按：

```text
normalized(mention_text) + entity_type
```

collapse 后再画图。

例如：

```text
"type 2 diabetes"
"Type-2 diabetes"
"TYPE 2 DIABETES"
```

暂时视为同一个节点。

然后比较：

```text
RAW graph
vs.
exact-normalized graph
```

如果 connected components 数量大幅下降，说明问题基本就是 entity resolution。

如果几乎没下降，再检查 entity boundary 和文档 topic overlap。

这一步几乎不需要 LLM，也不需要完整改 pipeline，却能迅速告诉你问题在哪里。

---

# 9. 我建议用 ablation，而不是凭 Neo4j 图主观判断

下一轮还是用这 10 篇文章即可，不用马上扩到 100 篇。

建议依次比较：

| Variant | 处理                                                     |
| ------- | ------------------------------------------------------ |
| A       | 当前 Bronze RAW graph                                    |
| B       | + exact normalized mention merge                       |
| C       | + abbreviation / acronym resolution                    |
| D       | + semantic candidate retrieval + LLM entity resolution |
| E       | + improved minimal-entity extraction prompt            |
| F       | + relation canonicalization                            |
| G       | + external identifier / ontology resolution            |

关注的核心指标应该是：

**Singleton-edge ratio**

```text
两个端点 degree 都为 1 的 edge / 总 edges
```

**Largest connected component ratio**

```text
LCC nodes / total nodes
```

**Canonical compression ratio**

```text
entity mentions / canonical entities
```

**Cross-document reuse rate**

```text
被 >=2 篇文档引用的 canonical entities
/
全部 canonical entities
```

**Entity resolution precision**

这是最重要的一个，不能为了降低 singleton ratio 而产生过度 merge。

以及：

```text
relation OTHER ratio
duplicate canonical triple ratio
entities per document
assertions per document
```

**Connectivity 本身绝不能作为唯一优化目标。**

错误地把：

```text
T1D
T2D
LADA
prediabetes
```

全部 merge 成 `diabetes`，图会非常漂亮、非常连通，但 KG 已经失去价值。

---

# 10. 对你这个项目，我推荐的最终 pipeline

我会把当前架构从：

```text
Paper
  ↓
LLM extraction
  ↓
EntityMention ─── RawAssertion ─── EntityMention
  ↓
Neo4j
```

完善为：

```text
                        ┌─ exact aliases
                        ├─ abbreviation
                        ├─ lexical retrieval
Paper                   ├─ embedding retrieval
  │                     └─ ontology/xref
  ▼
Bronze extraction
  │
  ├── exact mention
  ├── evidence
  ├── detailed relation
  └── qualifiers
  │
  ▼
Silver Entity Resolution
  │
  ├── candidate retrieval
  └── semantic identity decision
  │
  ▼
Canonical Entity
  │
  ▼
Relation Canonicalization
  │
  ▼
Assertion Deduplication / Evidence Aggregation
  │
  ▼
Gold KG
  │
  ├── Entity ─ relation ─ Entity
  ├── aliases
  ├── external IDs
  ├── evidence[]
  └── sources[]
```

这也更符合 X-Diabetes 当前希望将疾病类型、并发症、治疗、临床表型以及分子层知识形成统一知识支撑的需求，而不是把每篇文献中的 surface mention 当作独立知识节点。

### 最关键的判断

**现阶段我不会把主要精力放在“增加文献数量”。**

我建议开发顺序是：

**先实现 Silver canonical entity → 再改 entity-boundary prompt → 再实现 relation canonicalization → 最后重新用这 10 篇做 ablation。**

如果完成前两步之后，10 篇同主题 diabetes 文献的图仍然充满 isolated edges，那时候才值得认真怀疑你的 extraction prompt 或文档知识覆盖本身。

尤其值得注意的是，你仓库目前其实已经把 `Entity / EntityAlias / EntityExternalId / Assertion / RelationType` 的数据库骨架搭好了。 **因此这不是推翻当前架构，而是把原本计划中的 Phase II 真正实现出来。**
