# OpenAlex batch 000

本批次包含 `batch_0.txt` 中列出的 15 个 `data/works/updated_date=*` 目录。批次阶段只做
关键词/source 确定性筛选，结果持续写入服务器上的公共 catalog；暂不逐批执行 LLM
筛选。

## 1. 下载到服务器

```bash
BATCH=/home/bml/storage/mnt/v-vmkfid4oobb3c0qh/xdiabetes/openalex/batch_000
REMOTE=user@ip:/volume1/X-Diabetes/openalex/openalex-snapshot/data/works

mkdir -p "$BATCH/data/works"

scp -P 22222 -r \
  "${REMOTE}/updated_date=2016-*" \
  "$BATCH/data/works/"
```

将 `user@ip` 改为实际值。该命令只建立一次 SSH 连接，因此只需认证一次。不要复制
`legacy-data` 或异常后缀文件。

## 2. 简单检查

```bash
find "$BATCH/data/works" -mindepth 1 -maxdepth 1 -type d | sort
find "$BATCH/data/works" -type f -name 'part_[0-9][0-9][0-9][0-9].gz' | wc -l
du -sh "$BATCH"
```

应看到 15 个 `updated_date=*` 目录。

## 3. 执行批次筛选

```bash
python openalex_pipeline.py select "$BATCH" \
  --keyword "<关键词1>" \
  --keyword-mode any \
  --workspace data/openalex
```

按实际任务替换关键词；需要 source 限制时追加 `--source "<source 名称或 ID>"`。所有批次
必须使用同一个 `--workspace`，以复用 `catalog.sqlite3` 的 Work ID 幂等更新能力。

确认命令输出无失败且 `data/openalex/catalog.sqlite3` 已更新后，才可清理
`batch_000` 临时目录。实际正文和知识关系抽取应在候选批次收集完成、统一筛选后执行。
