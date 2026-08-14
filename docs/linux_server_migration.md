# Linux 服务器迁移与断点续跑

## 1. SQLite 数据文件

本项目的权威运行结果和断点状态都在一个 SQLite 文件中，包括文档、抽取结果和
`processing_jobs`。默认配置为：

```yaml
database:
  path: data/medical_kg.sqlite3
  timeout: 30
  echo: false
```

相对路径以 `config.yaml` 所在目录为基准。程序会自动创建父目录，并为每个数据库连接
启用外键约束、WAL 模式和 busy timeout。数据库应放在服务器本地磁盘，不建议直接放到
NFS、SMB 等网络文件系统。

## 2. 如何无损迁移运行结果

### 源服务器

先停止所有抽取进程，确保不再写入数据库。然后使用 SQLite 在线备份命令生成一致快照：

```bash
python -m medical_kg status --config config.yaml
python -m medical_kg stats --config config.yaml

sqlite3 data/medical_kg.sqlite3 ".backup 'medical_kg.backup.sqlite3'"
sqlite3 medical_kg.backup.sqlite3 "PRAGMA integrity_check;"
sha256sum medical_kg.backup.sqlite3 > medical_kg.backup.sqlite3.sha256
```

`PRAGMA integrity_check` 应输出 `ok`。将以下内容传到目标服务器：

- `medical_kg.backup.sqlite3` 和校验文件；
- 同一 Git commit 的项目代码；
- `config.yaml`，或在目标机根据 `config.example.yaml` 重新创建；
- 原始 `data/knowledge_base/` 可一并迁移，用于日后重新导入或核对。已入库文档的正文和
  哈希已经保存在 SQLite 中，仅续跑抽取时不依赖原 PDF。

例如：

```bash
rsync -a --info=progress2 medical_kg.backup.sqlite3 medical_kg.backup.sqlite3.sha256 \
  user@target:/path/to/construct_kg/
```

### 目标服务器

在项目目录执行：

```bash
sha256sum -c medical_kg.backup.sqlite3.sha256

mkdir -p data
cp medical_kg.backup.sqlite3 data/medical_kg.sqlite3
sqlite3 data/medical_kg.sqlite3 "PRAGMA integrity_check;"

conda create -n medical-kg python=3.10 -y
conda activate medical-kg
python -m pip install -e ".[pdf]"
export DEEPSEEK_API_KEY='...'

python -m medical_kg init-db --config config.yaml
```

检查 `config.yaml` 中的 `database.path`，并保持 `extraction.stage_version`、prompts、模型
名称、`--chunk-size` 和 `--chunk-overlap` 与迁移前一致。不要直接复制原服务器的 Conda
环境目录，应在目标机创建环境并重新安装依赖。

源进程被强制中断时，数据库中可能遗留 `RUNNING` 任务。确认源服务器已不再运行后，可在
目标机将它们重新排队：

```bash
sqlite3 data/medical_kg.sqlite3 \
  "UPDATE processing_jobs SET status='PENDING', worker_id=NULL, started_at=NULL, \
   heartbeat_at=NULL, lease_expires_at=NULL, finished_at=NULL, \
   error_message='Recovered after server migration' WHERE status='RUNNING';"
```

随后核对并续跑。下面的分块参数应替换为迁移前实际使用的值：

```bash
python -m medical_kg status --config config.yaml
python -m medical_kg stats --config config.yaml
python -m medical_kg extract --chunk-size 12000 --chunk-overlap 500 --config config.yaml
```

`SUCCESS` 任务会自动跳过；被中断且未提交的单个抽取 pass 会重新执行，不会留下半条结果。
如需重跑原有 `FAILED` 任务，执行：

```bash
python -m medical_kg retry-failed --config config.yaml
```

## 3. 直接复制数据库时的注意事项

只有在所有写入进程已停止后，才可以直接复制主数据库文件。WAL 模式运行期间可能同时存在
`medical_kg.sqlite3-wal` 和 `medical_kg.sqlite3-shm`；只复制主文件可能遗漏尚未 checkpoint
的数据。因此优先使用 `.backup`，或者停进程后执行：

```bash
sqlite3 data/medical_kg.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"
```

确认 `-wal` 文件已被合并后再复制主文件。

## 4. Linux 上的注意事项

- 项目统一使用 Conda 环境：先执行 `conda activate medical-kg`。若当前 Shell 尚未启用
  Conda，可先运行 `conda init bash`，重新登录后再激活。
- 在 `systemd` 等非交互环境中不建议依赖 `conda activate`，可直接使用
  `conda run -n medical-kg python -m medical_kg ...`。
- 确保运行用户对项目、数据库目录和 SQLite 文件有读写权限。
- SQLite 允许并发读取，但同一时刻只有一个写事务；WAL 和 busy timeout 会等待短暂锁竞争，
  仍不建议从多台服务器同时写同一个文件。
- Linux 文件名区分大小写，请保持 `config/`、`prompts/` 和数据路径的大小写一致。
- 长时间抽取建议使用 `systemd`、`tmux` 或 `screen`，避免 SSH 断开直接终止进程。
