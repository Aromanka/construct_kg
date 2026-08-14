# Linux 服务器迁移与断点续跑

## 1. 如何无损迁移运行结果

本项目的权威运行结果和断点状态都在 PostgreSQL 中，包括文档、抽取结果及
`processing_jobs`。当前项目**没有独立的 `checkpoints/` 目录**，因此完整迁移数据库，
再使用相同版本的代码、配置和抽取参数，即可继续运行。

### 源服务器

先停止所有抽取进程，避免迁移时仍有任务写入，然后备份数据库：

```bash
python -m medical_kg status --config config.yaml
python -m medical_kg stats --config config.yaml

pg_dump -h localhost -U postgres -d medical_kg \
  -Fc --no-owner --no-acl -f medical_kg.dump
sha256sum medical_kg.dump > medical_kg.dump.sha256
```

将以下内容传到目标服务器：

- `medical_kg.dump` 和校验文件；
- 同一 Git commit 的项目代码；
- `config.yaml`（含密钥，建议安全传输）或在目标机重新创建；
- 原始 `data/knowledge_base/` 可一并迁移，用于日后重新导入或核对；已入库文档的
  正文和哈希已在数据库中，仅续跑抽取时不依赖原 PDF。

例如：

```bash
rsync -a --info=progress2 medical_kg.dump medical_kg.dump.sha256 \
  user@target:/path/to/construct_kg/
```

### 目标服务器

建议使用与源服务器相同 PostgreSQL 大版本；目标版本也可以更高，但应使用目标机的
`pg_restore`。在项目目录执行：

```bash
sha256sum -c medical_kg.dump.sha256

createdb -h localhost -U postgres medical_kg
pg_restore -h localhost -U postgres -d medical_kg \
  --no-owner --no-acl --exit-on-error --single-transaction medical_kg.dump

conda create -n medical-kg python=3.10 -y
conda activate medical-kg
python -m pip install -e ".[pdf]"
export DEEPSEEK_API_KEY='...'
export POSTGRES_PASSWORD='...'

# Idempotently apply any schema additions introduced by the restored code version.
python -m medical_kg init-db --config config.yaml
```

检查 `config.yaml` 中数据库地址，并保持 `extraction.stage_version`、prompts、模型名称、
`--chunk-size` 和 `--chunk-overlap` 与迁移前一致。不要直接复制原服务器的 Conda 环境目录，
应在目标机创建环境并重新安装依赖。

源进程被强制中断时，数据库中可能遗留 `RUNNING` 任务。确认源服务器已不再运行后，可在
目标机立即将它们重新排队：

```bash
psql -h localhost -U postgres -d medical_kg -c \
  "UPDATE processing_jobs SET status='PENDING', worker_id=NULL, started_at=NULL, \
   heartbeat_at=NULL, lease_expires_at=NULL, finished_at=NULL, \
   error_message='Recovered after server migration' \
   WHERE status='RUNNING';"
```

随后核对并续跑。下面的分块参数应替换为迁移前实际使用的值：

```bash
python -m medical_kg status --config config.yaml
python -m medical_kg stats --config config.yaml
python -m medical_kg extract --chunk-size 12000 --chunk-overlap 500 --config config.yaml
```

`SUCCESS` 任务会自动跳过；被中断且未提交的单个抽取 pass 会重新执行，不会留下半条结果。
如需重跑原有 `FAILED` 任务，先执行：

```bash
python -m medical_kg retry-failed --config config.yaml
```

如果未来增加真正的文件型 `checkpoints/`，应在停止写入后用
`rsync -aH --checksum checkpoints/ user@target:/path/to/project/checkpoints/` 同步，并保留
目录结构、文件权限和符号链接；不要在进程写 checkpoint 时直接复制。

## 2. Linux 上的注意事项

- 项目统一使用 Conda 环境：先执行 `conda activate medical-kg`。若当前 Shell 尚未启用
  Conda，可先运行 `conda init bash`，重新登录后再激活。
- 在 `systemd` 等非交互环境中不建议依赖 `conda activate`，可直接使用
  `conda run -n medical-kg python -m medical_kg ...`。
- 环境变量使用 `export NAME=value`；建议通过服务环境文件或密钥管理器保存，不要提交
  `config.yaml` 和 API 密钥。
- Linux 文件名区分大小写，请保持 `config/`、`prompts/` 和数据路径的大小写一致，并确保
  运行用户对项目目录有读权限、对所需输出目录有写权限。
- 长时间抽取建议使用 `systemd`、`tmux` 或 `screen`，避免 SSH 断开直接终止进程。
