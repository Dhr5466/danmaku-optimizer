# 上传到 GitHub 指南（UPLOAD.md）

本文件说明本仓库**该提交什么、不该提交什么**，以及**如何把当前代码推到 GitHub**。

---

## 一、文件分类总览

### ✅ 必须提交（源码 / 配置模板 / 测试输入）

| 文件 | 大小 | 作用 |
|---|---|---|
| `danmaku_filter.py` | 15 KB | 审核主逻辑（LangGraph 图 + 本地模型 + DeepSeek） |
| `global_context_builder.py` | 22 KB | 全局上下文预热（Tavily 搜索 + 双版本上下文） |
| `test.py` | 6 KB | 入口脚本：读上下文 → 逐条审核 → 导出 CSV |
| `test2.py` | 4 KB | 入口脚本：预热阶段，生成 `global_context_result.json` |
| `config.json` | 385 B | 视频标题/简介/输出文件名配置 |
| `test.xml` | 650 KB | 弹幕输入样本（`test.py` 与 `test2.py` 都依赖它） |
| `requirements.txt` | 8 KB | 依赖锁定 |
| `.env.example` | 76 B | 环境变量模板（**不含真实 key**） |
| `.gitignore` | — | 忽略规则 |
| `README.md` / `UPLOAD.md` / `data/README.md` | — | 文档 |

### ❌ 绝对不能提交

| 文件/目录 | 原因 |
|---|---|
| `.env` | **含真实 API Key**，泄露后必须立刻在平台重新生成 |
| `jev-int8/`（`model.onnx` 324 MB、`tokenizer.json` 34 MB、`edgejev.json`） | 体积过大，GitHub 单文件上限 100 MB，直接 push 会被拒 |
| `output/`（`1graph.png`、`danmaku_review_results.csv`） | 运行时产物，每次跑都不一样 |
| `__pycache__/`、`*.pyc` | Python 字节码缓存 |
| `.idea/` | PyCharm 个人配置，含本机路径 |
| `backup/` | 临时备份 |

### ⚠️ 本地需要、但不提交（换机后能再生成）

| 文件 | 为什么还没提交 | 怎么再生成 |
|---|---|---|
| `data/global_context_result.json` | 由 LLM 生成的预热结果 | 跑 `python test2.py`（需 `DEEPSEEK_API_KEY` + `TAVILY_API_KEY`） |
| `data/known_terms.json` | 人工词典，运行中会被自动扩写 | 手工维护；或跑 `test2.py` 时由 `_evaluate_search_node` 自动写入 |
| `data/search_cache.json` | 搜索缓存，首次预热时创建 | 跑 `test2.py` 时自动生成 |
| `jev-int8/` | 见上 | 用 `edgejev build` 重新生成，或改 `danmaku_filter.py:28` 的 `DEFAULT_MODEL_PATH` 指向模型所在目录 |

> 若你希望「clone 下来就能直接跑 `python test.py`」，就把 `.gitignore` 里
> `data/known_terms.json`、`data/global_context_result.json` 两行注释掉再提交
> （两个文件合计约 9 KB，不含密钥，风险很低）。

### 🗑️ 已删除的多余文件

| 文件 | 判定依据 |
|---|---|
| `data/test.xml` | 与根目录 `test.xml` **哈希完全相同**（同一份文件两份拷贝），且 `config.json`/`test.py` 只引用根目录那份 |
| `prompts.json.py` | 扩展名是 `.py` 但内容是 JSON，且**全项目没有任何地方 import 或读取它**；其中的 prompt 早已内联进 `global_context_builder.py`。需要时可从 git 历史（提交 `67b4ef6`）取回 |
| `LangGragh.py` | 早已被 git 标记为删除（`D`），本次提交一并生效 |

---

## 二、上传前自检

```powershell
cd D:\pythondaima\danmaku-optimizer

# 1. 确认 .env 被忽略（必须输出一行 .env，说明忽略生效）
git check-ignore -v .env

# 2. 确认没有大文件被误加进暂存区（列表里不应出现 jev-int8 或 output）
git status --short

# 3. 确认待提交文件里没有密钥
git diff --cached | Select-String "sk-"
```

如果第 3 步有任何输出，**立刻停止**，说明密钥被写进了代码。

---

## 三、推送到 GitHub（逐步复制即可）

当前状态：分支 `main`，远程已指向 `https://github.com/Dhr5466/danmaku-optimizer.git`，
本地领先远程 0 个提交、有若干未提交改动。

### 第 0 步：在 GitHub 网页上建好空仓库

打开 https://github.com/new ，填仓库名（例如 `danmaku-optimizer`），
**不要勾选** "Add a README file" / ".gitignore" / "license"（避免和本地冲突），
创建后拿到仓库地址，形如：

- HTTPS：`https://github.com/<你的用户名>/danmaku-optimizer.git`
- SSH：`git@github.com:<你的用户名>/danmaku-optimizer.git`

### 第 1 步：确认远程地址

```powershell
cd D:\pythondaima\danmaku-optimizer
git remote -v
```

如果显示的**不是**你自己的仓库，改成你的：

```powershell
git remote set-url origin https://github.com/<你的用户名>/danmaku-optimizer.git
```

### 第 2 步：提交本次改动

```powershell
# 暂存全部改动（.gitignore 会自动挡掉 .env / jev-int8 / output）
git add -A

# 再看一眼到底要提交哪些文件（这一步很重要，逐行确认）
git status --short

# 提交
git commit -m "fix: 路径硬编码改为基于脚本目录的绝对路径；清理冗余文件；补充 requirements 与 .env.example"
```

### 第 3 步：推送

```powershell
git push origin main
```

### 第 4 步：验证

刷新 GitHub 仓库页面，应能看到 `danmaku_filter.py` 等源码，
且**看不到** `.env`、`jev-int8/`、`output/`。

---

## 四、常见问题

**Q1：`git push` 报 `Support for password authentication was removed`**
GitHub 已不支持账号密码推送，改用 Personal Access Token：

1. 打开 https://github.com/settings/tokens → Generate new token (classic)
2. 勾选 `repo` 权限，生成后**复制一次**（页面刷新就看不到了）
3. 推送时账号填 GitHub 用户名，密码处**粘贴 token**

**Q2：`git push` 报 `remote contains work that you do not have locally`**
远程有本地没有的提交（通常是在网页上建仓库时勾了 README）。先合并再推：

```powershell
git pull --rebase origin main
git push origin main
```

**Q3：误把 `.env` 提交了怎么办？**
提交历史里的密钥永远删不干净，**正确做法是先去平台作废并重新生成 key**，然后再清理：

```powershell
git rm --cached .env
git commit -m "chore: 移除误提交的 .env"
git push origin main
```

**Q4：`jev-int8/` 被误加进暂存区了**
从暂存区移除（本地文件保留）：

```powershell
git rm -r --cached jev-int8
git commit -m "chore: 停止跟踪本地模型目录"
```

**Q5：换了台机器，怎么跑起来？**
```powershell
git clone https://github.com/<你的用户名>/danmaku-optimizer.git
cd danmaku-optimizer
pip install -r requirements.txt
copy .env.example .env      # 然后填入真实 key
# 准备模型目录 jev-int8/（edgejev build 生成，或改 DEFAULT_MODEL_PATH 指向已有模型）
python test2.py             # 预热：生成 data/global_context_result.json
python test.py              # 审核
```

---

## 五、已知遗留问题（不影响上传）

1. **`test2.py` + `config.json` 仍用相对路径**：从非项目根目录运行会 `FileNotFoundError`，
   且输出会写到当前工作目录。`test.py` 已修好，`test2.py` 未在本次范围内。
2. **`test.xml` 有两份拷贝**：根目录那份是实际引用的，`data/test.xml` 已删除。
   若你更想保留 `data/` 那份，需要同时改 `test.py:15` 和 `config.json:4`。
3. **`requirements.txt` 是整机 `pip freeze` 快照**（204 行），包含与项目无关的包（如 `anthropic`、
   `torch` 相关）。别人 `pip install -r requirements.txt` 会装一堆用不上的东西。
   建议后续精简为直接依赖。
