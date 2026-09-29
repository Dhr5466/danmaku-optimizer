# data/ 目录说明

这个目录里放的是**运行时数据**和**测试输入**，两者在 Git 里的处理方式不同。

## 提交到 Git

| 文件 | 说明 |
|---|---|
| `test.xml` | 弹幕 XML 输入样本。根目录还有一份完全相同的拷贝，`test.py` 实际读的是**根目录**那份（见 `test.py:15`），这份是保留备份。 |

## 不提交到 Git（本地需要，可再生成）

| 文件 | 说明 | 如何生成 |
|---|---|---|
| `global_context_result.json` | 预热阶段的产物。`test.py` 启动第一件事就是读它，字段：`video_title` / `video_summary` / `full_context`（给 DeepSeek）/ `short_context`（给 edgejev）。**缺了它 `test.py` 会直接 FileNotFoundError。** | `python test2.py`（调用 DeepSeek + Tavily，会覆盖此文件） |
| `known_terms.json` | 人工词典。命中它的词会被标记为「已验证」，跳过 LLM 评估与搜索。 | 手工维护；`test2.py` 的评估节点也会把确认准确的词自动追加进来 |
| `search_cache.json` | Tavily 搜索缓存，避免同一个词反复联网搜索。 | 首次跑 `test2.py` 的搜索节点时自动创建 |

> 这三个文件都可以安全地删掉重跑，**不含任何密钥**。
> 如果你希望别人 clone 下来就能直接跑 `python test.py`，把 `.gitignore` 里
> `data/known_terms.json` 和 `data/global_context_result.json` 两行注释掉再提交即可。
