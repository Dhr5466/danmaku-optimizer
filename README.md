# 弹幕审核优化系统 (Danmaku Optimizer)

基于 LangGraph + 本地轻量模型 (edgejev) + 云端大模型 (DeepSeek) 的双层弹幕内容安全审核系统。

## 核心架构
1. **全局上下文预热 (GlobalContextBuilder)**：提取弹幕黑话/梗，通过 Tavily 搜索并评估，生成双版本上下文（详细版 + 精简版）。
2. **Lite 直过**：本地轻量模型高频拦截简单安全内容，耗时 <150ms。
3. **LLM 复核**：结合全局上下文，对模糊弹幕进行二次判定（支持 JSON 模式结构化输出）。

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env`：填入 `DEEPSEEK_API_KEY` 和 `TAVILY_API_KEY`
3. 运行：`python test.py`
