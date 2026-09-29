import os
import json
from typing_extensions import TypedDict
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.graph import START, END, StateGraph
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()

# ============ 项目路径统一管理（基于脚本所在目录，避免依赖 CWD）============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

KNOWN_TERMS_PATH = os.path.join(DATA_DIR, "known_terms.json")
SEARCH_CACHE_PATH = os.path.join(DATA_DIR, "search_cache.json")
GRAPH2_PATH = os.path.join(OUTPUT_DIR, "2graph.png")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



# 定义全局上下文构建的状态
class GlobalContextState(TypedDict, total=False):
    # 输入
    video_title: str
    video_summary: str
    all_danmaku_text: str  # 将一集弹幕拼接后的长文本（先去重）

    # 中间状态
    unknown_terms: list[str]  # 提取出的生词/黑话
    search_results: dict[str, str]  # 搜索结果缓存
    loop_count: int  # 循环次数控制，防止无限死循环
    failed_terms: dict[str, str]   #更精准的搜索词
    verified_terms: list[str]      # 人工词典命中的词，下游跳过评估
    # 输出
    global_context: str        # 全量版（给 DeepSeek）
    short_global_context: str  # 精简版（给 edgejev)

from pydantic import BaseModel, Field

class TermExtraction(BaseModel):
    """提取生词的结构化输出"""
    summary: str = Field(description="用一句话总结这集视频的剧情走向")
    unknown_terms: list[str] = Field(description="弹幕中你不理解的、最关键的几个网络热梗/黑话/专有名词")
class TermEvaluation(BaseModel):
    term: str = Field(description="被评估的生词")
    is_accurate: bool = Field(description="搜索结果是否准确解释了该词在动漫弹幕语境下的含义")
    better_query: str | None = Field(default=None, description="如果结果不准确，给出更精准的搜索词")

class SearchEvaluation(BaseModel):
    evaluations: list[TermEvaluation] = Field(description="每个生词的评估结果")

class ContextOutput(BaseModel):
    """最终上下文的结构化输出：同时生成详细版和精简版"""
    full_context: str = Field(
        description="给大模型用的详细上下文，800-1000字，包含剧情梗概、所有黑话/梗的详细释义、观众情绪倾向"
    )
    short_context: str = Field(
        description="给小模型用的极简上下文，150字以内，只要剧情一句话概括+最核心的2-3个梗名即可"
    )

class GlobalContextBuilder:
    def __init__(self, model_name: str = "deepseek-flash"):
        self.llm = ChatDeepSeek(
            model=model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0
        )
        self.structured_llm = self.llm.with_structured_output(TermExtraction,method="json_mode")#绑定模型
        self.eval_llm = self.llm.with_structured_output(SearchEvaluation,method="json_mode")#搜索模型
        self.context_llm = self.llm.with_structured_output(ContextOutput,method="json_mode")#总结模型
        self.max_loops = 2  # 最大 ReAct 循环次数，防止卡死

        # 定义搜索工具
        @tool
        def search_web(query: str) -> str:
            """当遇到你不理解的网络热梗、黑话或专有名词时，调用此工具查询。"""
            # 方使用 Tavily API
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True
                }
                try:
                    response = requests.post(url, json=payload, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("answer", str(data.get("results", "无结果")))
                except Exception as e:
                    return f"Tavily 搜索失败: {e}"

            # 如果都没配置，返回明确的提示，而不是假数据
            return "【系统提示】未配置搜索引擎 API。请在 .env 文件中配置 TAVILY_API_KEY 或 SERPER_API_KEY。"

        self.tools = [search_web]
        self.llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="auto")
        self._build_graph()

    # -------- 节点 1：提取生词 --------
    def _extract_terms_node(self, state: GlobalContextState):
        print("   [1/3] 正在调用 LLM 总结剧情并提取生词...")
        prompt = f"""
        请阅读以下视频简介和观众弹幕，完成三件事：
        1. 用一句话总结这集视频的大致剧情走向。
        2. 找出弹幕中你不理解的、最关键的几个网络热梗/黑话/专有名词（不要超过10个！）。
        3. 请你先尝试用你自己的知识猜测这些词的含义（不用100%准确，后续会验证）。

        【视频标题】{state.get('video_title', '')}
        【视频简介】{state.get('video_summary', '')}
        【弹幕节选（已去重）】{state.get('all_danmaku_text', '')[:3000]}...

        请以 JSON 格式输出，字段为 summary 和 unknown_terms。
        """
        try:
            # 直接调用结构化模型，返回的已经是 Pydantic 对象了，无需 json.loads
            response = self.structured_llm.invoke([HumanMessage(content=prompt)])

            return {
                "video_summary": response.summary,  # 直接用点号访问属性
                "unknown_terms": response.unknown_terms
            }
        except Exception as e:
            # 必须把错误打印出来，绝对不能静默失败！
            print(f"   ❌ 提取生词失败: {e}")
            return {
                "unknown_terms": []
            }

    # -------- 节点 2：批量搜索生词 --------
    def _search_terms_node(self, state: GlobalContextState):
        unknown_terms = state.get("unknown_terms", [])
        search_results = state.get("search_results", {})
        verified = set(state.get("verified_terms", []))

        # 过滤掉已经在 search_results 里的词
        terms_to_search = [t for t in unknown_terms if t not in search_results]
        if not terms_to_search:
            print("   [2/5] 无新增生词，跳过搜索")
            return {"search_results": search_results, "verified_terms": list(verified)}

        print(f"   [2/5] 开始并行搜索 {len(terms_to_search)} 个生词...")

        # ================== 1. 加载人工词典 ==================
        known_terms = {}
        if os.path.exists(KNOWN_TERMS_PATH):
            try:
                with open(KNOWN_TERMS_PATH, "r", encoding="utf-8") as f:
                    known_terms = json.load(f)
            except Exception as e:
                print(f"       ⚠️ known_terms.json 读取失败: {e}")

        # ================== 2. 加载搜索缓存 ==================
        cache_file = SEARCH_CACHE_PATH
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        video_title = state.get('video_title', '')
        series_name = video_title.split(" 第")[0] if " 第" in video_title else video_title

        # ================== 3. 三层查询逻辑 ==================
        def fetch_term(term):
            # 【优先级 1】人工词典 → 已验证，跳过评估
            if term in known_terms:
                return term, known_terms[term], 0.0, "📖 人工词典", True

            # 【优先级 2】本地缓存 → 待评估
            if term in cache:
                return term, cache[term], 0.0, "⚡ 缓存", False

            # 【优先级 3】Tavily 实时搜索 → 待评估
            start = time.perf_counter()
            search_query = f"{series_name}动漫小说弹幕里的'{term}'是什么意思"
            res = ""
            for tool in self.tools:
                if tool.name == "search_web":
                    res = tool.invoke(search_query)
            duration = time.perf_counter() - start
            cache[term] = res
            return term, res, duration, "🔍 Tavily", False

        # ================== 4. 并行执行 ==================
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_term, term): term for term in terms_to_search}
            for future in as_completed(futures):
                term, res, duration, source, is_verified = future.result()
                search_results[term] = res
                if is_verified:
                    verified.add(term)
                if duration == 0.0:
                    print(f"       {source}命中 [{term}] (0秒)")
                else:
                    print(f"       {source}搜索 [{term}] 耗时: {duration:.2f} 秒")

        # ================== 5. 回写缓存（只写 Tavily 新搜的）==================
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"       ⚠️ 写入缓存失败: {e}")

        return {
            "search_results": search_results,
            "verified_terms": list(verified)   # 传给下游：这些词跳过评估
        }
    # -------- 节点 3：审阅搜索结果的质量 --------
    def _evaluate_search_node(self, state: GlobalContextState):
            print("   [3/5] 正在审阅搜索结果的质量...")
            search_results = state.get("search_results", {})
            verified = set(state.get("verified_terms", []))

            if not search_results:
                return {"failed_terms": {}, "verified_terms": list(verified)}

            # 只评估不在 verified 里的词
            terms_to_eval = {k: v for k, v in search_results.items() if k not in verified}
            if not terms_to_eval:
                print("       ⏭️ 所有生词均已验证，跳过评估")
                return {"failed_terms": {}, "verified_terms": list(verified)}

            print(f"       需要评估 {len(terms_to_eval)} 个词（{len(verified)} 个已命中人工词典）")

            # 【关键】prompt 里只传 terms_to_eval，不要传 search_results
            prompt = f"""
        你是动漫《{state.get('video_title', '')}》的资深观众。
        请审阅以下搜索结果，判断它们是否准确解释了该词在【该动漫的弹幕语境】下的含义。

        【需要评估的搜索结果】
        {json.dumps(terms_to_eval, ensure_ascii=False, indent=2)}

        判断标准：
        - 如果搜索结果解释的是"该动漫的剧情/人物"，而不是"弹幕黑话的含义"，则判为不准确。
        - 如果结果明显错误（如把"黑皇"搜成了"别的动漫角色"），也判为不准确。
        - 如果结果不准确，请给出一个更精准的搜索词。

        请以 JSON 格式返回，包含 evaluations 数组。
        """
            try:
                response = self.eval_llm.invoke([HumanMessage(content=prompt)])
                failed = {}
                accurate = {}

                for ev in response.evaluations:
                    # 只处理我们真正评估过的词
                    if ev.term not in terms_to_eval:
                        continue
                    if not ev.is_accurate and ev.better_query:
                        failed[ev.term] = ev.better_query
                        print(f"       ⚠️ [{ev.term}] 结果不准确，将用新词重搜")
                    else:
                        print(f"       ✅ [{ev.term}] 结果准确")
                        accurate[ev.term] = terms_to_eval[ev.term]

                # ============ 1. 评估通过的词写入人工词典 ============
                if accurate:
                    known = {}
                    if os.path.exists(KNOWN_TERMS_PATH):
                        try:
                            with open(KNOWN_TERMS_PATH, "r", encoding="utf-8") as f:
                                known = json.load(f)
                        except Exception:
                            known = {}
                    known.update(accurate)
                    try:
                        with open(KNOWN_TERMS_PATH, "w", encoding="utf-8") as f:
                            json.dump(known, f, ensure_ascii=False, indent=2)
                        print(f"       📚 已自动将 {len(accurate)} 个词写入人工词典")
                    except Exception as e:
                        print(f"       ⚠️ 写入人工词典失败: {e}")

                # ============ 2. 【关键】评估通过的词加入 verified_terms ============
                verified.update(accurate.keys())

                return {
                    "failed_terms": failed,
                    "verified_terms": list(verified)  # ← 这是之前漏掉的关键返回
                }
            except Exception as e:
                print(f"   ❌ 评估失败: {e}")
                return {"failed_terms": {}, "verified_terms": list(verified)}


    def _retry_search_node(self, state: GlobalContextState):
        failed_terms = state.get("failed_terms", {})
        search_results = state.get("search_results", {})

        # 【新增】本轮重搜的轮次（每次重搜 +1）
        new_loop = state.get("loop_count", 0) + 1

        if not failed_terms:
            print("   [重搜] 没有需要重搜的词，直接进入生成阶段")
            return {"failed_terms": {}, "loop_count": new_loop}

        print(f"   [重搜] 正在用更精准的词重搜 {len(failed_terms)} 个生词（第 {new_loop} 轮）...")

        # 并行重搜
        def fetch(term, query):
            start = time.perf_counter()
            res = ""
            for tool in self.tools:
                if tool.name == "search_web":
                    res = tool.invoke(query)
            return term, res, time.perf_counter() - start

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch, term, q): term for term, q in failed_terms.items()}
            for future in as_completed(futures):
                term, res, duration = future.result()
                search_results[term] = res  # 覆盖旧结果
                print(f"       🔄 重搜 [{term}] 耗时: {duration:.2f} 秒")

        # 【新增】把重搜后的结果同步进缓存
        cache = {}
        if os.path.exists(SEARCH_CACHE_PATH):
            try:
                with open(SEARCH_CACHE_PATH, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        for term in failed_terms:
            cache[term] = search_results[term]
        try:
            with open(SEARCH_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"       ⚠️ 写入缓存失败: {e}")

        # 清空 failed_terms（让评估节点重新判断），并递增 loop_count
        return {
            "search_results": search_results,
            "failed_terms": {},
            "loop_count": new_loop
        }

    # -------- 节点 5：生成最终上下文 --------
    def _finalize_context_node(self, state: GlobalContextState):
        print("   [3/3] 正在生成双版本全局上下文（详细版 + 精简版）...")

        # 把搜索结果从 JSON 字符串转成自然语言，避免模型直接复读
        search_results = state.get('search_results', {})
        terms_text = "\n".join([f"- {term}：{expl}" for term, expl in search_results.items()]) if search_results else "无"

        prompt = f"""
        你是一位《{state.get('video_title', '')}》的资深内容审核专家。请基于以下信息，生成两份弹幕审核用的背景上下文。

        【视频标题】{state.get('video_title', '')}
        【剧情总结】{state.get('video_summary', '')}
        【黑话/梗的释义】
        {terms_text}

        【弹幕样本节选】
        {state.get('all_danmaku_text', '')[:3000]}

        请输出两份上下文：
        1. full_context（详细版，800-1000字）：
           - 先一句话概括剧情。
           - 然后逐条解释弹幕中出现的黑话/梗的真实含义（注意纠正搜索结果的偏差，如果某个词的解释明显违背常识，请用你的知识修正）。
           - 最后总结观众情绪的整体倾向（夸赞/吐槽/中立），并指出哪些内容属于正常互动、哪些可能涉嫌违规。

        2. short_context（精简版，150字以内）：
           - 用一句话概括剧情。
           - 只用列出最重要的 2-3 个黑话及其一句话释义。
           - 一句话总结整体氛围。

        请以 JSON 格式返回，包含 full_context 和 short_context 两个字段。
        """
        try:
            response = self.context_llm.invoke([HumanMessage(content=prompt)])
            print(f"       ✅ 详细版长度: {len(response.full_context)} 字")
            print(f"       ✅ 精简版长度: {len(response.short_context)} 字")
            return {
                "global_context": response.full_context,
                "short_global_context": response.short_context
            }
        except Exception as e:
            print(f"   ❌ 生成上下文失败: {e}")
            # 兜底：用简单拼接
            fallback = f"视频标题：{state.get('video_title')}\n剧情：{state.get('video_summary')}\n黑话：{terms_text}"
            return {
                "global_context": fallback,
                "short_global_context": fallback[:150]
            }

    # -------- 路由函数：决定是否需要再次搜索 --------
    def _route_after_extract(self, state: GlobalContextState):
        # 如果有生词，且循环次数未超限，去搜索
        if state.get("unknown_terms") and state.get("loop_count", 0) <= self.max_loops:
            return "search_terms_node"
        # 否则直接去生成总结
        return "finalize_context_node"

    def _route_after_evaluate(self, state: GlobalContextState):
        # 只要还有搜错的词，且循环次数未超限，就继续重搜
        if state.get("failed_terms") and state.get("loop_count", 0) < self.max_loops:
            return "retry_search_node"
        return "finalize_context_node"
    # -------- 构图 --------
    def _build_graph(self):
        builder = StateGraph(GlobalContextState)

        builder.add_node("extract_terms_node", self._extract_terms_node)
        builder.add_node("search_terms_node", self._search_terms_node)
        builder.add_node("evaluate_search_node", self._evaluate_search_node)  # 新增
        builder.add_node("retry_search_node", self._retry_search_node)  # 新增
        builder.add_node("finalize_context_node", self._finalize_context_node)

        builder.add_edge(START, "extract_terms_node")

        builder.add_conditional_edges(
            "extract_terms_node",
            self._route_after_extract,
            {
                "search_terms_node": "search_terms_node",
                "finalize_context_node": "finalize_context_node"
            }
        )

        # 【ReAct 关键】搜索完 → 评估 → 决定重搜或总结
        builder.add_edge("search_terms_node", "evaluate_search_node")
        builder.add_conditional_edges(
            "evaluate_search_node",
            self._route_after_evaluate,
            {
                "retry_search_node": "retry_search_node",
                "finalize_context_node": "finalize_context_node"
            }
        )
        builder.add_edge("retry_search_node", "evaluate_search_node")  # 【回边】重搜后再评估

        builder.add_edge("finalize_context_node", END)

        self.graph = builder.compile()
        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            with open(GRAPH2_PATH, "wb") as f:
                f.write(png_data)
        except Exception:
            pass
    def build_context(self, video_title: str, video_summary: str, all_danmaku_text: str) -> dict:
        """运行全局上下文构建流程，返回 {'full': 详细版, 'short': 精简版}"""
        init_state = {
            "video_title": video_title,
            "video_summary": video_summary,
            "all_danmaku_text": all_danmaku_text,
            "loop_count": 0
        }
        result = self.graph.invoke(init_state)
        return {
            "full": result.get("global_context", ""),
            "short": result.get("short_global_context", "")
        }