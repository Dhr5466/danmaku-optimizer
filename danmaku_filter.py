import os
import sqlite3
import time
import urllib.request
from typing import Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import START, END, StateGraph
#from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver   # 替换 SqliteSave
from langgraph.types import interrupt

from edgejev import Agent

load_dotenv()

# ============ 项目路径统一管理（基于脚本所在目录，避免依赖 CWD）============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GRAPH1_PATH = os.path.join(OUTPUT_DIR, "1graph.png")
# 本地 INT8 模型目录：随项目目录定位（不随 CWD 变化）
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "jev-int8")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 1. 状态与数据结构定义
# =========================================================
class ModerationState(TypedDict, total=False):
    # ==================== 输入字段 ====================
    input_text: str                      # 待审核的单条弹幕原文
    video_title: str | None              # 视频标题（如"遮天 第181集"），作为审核辅助语境
    video_summary: str | None            # 视频简介，补充剧情背景
    all_danmaku_summary: str | None      # 全集弹幕总结（可选，由外部传入）
    preprocessed_text: str               # 预处理后的弹幕文本（去空格等）

    # ==================== 全局上下文（离线预热阶段产出） ====================
    global_context: str | None           # 【给大模型用】详细版上下文，800-1000字，含剧情+黑话详解+违规提示
    short_global_context: str | None     # 【给小模型用】精简版上下文，150字以内，只保留核心梗

    # ==================== 上下文构建节点输出 ====================
    context_text: str                    # 拼接完成的大模型上下文（full_context + 当前弹幕）
    light_context_text: str              # 拼接完成的小模型上下文（short_context + 当前弹幕）

    # ==================== 轻量模型（edgejev）初筛结果 ====================
    light_label: str | None              # 初筛分类结果：safe / violation / gray
    light_confidence: float | None       # 初筛置信度（0~1）

    # ==================== 大模型（DeepSeek）复核结果 ====================
    llm_label: str | None                # 复核分类结果：safe / violation / gray
    llm_confidence: float | None         # 复核置信度（当前未使用）

    # ==================== 最终结果 ====================
    final_label: str | None              # 最终审核标签：safe / violation / gray
    route: str | None                    # 路由路径，用于统计与调试
                                         # 可能值：light_model_pass / light_model_block /
                                         #         llm_review / llm_final / human_review

    # ==================== 性能统计 ====================
    light_latency_ms: float              # 轻量模型初筛耗时（毫秒）
    llm_latency_ms: float                # 大模型复核耗时（毫秒）


class ModerationResult(BaseModel):
    label: Literal["safe", "violation", "gray"]


# =========================================================
# 2. 核心审核类封装
# =========================================================
class DanmakuFilter:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, light_model_max_chars: int = 512):
        self.model_path = model_path
        self.light_model_max_chars = light_model_max_chars
        self._setup_db()
        self._setup_models()
        self._build_graph()
        # 【诊断】打印当前用的 checkpointer 类型
        print(f"🔍 [诊断] Checkpointer 类型: {type(self.memory).__name__}")
    # -------- 初始化步骤 --------
    def _setup_db(self):
        self.memory = MemorySaver()
        '''        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # 不再下载，直接用本地文件
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        self.memory.setup()  # 让 SqliteSaver 自己创建表结构'''


    def _setup_models(self):
        # 1. 本地轻量模型
        self.light_model = Agent(self.model_path)

        # 2. 云端大模型
        self.llm = ChatDeepSeek(
            model="deepseek-flash",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},  # 关掉思考模式
        )
        self.structured_llm = self.llm.with_structured_output(
            ModerationResult,  method="json_mode", include_raw=True
        )


        self.sys_msg = SystemMessage(content="""
你是一个互联网内容安全审核模型。你需要审核用户发送的弹幕。
分类只有三个：safe、violation、gray
其中：
safe：正常、安全内容。
violation：存在明确的攻击、辱骂、色情、暴力等违规风险。
gray：即使结合提供的上下文，仍然无法可靠判断。

视频标题和视频简介只能作为辅助语境，不能因为视频本身属于搞笑、犯罪、影视等题材，就降低对用户弹幕本身的审核标准。
请只返回：safe 或者 violation 或者 gray，不要输出其他解释。
请严格按照 JSON 格式返回，且只包含一个字段：
{"label": "safe" | "violation" | "gray"}
""")

    # -------- 节点函数 --------
    def _preprocess_node(self, state: ModerationState):
        return {"preprocessed_text": state["input_text"].strip()}

    # 【新增】上下文构建节点
    # -------- 上下文构建节点（纯拼接，不做任何 LLM 调用）--------
    def _context_build_node(self, state: ModerationState):
        """
        职责：把外部传入的全局上下文和当前弹幕拼接成两份 Prompt。
        """
        text = state["preprocessed_text"]
        video_title = state.get("video_title") or "无"
        global_context = state.get("global_context") or "无"

        # ==================== 1. 详细版上下文（给 DeepSeek）====================
        full_context = (
            f"【视频标题】{video_title}\n"
            f"【全局背景】\n{global_context}\n"
            f"【当前弹幕】{text}"
        )

        # ==================== 2. 精简版上下文（给 edgejev）====================
        short_context = state.get("short_global_context") or ""
        if short_context:
            light_context = f"【视频】{video_title}\n【语境】{short_context}\n【弹幕】{text}"
        else:
            light_context = f"【视频】{video_title}\n【弹幕】{text}"

        return {
            "context_text": full_context,  # 给大模型复核用
            "light_context_text": light_context  # 给小模型初筛用
        }

    def _light_review_node(self, state: ModerationState):
        start_time = time.perf_counter()
        context_text = state.get("light_context_text") or state["preprocessed_text"]

        result = self.light_model.system_one(
            context_text,
            {
                "category": {
                    "type": "choice",
                    "instructions": "请对这段互联网弹幕进行内容安全分类",
                    "criteria": {
                        "safe": "正常、安全的内容，没有明显攻击、辱骂、色情、暴力等风险",
                        "violation": "存在明确违规、攻击、辱骂、色情、暴力等风险",
                        "gray": "语义存在歧义，仅凭当前文本无法可靠判断，需要进一步结合上下文"
                    }
                }
            }
        )

        category_result = result["answers"]["category"]
        latency = (time.perf_counter() - start_time) * 1000

        return {
            "light_label": category_result["choice"],
            "light_confidence": max(category_result["probabilities"].values()),
            "light_latency_ms": latency
        }

    def _pass_node(self, state: ModerationState):
        return {"final_label": "safe", "route": "light_model_pass"}

    def _block_node(self, state: ModerationState):
        return {"final_label": "violation", "route": "light_model_block"}

    def _llm_review_node(self, state: ModerationState):
        start_time = time.perf_counter()
        # 【修改】使用构建好的上下文
        context_text = state.get("context_text") or state["preprocessed_text"]

        prompt = f"""
【上下文信息】
{context_text}
请以 JSON 格式返回审核结果。
"""
        response = self.structured_llm.invoke([self.sys_msg, HumanMessage(content=prompt)])
        label = response["parsed"].label

        usage = getattr(response["raw"], "usage_metadata", None) or {}
        if not usage:
            usage = response["raw"].response_metadata.get("token_usage", {})

        latency = (time.perf_counter() - start_time) * 1000

        # 【修改】判定为 safe 或 violation 时，直接在这里设置 final_label
        final_label = label if label in {"safe", "violation"} else None

        return {
            "llm_label": label,
            "final_label": final_label,  # 新增
            "llm_confidence": None,
            "llm_latency_ms": latency,
            "route": "llm_review",
            "input_tokens": state.get("input_tokens", 0) + usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": state.get("output_tokens", 0) + usage.get("output_tokens",
                                                                       usage.get("completion_tokens", 0))
        }

    def _human_review_node(self, state: ModerationState):
        human_label = interrupt({
            "message": "模型无法确定，请人工审核",
            "text": state["input_text"],
            "light_label": state.get("light_label"),
            "llm_label": state.get("llm_label")
        })
        return {"final_label": human_label if human_label in {"safe", "violation"} else "gray", "route": "human_review"}

    # -------- 路由函数 --------
    def _route_after_light(self, state: ModerationState):
        label = state.get("light_label")
        confidence = state.get("light_confidence") or 0.0

        if label == "safe" and confidence >= 0.50:
            return "pass_node"
        elif label == "violation" and confidence >= 0.8:
            return "block_node"
        return "llm_review_node"

    # 【修改】删除了 llm_final_node 后的路由逻辑
    def _route_after_llm(self, state: ModerationState):
        # 大模型如果给出了确定答案，直接去 END；否则去人工
        if state.get("llm_label") in {"safe", "violation"}:
            return END
        return "human_review_node"

    # -------- 构图与执行 --------
    def _build_graph(self):
        builder = StateGraph(ModerationState)

        # 添加节点
        builder.add_node("preprocess_node", self._preprocess_node)
        builder.add_node("context_build_node", self._context_build_node)  # 【新增】
        builder.add_node("light_review_node", self._light_review_node)
        builder.add_node("pass_node", self._pass_node)
        builder.add_node("block_node", self._block_node)
        builder.add_node("llm_review_node", self._llm_review_node)
        # builder.add_node("llm_final_node", self._llm_final_node)  # 【删除】
        builder.add_node("human_review_node", self._human_review_node)

        # 连线
        builder.add_edge(START, "preprocess_node")
        builder.add_edge("preprocess_node", "context_build_node")  # 【修改】指向上下文构建
        builder.add_edge("context_build_node", "light_review_node")  # 【修改】上下文构建完再去轻量模型

        builder.add_conditional_edges(
            "light_review_node",
            self._route_after_light,
            {"pass_node": "pass_node", "block_node": "block_node", "llm_review_node": "llm_review_node"}
        )
        builder.add_conditional_edges(
            "llm_review_node",
            self._route_after_llm,
            {END: END, "human_review_node": "human_review_node"}  # 【修改】直接去 END
        )

        builder.add_edge("pass_node", END)
        builder.add_edge("block_node", END)
        builder.add_edge("human_review_node", END)

        self.graph = builder.compile(checkpointer=self.memory)

        try:
            png_data = self.graph.get_graph().draw_mermaid_png()
            with open(GRAPH1_PATH, "wb") as f:
                f.write(png_data)
        except Exception:
            pass

    def run(
            self,
            text: str,  # 待审核的单条弹幕原文
            video_title: str = None,  # 视频标题（如"遮天 第181集"），辅助审核语境
            video_summary: str = None,  # 视频简介，补充剧情背景
            global_context: str = None,  # 【给大模型用】详细版全局上下文（800-1000字）
            short_global_context: str = None  # 【给小模型用】精简版全局上下文（150字以内）
    ) -> dict:
        """
        对外暴露的唯一执行入口。

        参数说明：
        - text：单条弹幕文本
        - video_title / video_summary：辅助语境，帮助模型理解剧情
        - global_context：由全局预热 agent 生成的详细版上下文，喂给 DeepSeek
        - short_global_context：精简版上下文，喂给本地 edgejev 轻量模型（防止爆 token）

        返回：
        - 包含 final_label（safe/violation/gray）、route（路由路径）、
          light_latency_ms、llm_latency_ms 等字段的结果字典
        """
        # 为本次请求生成唯一 thread_id（时间戳 + 文本哈希），避免状态互相污染
        thread_id = f"mod-{int(time.time() * 1000)}-{hash(text)}"
        config = {"configurable": {"thread_id": thread_id}}

        # 组装初始状态，注入到 LangGraph
        test_input = {
            "input_text": text,
            "video_title": video_title,
            "video_summary": video_summary,
            "global_context": global_context,  # 详细版：供大模型复核使用
            "short_global_context": short_global_context,  # 精简版：供轻量模型初筛使用
            "light_latency_ms": 0.0,  # 轻量模型耗时初始化为 0
            "llm_latency_ms": 0.0  # 大模型耗时初始化为 0
        }

        # 调用 LangGraph 执行审核流程
        result = self.graph.invoke(test_input, config=config)

        # 兜底：如果走的是 human_review 分支，final_label 可能是 None
        if result.get("final_label") is None:
            result["final_label"] = "gray"

        return result