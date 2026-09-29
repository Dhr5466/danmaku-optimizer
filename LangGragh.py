import os
import sqlite3
import time
import urllib.request

from typing import Literal
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from edgejev import Agent


# =========================================================
# 1. 准备数据库
# =========================================================

os.makedirs("state_db", exist_ok=True)

url = "https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db"
db_path = "state_db/example.db"

if not os.path.exists(db_path):
    print("正在下载 example.db，请稍候...")

    try:
        urllib.request.urlretrieve(url, db_path)
        print("✅ 下载完成！文件已保存到:", db_path)
    except Exception as e:
        print("❌ 下载失败，错误信息:", e)

else:
    print("⏭️ 文件已存在，跳过下载。")


conn = sqlite3.connect(
    db_path,
    check_same_thread=False
)

print("🔗 成功连接数据库！")

memory = SqliteSaver(conn)


# =========================================================
# 2. 定义 State
# =========================================================

class ModerationState(TypedDict, total=False):

    # ===== 输入 =====
    input_text: str

    # ===== 视频上下文 =====
    video_title: str | None
    video_summary: str | None

    # ===== 输入预处理 =====
    preprocessed_text: str

    # ===== 轻量模型审核 =====
    light_label: str | None
    light_confidence: float | None

    # ===== LLM 复核 =====
    llm_label: str | None
    llm_confidence: float | None

    # ===== 最终结果 =====
    final_label: str | None
    route: str | None

    # ===== 性能统计 =====
    latency_ms: float
    input_tokens: int
    output_tokens: int


# =========================================================
# 3. 文本预处理
# =========================================================

def preprocesstext(orin_text: str):

    # 目前先只做最简单的清洗
    # 后面你可以在这里加入：
    # 去除多余空格
    # emoji处理
    # 特殊字符处理
    # 繁简转换等

    return orin_text.strip()


def preprocess_node(state: ModerationState):

    text = state["input_text"]

    cleaned_text = preprocesstext(text)

    return {
        "preprocessed_text": cleaned_text
    }


# =========================================================
# 4. 轻量级模型 Jev
# =========================================================

# 初始化 Agent，指向模型目录
lightweightmodel = Agent("../jev-int8")


def light_review_node(state: ModerationState):

    """
    轻量审核节点

    Jev 首先快速判断：
    safe / violation / gray
    """

    start_time = time.perf_counter()

    text_to_review = state["preprocessed_text"]

    result = lightweightmodel.system_one(
        text_to_review,
        {
            "category": {

                "type": "choice",

                "instructions": "请对这段互联网弹幕进行内容安全分类",

                "criteria": {

                    "safe":
                        "正常、安全的内容，没有明显攻击、辱骂、色情、暴力等风险",

                    "violation":
                        "存在明确违规、攻击、辱骂、色情、暴力等风险",

                    "gray":
                        "语义存在歧义，仅凭当前文本无法可靠判断，需要进一步结合上下文"
                }
            }
        }
    )

    category_result = result["answers"]["category"]

    label = category_result["choice"]

    confidence = max(
        category_result["probabilities"].values()
    )

    latency = (
        time.perf_counter() - start_time
    ) * 1000

    return {

        "light_label": label,

        "light_confidence": confidence,

        "latency_ms":
            state.get("latency_ms", 0.0) + latency
    }


# =========================================================
# 5. 轻量模型之后的路由
# =========================================================

def route_after_light_review(
    state: ModerationState

) -> Literal[
    "pass_node",
    "block_node",
    "llm_review_node"
]:

    label = state.get("light_label")

    confidence = (
        state.get("light_confidence")
        or 0.0
    )

    # 高置信度安全
    if (
        label == "safe"
        and confidence >= 0.90
    ):

        return "pass_node"

    # 高置信度违规
    elif (
        label == "violation"
        and confidence >= 0.90
    ):

        return "block_node"

    # gray
    # 或者置信度不足
    else:

        return "llm_review_node"


# =========================================================
# 6. 高置信度直接决策
# =========================================================

def pass_node(state: ModerationState):

    return {

        "final_label": "safe",

        "route": "light_model_pass"
    }


def block_node(state: ModerationState):

    return {

        "final_label": "violation",

        "route": "light_model_block"
    }


# =========================================================
# 7. LLM
# =========================================================

llm = ChatOpenAI(

    model="deepseek-chat",

    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),

    base_url="https://api.deepseek.com",

    temperature=0
)


sys_msg = SystemMessage(
    content="""
你是一个互联网内容安全审核模型。

你需要审核用户发送的弹幕。

分类只有三个：

safe
violation
gray

其中：

safe：
正常、安全内容。

violation：
存在明确的攻击、辱骂、色情、暴力等违规风险。

gray：
即使结合提供的上下文，仍然无法可靠判断。

视频标题和视频简介只能作为辅助语境，
不能因为视频本身属于搞笑、犯罪、影视等题材，
就降低对用户弹幕本身的审核标准。

请只返回：

safe

或者：

violation

或者：

gray

不要输出其他解释。
"""
)


def llm_review_node(state: ModerationState):

    start_time = time.perf_counter()

    text = state["preprocessed_text"]

    video_title = state.get(
        "video_title"
    )

    video_summary = state.get(
        "video_summary"
    )

    prompt = f"""
【视频标题】
{video_title or "无"}

【视频简介】
{video_summary or "无"}

【需要审核的弹幕】
{text}
"""

    response = llm.invoke(
        [
            sys_msg,
            HumanMessage(content=prompt)
        ]
    )

    label = response.content.strip().lower()

    # 简单兜底
    if label not in {
        "safe",
        "violation",
        "gray"
    }:

        label = "gray"

    # 获取 token 数量
    usage = getattr(
        response,
        "usage_metadata",
        None
    ) or {}

    input_tokens = usage.get(
        "input_tokens",
        0
    )

    output_tokens = usage.get(
        "output_tokens",
        0
    )

    latency = (
        time.perf_counter() - start_time
    ) * 1000

    return {

        "llm_label": label,

        # 暂时不要伪造 LLM confidence
        "llm_confidence": None,

        "input_tokens":
            state.get("input_tokens", 0)
            + input_tokens,

        "output_tokens":
            state.get("output_tokens", 0)
            + output_tokens,

        "latency_ms":
            state.get("latency_ms", 0.0)
            + latency,

        "route": "llm_review"
    }


# =========================================================
# 8. LLM 审核后的路由
# =========================================================

def route_after_llm_review(
    state: ModerationState

) -> Literal[
    "llm_final_node",
    "human_review_node"
]:

    label = state.get("llm_label")

    # safe / violation 已经可以出结果
    if label in {
        "safe",
        "violation"
    }:

        return "llm_final_node"

    # gray 留给人工
    return "human_review_node"


# =========================================================
# 9. LLM 最终结果
# =========================================================

def llm_final_node(
    state: ModerationState
):

    return {

        "final_label":
            state["llm_label"],

        "route":
            "llm_final"
    }


# =========================================================
# 10. 人工审核
# =========================================================

def human_review_node(
    state: ModerationState
):

    human_label = interrupt({

        "message":
            "模型无法确定，请人工审核",

        "text":
            state["input_text"],

        "light_label":
            state.get("light_label"),

        "light_confidence":
            state.get(
                "light_confidence"
            ),

        "llm_label":
            state.get("llm_label")
    })

    # 人工恢复时传入：
    # safe
    # 或 violation

    if human_label not in {
        "safe",
        "violation"
    }:

        human_label = "gray"

    return {

        "final_label":
            human_label,

        "route":
            "human_review"
    }


# =========================================================
# 11. Graph
# =========================================================

builder = StateGraph(
    ModerationState
)


# 添加节点
builder.add_node(
    "preprocess_node",
    preprocess_node
)

builder.add_node(
    "light_review_node",
    light_review_node
)

builder.add_node(
    "pass_node",
    pass_node
)

builder.add_node(
    "block_node",
    block_node
)

builder.add_node(
    "llm_review_node",
    llm_review_node
)

builder.add_node(
    "llm_final_node",
    llm_final_node
)

builder.add_node(
    "human_review_node",
    human_review_node
)


# =========================================================
# 12. 连线
# =========================================================

builder.add_edge(
    START,
    "preprocess_node"
)

builder.add_edge(
    "preprocess_node",
    "light_review_node"
)


# Jev之后判断去哪里
builder.add_conditional_edges(

    "light_review_node",

    route_after_light_review,

    {
        "pass_node":
            "pass_node",

        "block_node":
            "block_node",

        "llm_review_node":
            "llm_review_node"
    }
)


# LLM之后继续判断
builder.add_conditional_edges(

    "llm_review_node",

    route_after_llm_review,

    {
        "llm_final_node":
            "llm_final_node",

        "human_review_node":
            "human_review_node"
    }
)


# 最终节点结束
builder.add_edge(
    "pass_node",
    END
)

builder.add_edge(
    "block_node",
    END
)

builder.add_edge(
    "llm_final_node",
    END
)

builder.add_edge(
    "human_review_node",
    END
)


# =========================================================
# 13. 编译
# =========================================================

graph = builder.compile(
    checkpointer=memory
)


# =========================================================
# 14. 测试
# =========================================================

config = {

    "configurable": {

        "thread_id":
            "moderation-test-001"
    }
}


test_input = {

    "input_text":
        "你行你上啊",

    "video_title":
        "爆笑游戏挑战",

    "video_summary":
        "两名主播正在进行游戏比赛，并互相调侃。",

    "latency_ms":
        0.0,

    "input_tokens":
        0,

    "output_tokens":
        0
}


result = graph.invoke(
    test_input,
    config=config
)


print(result)