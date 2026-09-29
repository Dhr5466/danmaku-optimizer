import os
import json
import time
from dotenv import load_dotenv
from global_context_builder import GlobalContextBuilder

load_dotenv()


def load_config(config_path: str = "config.json") -> dict:
    """加载视频配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_danmaku_xml(xml_path: str) -> list[str]:
    """解析B站格式的弹幕XML，返回弹幕文本列表"""
    import xml.etree.ElementTree as ET
    if not os.path.exists(xml_path):
        print(f"❌ 找不到文件: {xml_path}")
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        danmaku_list = []
        for d in root.findall('d'):
            text = d.text
            if text:
                danmaku_list.append(text.strip())
        print(f"✅ 成功从 {xml_path} 解析出 {len(danmaku_list)} 条弹幕！")
        return danmaku_list
    except Exception as e:
        print(f"❌ 解析 XML 失败: {e}")
        return []


def build_global_context_for_video(video_title, video_summary, danmaku_list, output_txt, output_json):
    if not danmaku_list:
        print("⚠️ 弹幕列表为空！")
        return

    # 保序去重（避免每次运行结果不一致）
    unique_danmaku = list(dict.fromkeys(danmaku_list))
    all_danmaku_text = " | ".join(unique_danmaku)
    print(f"📊 原始 {len(danmaku_list)} 条，去重后 {len(unique_danmaku)} 条。")
    print(f"📝 拼接后文本长度：{len(all_danmaku_text)} 字符")

    builder = GlobalContextBuilder()
    result = builder.build_context(
        video_title=video_title,
        video_summary=video_summary,
        all_danmaku_text=all_danmaku_text
    )

    global_context = result["full"]
    short_context = result["short"]

    # 保存为 JSON（方便另一个 agent 读取）
    output_data = {
        "video_title": video_title,
        "video_summary": video_summary,
        "full_context": global_context,
        "short_context": short_context
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 JSON: {os.path.abspath(output_json)}")

    # 同时保存人类可读的 txt
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"【视频标题】\n{video_title}\n")
        f.write("=" * 60 + "\n")
        f.write(f"【详细版上下文（给 DeepSeek，{len(global_context)} 字）】\n")
        f.write(global_context + "\n")
        f.write("=" * 60 + "\n")
        f.write(f"【精简版上下文（给 edgejev，{len(short_context)} 字）】\n")
        f.write(short_context + "\n")
        f.write("=" * 60 + "\n")
    print(f"✅ 已保存 TXT: {os.path.abspath(output_txt)}")

    return output_data


if __name__ == "__main__":
    # ============ 1. 加载配置 ============
    config = load_config("config.json")
    print(f"📂 已加载配置: {config['video_title']}")

    # ============ 2. 解析 XML ============
    all_danmaku = parse_danmaku_xml(config["xml_file_path"])
    if not all_danmaku:
        exit(1)

    TEST_LIMIT = config.get("test_limit", 20000)
    test_danmaku = all_danmaku[:TEST_LIMIT]
    print(f"✂️ 截取前 {len(test_danmaku)} 条弹幕。")

    # ============ 3. 构建全局上下文 ============
    print("⏱️ 开始计时...")
    start_time = time.perf_counter()

    result = build_global_context_for_video(
        video_title=config["video_title"],
        video_summary=config["video_summary"],
        danmaku_list=test_danmaku,
        output_txt=config["output_txt"],
        output_json=config["output_json"]
    )

    end_time = time.perf_counter()
    print(f"\n{'='*50}")
    print(f"🕒 本次全局上下文构建总耗时: {end_time - start_time:.2f} 秒")
    print(f"{'='*50}")