import json
import csv
import time
import xml.etree.ElementTree as ET
from danmaku_filter import DanmakuFilter


def load_global_context(path="global_context_result.json"):
    """读取全局预热 agent 生成的上下文"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_danmaku_xml(xml_path):
    """解析B站弹幕 XML"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    danmaku_list = []
    for d in root.findall('d'):
        text = d.text
        if text:
            danmaku_list.append(text.strip())
    return danmaku_list


def main():
    # ============ 配置 ============
    XML_FILE = "test.xml"
    CONTEXT_JSON = "global_context_result.json"
    OUTPUT_CSV = "danmaku_review_results.csv"
    TEST_LIMIT = 200        # 先跑 200 条，看效果再放开

    # ============ 1. 加载全局上下文 ============
    print("📂 加载全局上下文...")
    ctx = load_global_context(CONTEXT_JSON)
    video_title = ctx["video_title"]
    video_summary = ctx["video_summary"]
    full_context = ctx["full_context"]        # 给 DeepSeek
    short_context = ctx["short_context"]      # 给 edgejev
    print(f"   视频: {video_title}")
    print(f"   详细版: {len(full_context)} 字 | 精简版: {len(short_context)} 字")

    # ============ 2. 加载弹幕 ============
    print("📂 解析弹幕 XML...")
    all_danmaku = parse_danmaku_xml(XML_FILE)
    test_danmaku = list(dict.fromkeys(all_danmaku[:TEST_LIMIT]))  # 保序去重
    print(f"   原始 {len(all_danmaku)} 条 → 取前 {TEST_LIMIT} 条 → 去重后 {len(test_danmaku)} 条")

    # ============ 3. 初始化过滤器 ============
    print("🚀 初始化 DanmakuFilter...")
    filter_app = DanmakuFilter()

    # ============ 4. 逐条审核 ============
    results = []
    start_all = time.time()

    for i, text in enumerate(test_danmaku, 1):
        try:
            result = filter_app.run(
                text=text,
                video_title=video_title,
                video_summary=video_summary,
                global_context=full_context,
                short_global_context=short_context
            )
            pred_label = result.get("final_label") or "gray"
            route = result.get("route", "")
            latency = result.get("light_latency_ms", 0) + result.get("llm_latency_ms", 0)
        except Exception as e:
            import traceback
            print(f"\n❌ [{i}] 审核失败: {type(e).__name__}: {e}")
            traceback.print_exc()  # 打印完整堆栈
            pred_label = "error"
            route = f"exception"
            latency = 0
            # 打印 3 次后直接退出
            if i >= 3:
                print("\n⚠️ 前 3 条全部失败，退出。请看上面完整错误信息。")
                return

        route_display = {
            "light_model_pass": "Lite直过",
            "light_model_block": "Lite拦截",
            "llm_review": "LLM复核",
            "llm_final": "LLM终审",
            "human_review": "人工"
        }.get(route, route)

        results.append({
            "序号": i,
            "弹幕": text,
            "预测标签": pred_label,
            "路由": route_display,
            "耗时ms": round(latency, 1)
        })

        status = "✅" if pred_label == "safe" else ("🚫" if pred_label == "violation" else "❓")
        print(f"[{i:03d}] {status} {pred_label:<9} | {route_display:<8} | {latency:>5.0f}ms | {text[:40]}")

    total_time = time.time() - start_all

    # ============ 5. 保存 CSV（utf-8-sig 保证 Excel 不乱码）============
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["序号", "弹幕", "预测标签", "路由", "耗时ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ 结果已保存: {OUTPUT_CSV}")

    # ============ 6. 汇总 ============
    label_count, route_count = {}, {}
    for r in results:
        label_count[r["预测标签"]] = label_count.get(r["预测标签"], 0) + 1
        route_count[r["路由"]] = route_count.get(r["路由"], 0) + 1

    print("\n" + "=" * 60)
    print("📈 审核汇总")
    print("=" * 60)
    print(f"总条数: {len(results)}")
    print(f"总耗时: {total_time:.2f} 秒 | 平均: {total_time/len(results)*1000:.0f} ms/条")
    print(f"\n标签分布: {label_count}")
    print(f"路由分布: {route_count}")

    # 把可能出问题的挑出来给人看
    violations = [r for r in results if r["预测标签"] == "violation"]
    grays = [r for r in results if r["预测标签"] == "gray"]

    if violations:
        print(f"\n🚫 检出违规 {len(violations)} 条 (前 20 条):")
        for r in violations[:20]:
            print(f"   [{r['序号']:03d}] {r['弹幕']}")
    if grays:
        print(f"\n❓ 灰度/不确定 {len(grays)} 条 (前 20 条):")
        for r in grays[:20]:
            print(f"   [{r['序号']:03d}] {r['弹幕']}")


if __name__ == "__main__":
    main()