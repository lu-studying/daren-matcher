import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.feedback import feedback_by_brief, latest_run, log_event, save_run, stats
from src.pipeline.match import run_match

SAMPLE_BRIEFS = {
    "美妆·新品面霜": "我们是一款主打敏感肌的国产面霜，客单价150元，想在小红书找5个美妆达人做测评种草，总预算3万，粉丝量1-20万都可以，风格要真实测评感，不要硬广。",
    "3C·智能耳机": "新品降噪耳机发布，预算10万，想在抖音找3个科技数码达人做开箱视频，粉丝50万以上，最好有对比评测经验。",
    "食品·代餐零食": "新上线一款低卡代餐棒，想找B站和小红书的美食区达人带货，预算2万，粉丝量级不限，内容要有趣味性。",
}


@st.cache_data
def load_onepager() -> str:
    """读取 docs/one-pager.md 的正文。

    用 ROOT 拼绝对路径：Streamlit Cloud 的工作目录和本地不同，相对路径会读不到。
    文件缺失时给提示语而不是抛错，避免说明页的疏忽拖垮整个应用。
    """
    path = ROOT / "docs" / "one-pager.md"
    if not path.exists():
        return "一页说明未找到（docs/one-pager.md 未随应用一起部署）。"
    return path.read_text(encoding="utf-8")


@st.dialog("一页说明", width="large", dismissible=False)
def onepager_dialog():
    """首次进入时主动弹出的说明弹窗。

    关掉的方式只有底部的「我已了解」：写 doc_seen 后 rerun，弹窗就不会再出现。
    dismissible=False 是刻意的——允许点空白处关闭的话，关掉不会置位，下一次 rerun 又会弹回来。
    """
    st.markdown(load_onepager())
    st.caption("关闭后可随时在「一页说明」页签回看。")
    if st.button("我已了解", type="primary", use_container_width=True):
        st.session_state.doc_seen = True
        st.rerun()


st.set_page_config(page_title="达人智能匹配", page_icon="🎯", layout="wide")
st.title("🎯 达人智能匹配工具")
st.caption("「在线匹配」可直接试跑；「一页说明」是设计思路、评测与已知局限。")

tab_match, tab_doc = st.tabs(["在线匹配", "一页说明"])

if not st.session_state.get("doc_seen"):
    onepager_dialog()

# 先填说明页：匹配区在"需求未识别"时用 st.stop() 提前结束脚本，放在后面会被一起跳掉
with tab_doc:
    st.markdown(load_onepager())

with tab_match:
    typed = st.text_area("投放需求（brief）", height=140, placeholder="例如：我们是一款主打敏感肌的国产面霜，客单价150元，面向敏感肌群体，想在小红书找5个美妆达人做测评种草，总预算3万，粉丝量1-20万都可以，风格要真实测评感，不要硬广")
    if typed.strip() and st.session_state.get("sample") not in (None, "（自定义）"):
        # 手动输入优先：把遗留的示例选择重置掉，否则下拉框显示的内容会和实际使用的 brief 不一致
        st.session_state["sample"] = "（自定义）"
    sample = st.selectbox(
        "或直接选一个示例",
        ["（自定义）", *SAMPLE_BRIEFS],
        key="sample",
        disabled=bool(typed.strip()),
    )
    brief = typed.strip() or SAMPLE_BRIEFS.get(sample, "")

    if st.button("开始匹配", type="primary", disabled=not brief):
        log_event(brief, "search")
        with st.status("匹配中", expanded=True) as status:
            step = {"ph": None, "label": ""}

            def progress(msg):
                """推进到下一步：把上一步改写成 ✅ 已完成，再追加一行 🔄 进行中。

                作为 run_match(progress=...) 的回调，靠闭包里的 step 记住上一步，
                所以步与步之间不会互相覆盖。
                """
                if step["ph"] is not None:
                    step["ph"].markdown(f"✅ {step['label']}")
                step["ph"] = st.empty()
                step["label"] = msg
                step["ph"].markdown(f"🔄 {msg}")

            new_result = run_match(brief, progress=progress)
            failed = bool(new_result.get("error"))
            if step["ph"] is not None:
                step["ph"].markdown(f"{'⚠️' if failed else '✅'} {step['label']}")
            status.update(label="需求未识别" if failed else "匹配完成",
                          state="error" if failed else "complete", expanded=True)

        if not failed:
            save_run(brief, new_result)  # 没跑完的匹配不进历史记录
        st.session_state.result = new_result
        st.session_state.result_brief = brief
        st.session_state.feedback = {}

    if "result" not in st.session_state:
        saved = latest_run()
        if saved:
            st.session_state.result = saved["result"]
            st.session_state.result_brief = saved["brief"]
            st.session_state.feedback = feedback_by_brief(saved["brief"])

    result = st.session_state.get("result")
    if result:
        if result.get("error"):
            st.warning(result["error"])
            st.subheader("Brief 解析结果")
            st.json(result["brief_dict"])
            st.stop()  # 没有可用条件，不渲染推荐清单

        if st.session_state.result_brief != brief:
            st.caption("⚠️ 输入已修改，下方结果仍基于上一次提交的内容，点「开始匹配」重算。")
        # brief 要求的人数只作参考：系统固定给一批候选供人工对比审核，不按人数裁到只剩 1 个
        want = result["brief_dict"].get("count")
        hint = f"需求方期望 {want} 位，" if want else ""
        st.success(f"{hint}召回 {result['recall_count']} 个候选，展示 Top {len(result['results'])} 供审核"
                   + ("（硬性条件过滤后候选过少，已放宽）" if result["fallback"] else ""))

        st.subheader("Brief 解析结果")
        st.json(result["brief_dict"])

        st.subheader("推荐清单")
        for i, r in enumerate(result["results"], 1):
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{i}. {r['name']}** 　评分 **{r['score']}**")
                    st.write(r["profile_text"])
                    st.markdown(f"✅ 匹配理由：{r['reason']}")
                    if r.get("risk"):
                        st.markdown(f"⚠️ 风险：{r['risk']}")
                with col2:
                    did = str(r["daren_id"])
                    done = st.session_state.feedback.get(did)
                    if done:
                        st.markdown("✅ 已采纳" if done == "adopt" else "🚫 已标记不合适")
                    else:
                        if st.button("👍 采纳", key=f"ok{i}"):
                            log_event(st.session_state.result_brief, "adopt", did)
                            st.session_state.feedback[did] = "adopt"
                            st.rerun()
                        if st.button("👎 不合适", key=f"no{i}"):
                            log_event(st.session_state.result_brief, "reject", did)
                            st.session_state.feedback[did] = "reject"
                            st.rerun()

    st.divider()
    s = stats()
    st.caption(f"📊 累计使用 {s['searches']} 次 ｜ 采纳 {s['adoptions']} ｜ 标记不合适 {s['rejections']}")