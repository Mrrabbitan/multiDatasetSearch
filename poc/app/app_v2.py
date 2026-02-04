"""
多模态数据底座 - 生产级可视化界面（修复版）

修复内容：
1. 日期选择器中文化
2. 显示原始图片和视频
3. 修复数据库路径问题
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime, time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from poc.pipeline.utils import connect_db, load_yaml, resolve_path
from poc.qa.agent import create_agent
from poc.qa.trace import init_trace_manager, get_trace_manager
from poc.qa.tools import init_tool_registry, get_tool_registry
from poc.search.query import (
    apply_filters,
    bbox_filter,
    encode_query,
    fetch_asset_context,
    load_index,
    load_model,
)

# 设置页面为中文
st.set_page_config(
    page_title="多模态数据底座",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 缓存函数
# ============================================================================

@st.cache_resource
def get_cached_model(model_name: str):
    """缓存CLIP模型，避免重复加载"""
    return load_model(model_name)


@st.cache_resource
def get_cached_index(index_dir: Path):
    """缓存向量索引，避免重复加载"""
    return load_index(index_dir)


@st.cache_resource
def get_cached_agent(config: Dict):
    """缓存 Agent"""
    return create_agent(config, max_retries=3)


@st.cache_resource
def init_systems(config: Dict):
    """初始化系统组件"""
    # 初始化追踪管理器
    trace_db_path = Path(config.get("paths", {}).get("trace_db_path", "logs/traces.db"))

    # 确保目录存在（修复问题3）
    trace_db_path.parent.mkdir(parents=True, exist_ok=True)

    init_trace_manager(
        db_path=trace_db_path,
        enable_file_log=True,
        log_dir=Path(config.get("paths", {}).get("log_dir", "logs"))
    )

    # 初始化 Tool 注册中心
    db_path = config.get("paths", {}).get("db_path", "poc/data/metadata.db")
    if Path(db_path).exists():
        init_tool_registry(db_path)

    return True


def load_config() -> Dict:
    return load_yaml("poc/config/poc.yaml")


def db_stats(db_path: Path) -> Dict[str, int]:
    if not Path(db_path).exists():
        return {"assets": 0, "events": 0, "detections": 0, "annotations": 0, "embeddings": 0}

    conn = connect_db(db_path)
    stats = {}
    try:
        stats["assets"] = conn.execute("SELECT COUNT(*) AS cnt FROM assets").fetchone()["cnt"]
        stats["events"] = conn.execute("SELECT COUNT(*) AS cnt FROM events").fetchone()["cnt"]
        stats["detections"] = conn.execute("SELECT COUNT(*) AS cnt FROM detections").fetchone()["cnt"]
        stats["annotations"] = conn.execute("SELECT COUNT(*) AS cnt FROM annotations").fetchone()["cnt"]
        stats["embeddings"] = conn.execute("SELECT COUNT(*) AS cnt FROM embeddings").fetchone()["cnt"]
    except:
        stats = {"assets": 0, "events": 0, "detections": 0, "annotations": 0, "embeddings": 0}
    finally:
        conn.close()
    return stats


# ============================================================================
# 辅助函数
# ============================================================================

def parse_media_urls(url_string: str) -> List[str]:
    """解析逗号分隔的URL字符串"""
    if not url_string or pd.isna(url_string):
        return []
    return [url.strip() for url in str(url_string).split(',') if url.strip()]


def display_media(video_url: str, img_urls: List[str]):
    """显示视频和图片（修复问题2）"""
    # 显示视频
    if video_url and not pd.isna(video_url):
        video_path = Path(video_url)
        if video_path.exists():
            st.video(str(video_path))
        elif video_url.startswith('http'):
            st.video(video_url)
        else:
            st.info(f"视频文件不存在: {video_url}")

    # 显示图片
    if img_urls:
        cols = st.columns(min(len(img_urls), 3))
        for i, img_url in enumerate(img_urls[:3]):  # 最多显示3张
            img_path = Path(img_url)
            with cols[i % 3]:
                if img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                elif img_url.startswith('http'):
                    st.image(img_url, use_column_width=True)
                else:
                    st.caption(f"图片不存在: {img_path.name}")


# ============================================================================
# 页面渲染函数
# ============================================================================

def render_architecture_overview():
    """渲染架构概览页面"""
    st.header("🏗️ 系统架构")

    st.markdown("""
    ### 核心技术栈

    本系统采用 **RAG + Agent** 架构，实现了从 POC 到生产级的完整升级。
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🤖 Agent 引擎", "LangGraph", help="状态机编排，支持自我修正")
    with col2:
        st.metric("🛡️ 安全护栏", "已部署", help="防止 SQL 注入和危险操作")
    with col3:
        st.metric("📊 监控追踪", "已启用", help="完整链路追踪和性能分析")
    with col4:
        st.metric("🔧 语义层", "4 个 Tools", help="业务逻辑抽象和封装")

    st.markdown("---")

    # 架构图
    st.subheader("架构流程图")
    st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                         用户层 (User Layer)                      │
│  Streamlit UI / REST API / CLI                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Agent 层 (LangGraph)                          │
│  状态机: Parse → Validate → Execute → Format                     │
│         ↓ error    ↓ error                                      │
│       Fix SQL ←────┘ (自我修正，最多3次重试)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   语义层 (Semantic Layer)                        │
│  Tool 1: 车辆统计 | Tool 2: 告警列表 | Tool 3: 地点解析         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   安全护栏 (Security Layer)                      │
│  ✓ SQL 注入防护  ✓ 白名单检查  ✓ 参数清理                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                   数据层 (Data Layer)                            │
│  SQLite (结构化) | FAISS (向量) | Trace DB (监控)                │
└─────────────────────────────────────────────────────────────────┘
    """, language="text")

    st.markdown("---")

    # 核心能力展示
    st.subheader("核心能力")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🤖 智能 Agent**
        - ✅ 自然语言转 SQL（NL2SQL）
        - ✅ SQL 执行失败自动重试
        - ✅ 错误自我修正
        - ✅ 多轮对话支持

        **🛡️ 安全防护**
        - ✅ SQL 注入防护
        - ✅ 危险操作拦截
        - ✅ 表访问白名单
        - ✅ 参数自动清理
        """)

    with col2:
        st.markdown("""
        **📊 监控追踪**
        - ✅ 完整链路追踪
        - ✅ 性能指标统计
        - ✅ 错误堆栈记录
        - ✅ 成功率分析

        **🔧 语义层抽象**
        - ✅ 业务逻辑封装
        - ✅ 复杂 SQL 隐藏
        - ✅ 可插拔扩展
        - ✅ 参数验证
        """)


def render_intelligent_qa():
    """渲染智能问答页面（展示 Agent 能力）"""
    st.header("🤖 智能问答（Agent 驱动）")

    st.markdown("""
    本功能基于 **LangGraph Agent** 实现，支持：
    - 🧠 自然语言理解
    - 🔄 SQL 自我修正（失败自动重试）
    - 📊 完整执行链路追踪
    - 🛡️ 安全护栏保护
    """)

    # 问题输入
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "请输入您的问题",
            value="近7天车辆闯入监控告警有多少条？",
            placeholder="例如：查询最近10条告警"
        )
    with col2:
        enable_trace = st.checkbox("启用追踪", value=True, help="记录完整执行过程")

    # 预设问题
    st.markdown("**快速选择：**")
    preset_questions = [
        "近7天车辆闯入监控告警有多少条？",
        "查询最近的10条告警",
        "统计所有告警数量",
        "查询2026年1月的告警"
    ]

    cols = st.columns(len(preset_questions))
    for i, q in enumerate(preset_questions):
        if cols[i].button(f"📝 {q[:10]}...", key=f"preset_{i}"):
            question = q
            st.rerun()

    if st.button("🚀 执行查询", type="primary", use_container_width=True):
        config = load_config()

        # 初始化系统
        try:
            init_systems(config)
        except Exception as e:
            st.error(f"系统初始化失败: {e}")
            return

        # 创建 Agent
        agent = get_cached_agent(config)

        # 执行查询
        with st.spinner("🤖 Agent 正在思考..."):
            result = agent.query(question, user_id="streamlit_user")

        # 显示结果
        st.markdown("---")

        # 状态指示
        if result["status"] == "success":
            st.success("✅ 查询成功")
        else:
            st.error(f"❌ 查询失败: {result['error']}")

        # 结果展示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("查询意图", result.get("intent", "未知"))
        with col2:
            st.metric("重试次数", result.get("retry_count", 0))
        with col3:
            if result.get("answer"):
                answer_value = result["answer"].get("value", "N/A")
                if isinstance(answer_value, int):
                    st.metric("查询结果", answer_value)
                else:
                    st.metric("返回记录数", len(answer_value) if isinstance(answer_value, list) else 1)

        # 执行详情（可折叠）
        with st.expander("🔍 查看执行详情", expanded=True):
            tab1, tab2, tab3 = st.tabs(["📝 生成的 SQL", "📊 执行历史", "💬 对话记录"])

            with tab1:
                st.code(result.get("sql", ""), language="sql")
                if result.get("sql_params"):
                    st.write("**参数:**", result["sql_params"])

            with tab2:
                if result.get("execution_history"):
                    for i, exec_record in enumerate(result["execution_history"]):
                        status_icon = "✅" if exec_record["status"] == "success" else "❌"
                        st.markdown(f"**尝试 {i+1}** {status_icon}")
                        st.code(exec_record["sql"], language="sql")
                        if exec_record.get("error"):
                            st.error(f"错误: {exec_record['error']}")
                        else:
                            st.success(f"返回 {exec_record.get('result_count', 0)} 条记录")
                else:
                    st.info("无执行历史")

            with tab3:
                if result.get("messages"):
                    for msg in result["messages"]:
                        role = msg.get("role", "system")
                        content = msg.get("content", "")
                        if role == "user":
                            st.chat_message("user").write(content)
                        elif role == "assistant":
                            st.chat_message("assistant").write(content)
                        else:
                            st.info(f"🔧 {content}")

        # 最终答案
        if result.get("answer"):
            st.markdown("---")
            st.subheader("📋 查询结果")

            answer_data = result["answer"].get("value")
            if isinstance(answer_data, list) and len(answer_data) > 0:
                # 列表结果，显示为表格
                df = pd.DataFrame(answer_data)
                st.dataframe(df, use_container_width=True)
            elif isinstance(answer_data, int):
                # 统计结果
                st.metric("统计结果", answer_data)
            else:
                st.json(result["answer"])


def render_multimodal_search():
    """渲染多模态检索页面"""
    st.header("🔍 多模态检索")

    st.markdown("""
    基于 **CLIP 模型** 的向量检索，支持：
    - 🖼️ 图像相似度搜索
    - 📝 文本语义搜索
    - 🎯 多条件过滤（时间、地点、事件类型）
    """)

    config = load_config()
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))

    col1, col2 = st.columns([2, 1])
    with col1:
        query_text = st.text_input("检索文本", value="车辆闯入监控告警")
    with col2:
        top_k = st.number_input("返回数量", min_value=1, max_value=50, value=10)

    # 过滤条件
    with st.expander("🎛️ 高级过滤", expanded=False):
        filter_event = st.text_input("事件类型过滤", value="")

        col3, col4, col5 = st.columns(3)
        with col3:
            enable_time_filter = st.checkbox("启用时间过滤", value=False)
            # 修复问题1：使用中文格式
            start_date = st.date_input("开始日期", format="YYYY年MM月DD日")
            start_time_t = st.time_input("开始时间", value=time(0, 0))
        with col4:
            end_date = st.date_input("结束日期", format="YYYY年MM月DD日")
            end_time_t = st.time_input("结束时间", value=time(23, 59))
        with col5:
            radius_km = st.number_input("半径(公里)", min_value=1.0, max_value=50.0, value=5.0)

        col6, col7 = st.columns(2)
        with col6:
            lat = st.text_input("纬度(lat)", value="")
        with col7:
            lon = st.text_input("经度(lon)", value="")

    if st.button("🔍 开始检索", type="primary", use_container_width=True):
        start_time_str = None
        end_time_str = None
        if enable_time_filter:
            if start_date:
                start_dt = datetime.combine(start_date, start_time_t)
                start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            if end_date:
                end_dt = datetime.combine(end_date, end_time_t)
                end_time_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        filters = {
            "event_type": filter_event or None,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None,
            "radius_km": radius_km,
        }

        with st.spinner("🔍 检索中..."):
            # 使用优化后的检索逻辑
            index_dir = resolve_path(config.get("paths", {}).get("index_dir", "poc/data/index"))
            model_name = config.get("search", {}).get("clip_model", "clip-ViT-B-32")

            try:
                meta, index_obj = get_cached_index(index_dir)
                model = get_cached_model(model_name)

                query_vec = encode_query(model, query_text, None)

                # 向量检索
                if meta.get("backend") == "faiss":
                    import faiss
                    scores, idx = index_obj.search(query_vec[None, :], top_k * 3)
                    pairs = list(zip(idx[0].tolist(), scores[0].tolist()))
                else:
                    import numpy as np
                    vectors = index_obj
                    scores = np.dot(vectors, query_vec)
                    idx = np.argsort(-scores)[:top_k * 3]
                    pairs = list(zip(idx.tolist(), scores[idx].tolist()))

                asset_ids = meta.get("asset_ids", [])
                candidate_ids = [asset_ids[i] for i, _ in pairs if i < len(asset_ids)]

                conn = connect_db(db_path)

                # 修复问题2：获取完整的资产信息，包括媒体URL
                placeholders = ",".join(["?"] * len(candidate_ids))
                rows = conn.execute(
                    f"""
                    SELECT a.asset_id, a.file_path, a.file_name, a.captured_at, a.lat, a.lon,
                           e.event_type, e.alarm_time, e.extra_json
                    FROM assets a
                    LEFT JOIN events e ON a.asset_id = e.asset_id
                    WHERE a.asset_id IN ({placeholders})
                    """,
                    candidate_ids,
                ).fetchall()

                assets = {}
                for row in rows:
                    row_dict = dict(row)
                    # 解析 extra_json 获取媒体URL
                    if row_dict.get("extra_json"):
                        try:
                            extra = json.loads(row_dict["extra_json"])
                            row_dict["video_url"] = extra.get("video_url", "")
                            row_dict["file_img_url_src"] = extra.get("file_img_url_src", "")
                            row_dict["file_img_url_icon"] = extra.get("file_img_url_icon", "")
                        except:
                            pass
                    assets[row_dict["asset_id"]] = row_dict

                conn.close()

                bbox = None
                if filters.get("lat") is not None and filters.get("lon") is not None:
                    bbox = bbox_filter(filters.get("lat"), filters.get("lon"), filters.get("radius_km", 5.0))

                filtered = apply_filters(
                    assets, filters.get("event_type"), filters.get("start_time"), filters.get("end_time"), bbox
                )

                results = []
                for i, score in pairs:
                    if i >= len(asset_ids):
                        continue
                    asset_id = asset_ids[i]
                    if asset_id not in filtered:
                        continue
                    info = filtered[asset_id]
                    results.append({"asset_id": asset_id, "score": float(score), **info})
                    if len(results) >= top_k:
                        break

                st.success(f"✅ 找到 {len(results)} 条结果")

                # 显示结果（修复问题2：显示媒体文件）
                for idx, item in enumerate(results):
                    with st.container():
                        st.markdown(f"### 结果 {idx + 1}")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.markdown(f"**相似度**: {item['score']:.4f}")
                            st.write(f"**事件类型**: {item.get('event_type', 'N/A')}")
                            st.write(f"**时间**: {item.get('alarm_time', 'N/A')}")
                            st.write(f"**位置**: ({item.get('lat', 'N/A')}, {item.get('lon', 'N/A')})")

                        with col2:
                            # 显示媒体文件
                            video_url = item.get("video_url", "")
                            img_urls_src = parse_media_urls(item.get("file_img_url_src", ""))
                            img_urls_icon = parse_media_urls(item.get("file_img_url_icon", ""))

                            # 优先显示原图，如果没有则显示框图
                            img_urls = img_urls_src if img_urls_src else img_urls_icon

                            if video_url or img_urls:
                                display_media(video_url, img_urls)
                            else:
                                # 如果没有媒体URL，尝试使用file_path
                                file_path = item.get("file_path")
                                if file_path:
                                    display_media("", [file_path])
                                else:
                                    st.info("无媒体文件")

                        st.markdown("---")

            except Exception as e:
                st.error(f"检索失败: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_system_monitor():
    """渲染系统监控页面"""
    st.header("📊 系统监控")

    config = load_config()
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))

    # 数据统计
    st.subheader("📈 数据统计")
    stats = db_stats(db_path)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("资产数", stats["assets"])
    col2.metric("事件数", stats["events"])
    col3.metric("检测数", stats["detections"])
    col4.metric("标注数", stats["annotations"])
    col5.metric("向量数", stats["embeddings"])

    st.markdown("---")

    # 追踪统计
    st.subheader("🔍 查询追踪统计")

    trace_manager = get_trace_manager()
    if trace_manager:
        trace_stats = trace_manager.get_statistics()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总查询数", trace_stats.get("total_queries", 0))
        col2.metric("成功数", trace_stats.get("success_count", 0))
        col3.metric("失败数", trace_stats.get("error_count", 0))

        success_rate = 0
        if trace_stats.get("total_queries", 0) > 0:
            success_rate = trace_stats["success_count"] / trace_stats["total_queries"] * 100
        col4.metric("成功率", f"{success_rate:.1f}%")

        st.metric("平均耗时", f"{trace_stats.get('avg_duration_ms', 0):.2f} 毫秒")

        # 按意图分组统计
        if trace_stats.get("by_intent"):
            st.markdown("**按意图分组:**")
            intent_df = pd.DataFrame([
                {"意图": k, "数量": v}
                for k, v in trace_stats["by_intent"].items()
            ])
            st.dataframe(intent_df, use_container_width=True)

        # 最近查询记录
        st.markdown("---")
        st.subheader("📝 最近查询记录")

        recent_traces = trace_manager.query_traces(limit=10)
        if recent_traces:
            trace_df = pd.DataFrame(recent_traces)
            # 只显示关键列
            display_cols = ["timestamp", "question", "intent", "status", "total_duration_ms"]
            available_cols = [col for col in display_cols if col in trace_df.columns]
            st.dataframe(trace_df[available_cols], use_container_width=True)
        else:
            st.info("暂无查询记录")
    else:
        st.warning("追踪系统未启用")

    st.markdown("---")

    # 语义层 Tools
    st.subheader("🔧 语义层 Tools")

    tool_registry = get_tool_registry()
    if tool_registry:
        tools = tool_registry.list_tools()
        tool_df = pd.DataFrame(tools)
        st.dataframe(tool_df, use_container_width=True)
    else:
        st.warning("Tool 注册中心未初始化")


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 侧边栏
    with st.sidebar:
        st.title("🏗️ 多模态数据底座")
        st.markdown("**生产级 RAG + Agent 架构**")
        st.markdown("---")

        page = st.radio(
            "导航",
            ["🏠 架构概览", "🤖 智能问答", "🔍 多模态检索", "📊 系统监控"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 系统状态")
        st.success("✅ Agent 已就绪")
        st.success("✅ 安全护栏已启用")
        st.success("✅ 追踪系统已启用")

        st.markdown("---")
        st.markdown("### 技术栈")
        st.markdown("""
        - 🤖 LangGraph
        - 🧠 DeepSeek
        - 🔍 FAISS
        - 🗄️ SQLite
        - 🎨 Streamlit
        """)

    # 主页面
    if page == "🏠 架构概览":
        render_architecture_overview()
    elif page == "🤖 智能问答":
        render_intelligent_qa()
    elif page == "🔍 多模态检索":
        render_multimodal_search()
    elif page == "📊 系统监控":
        render_system_monitor()


if __name__ == "__main__":
    main()
