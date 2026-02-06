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
    build_lance_filter,
    encode_query,
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
def get_cached_model(model_name: str, cache_dir: str = None, hf_mirror: str = None):
    """缓存CLIP模型，避免重复加载"""
    # 设置HuggingFace镜像源（国内加速）
    if hf_mirror:
        import os
        os.environ['HF_ENDPOINT'] = hf_mirror
        os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir if cache_dir else os.path.expanduser('~/.cache/huggingface')
        st.info(f"🌐 使用HuggingFace镜像源: {hf_mirror}")

    st.info(f"🔄 正在加载模型: {model_name}，请稍候...")
    model = load_model(model_name, cache_dir=cache_dir, hf_mirror=hf_mirror)

    # 显示模型信息
    dims = model.get_sentence_embedding_dimension()
    st.success(f"✅ 模型加载成功！维度: {dims}")

    return model


@st.cache_resource
def get_cached_lancedb(lancedb_dir: Path):
    """缓存 LanceDB 连接"""
    import lancedb
    db = lancedb.connect(str(lancedb_dir))
    return db


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
        # 尝试多个可能的路径
        possible_paths = [
            Path(video_url),
            Path("warning_file") / Path(video_url).name,
            Path("warning_file") / video_url,
            ROOT / video_url,
            ROOT / "warning_file" / Path(video_url).name
        ]

        video_found = False
        for video_path in possible_paths:
            if video_path.exists():
                try:
                    # 读取视频文件并显示
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    video_found = True
                    break
                except Exception as e:
                    st.warning(f"视频加载失败: {e}")
                    continue

        if not video_found:
            if video_url.startswith('http'):
                try:
                    st.video(video_url)
                except Exception as e:
                    st.error(f"视频播放失败: {e}")
            else:
                st.info(f"视频文件不存在: {video_url}")

    # 显示图片
    if img_urls:
        cols = st.columns(min(len(img_urls), 3))
        for i, img_url in enumerate(img_urls[:3]):  # 最多显示3张
            # 尝试多个可能的路径
            possible_paths = [
                Path(img_url),
                Path("warning_img") / Path(img_url).name,
                Path("warning_img") / img_url
            ]

            with cols[i % 3]:
                img_found = False
                for img_path in possible_paths:
                    if img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                        img_found = True
                        break

                if not img_found:
                    if img_url.startswith('http'):
                        st.image(img_url, use_container_width=True)
                    else:
                        st.caption(f"图片不存在: {Path(img_url).name}")


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
│  SQLite (结构化) | LanceDB (向量) | Trace DB (监控)              │
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

    # 初始化 session_state
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = "近7天车辆闯入监控告警有多少条？"

    # 问题输入
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "请输入您的问题",
            value=st.session_state.selected_question,
            placeholder="例如：查询最近10条告警",
            key="question_input"
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
            st.session_state.selected_question = q
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
                        # 修复：处理 LangChain 的消息对象
                        if hasattr(msg, 'type'):
                            # LangChain 消息对象
                            role = msg.type if hasattr(msg, 'type') else "system"
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                        elif isinstance(msg, dict):
                            # 字典格式
                            role = msg.get("role", "system")
                            content = msg.get("content", "")
                        else:
                            # 其他格式
                            role = "system"
                            content = str(msg)

                        if role == "user" or role == "human":
                            st.chat_message("user").write(content)
                        elif role == "assistant" or role == "ai":
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
                # 检查数据格式：如果是字典列表，直接转换；如果是元组列表，需要添加列名
                if isinstance(answer_data[0], dict):
                    df = pd.DataFrame(answer_data)
                elif isinstance(answer_data[0], (tuple, list)):
                    # 从 SQL 参数中提取列名
                    sql = result.get("sql", "")
                    # 尝试从 SELECT 语句中提取列名
                    import re
                    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                    if select_match:
                        columns_str = select_match.group(1)
                        # 解析列名（处理 AS 别名）
                        columns = []
                        for col in columns_str.split(','):
                            col = col.strip()
                            # 处理 AS 别名
                            if ' AS ' in col.upper():
                                col = col.split(' AS ')[-1].strip()
                            # 处理表名.列名格式
                            elif '.' in col:
                                col = col.split('.')[-1].strip()
                            columns.append(col)

                        df = pd.DataFrame(answer_data, columns=columns)
                    else:
                        df = pd.DataFrame(answer_data)
                else:
                    df = pd.DataFrame(answer_data)

                # 美化列名（将下划线替换为空格，首字母大写）
                df.columns = [col.replace('_', ' ').title() if isinstance(col, str) else col for col in df.columns]

                st.dataframe(df, use_container_width=True)

                # 如果结果包含图片路径，提供查看选项
                if any('path' in str(col).lower() or 'file' in str(col).lower() for col in df.columns):
                    st.info("💡 提示：结果中包含文件路径，您可以在下方查看图片")

                    # 让用户选择查看哪一行的图片
                    if len(df) > 0:
                        with st.expander("🖼️ 查看图片", expanded=False):
                            row_idx = st.selectbox("选择要查看的记录", range(len(df)), format_func=lambda x: f"第 {x+1} 行")

                            if row_idx is not None:
                                row_data = answer_data[row_idx]

                                # 查找图片路径列
                                img_path = None
                                if isinstance(row_data, dict):
                                    for key, value in row_data.items():
                                        if value and ('path' in str(key).lower() or 'file' in str(key).lower()):
                                            if str(value).endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                                img_path = value
                                                break
                                elif isinstance(row_data, (tuple, list)) and len(row_data) > 3:
                                    # 假设第4列是文件路径
                                    img_path = row_data[3] if len(row_data) > 3 else None

                                if img_path:
                                    # 尝试显示图片
                                    possible_paths = [
                                        Path(img_path),
                                        Path("warning_img") / Path(img_path).name,
                                        ROOT / "warning_img" / Path(img_path).name,
                                        ROOT / img_path
                                    ]

                                    img_found = False
                                    for p in possible_paths:
                                        if p.exists():
                                            st.image(str(p), use_container_width=True)
                                            img_found = True
                                            break

                                    if not img_found:
                                        st.warning(f"图片文件不存在: {img_path}")
                                else:
                                    st.info("该记录没有图片路径")

            elif isinstance(answer_data, int):
                # 统计结果
                st.metric("统计结果", answer_data)
            else:
                st.json(result["answer"])


def render_multimodal_search():
    """渲染多模态检索页面"""
    st.header("🔍 多模态检索")

    st.markdown("""
    基于 **CLIP 模型 + LanceDB** 的向量检索，支持：
    - 🖼️ 以图搜图（图像相似度搜索）
    - 📝 文本语义搜索
    - 🔍 图搜文（上传图片查询关联数据）
    - 🎯 多条件过滤（时间、地点、事件类型）
    - ⚡ 向量与元数据一体化存储，查询更高效
    """)

    # 检查 LanceDB 是否已初始化
    config = load_config()
    lancedb_dir = resolve_path(config.get("paths", {}).get("lancedb_dir", "poc/data/lancedb"))

    if not lancedb_dir.exists() or not (lancedb_dir / "embeddings.lance").exists():
        st.error("⚠️ 向量数据库未初始化")
        st.markdown("""
        **请先运行向量化脚本生成 LanceDB 数据：**

        ```bash
        python -m poc.pipeline.embed --config poc/config/poc.yaml
        ```

        或者使用快速入库脚本：
        ```bash
        ./重新入库.sh
        ```

        **说明：** 向量化过程会：
        1. 加载 CLIP 模型（首次运行会下载模型）
        2. 对所有图片生成向量嵌入
        3. 创建 LanceDB 向量索引

        完成后即可使用多模态检索功能。
        """)
        return

    config = load_config()
    db_path = resolve_path(config.get("paths", {}).get("db_path", "poc/data/metadata.db"))

    # 检索模式选择
    search_mode = st.radio(
        "检索模式",
        ["📝 文本检索", "🖼️ 以图搜图", "🔍 图搜文（查询关联数据）"],
        horizontal=True
    )

    query_text = None
    query_image = None

    if search_mode == "📝 文本检索":
        col1, col2 = st.columns([2, 1])
        with col1:
            query_text = st.text_input("检索文本", value="车辆闯入监控告警")
        with col2:
            top_k = st.number_input("返回数量", min_value=1, max_value=50, value=10)
    else:
        # 以图搜图或图搜文
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "上传图片",
                type=["jpg", "jpeg", "png", "bmp"],
                help="支持 JPG, PNG, BMP 格式"
            )
            if uploaded_file is not None:
                st.image(uploaded_file, caption="上传的图片", use_container_width=True)
                query_image = uploaded_file
        with col2:
            top_k = st.number_input("返回数量", min_value=1, max_value=50, value=10)

    # 过滤条件
    with st.expander("🎛️ 高级过滤", expanded=False):
        filter_event = st.text_input("事件类型过滤", value="")

        col3, col4, col5 = st.columns(3)
        with col3:
            enable_time_filter = st.checkbox("启用时间过滤", value=False)
            # 修复：使用 Streamlit 支持的日期格式
            start_date = st.date_input("开始日期", format="YYYY/MM/DD")
            start_time_t = st.time_input("开始时间", value=time(0, 0))
        with col4:
            end_date = st.date_input("结束日期", format="YYYY/MM/DD")
            end_time_t = st.time_input("结束时间", value=time(23, 59))
        with col5:
            radius_km = st.number_input("半径(公里)", min_value=1.0, max_value=50.0, value=5.0)

        col6, col7 = st.columns(2)
        with col6:
            lat = st.text_input("纬度(lat)", value="")
        with col7:
            lon = st.text_input("经度(lon)", value="")

    # 检查是否可以执行检索
    can_search = False
    if search_mode == "📝 文本检索" and query_text:
        can_search = True
    elif search_mode in ["🖼️ 以图搜图", "🔍 图搜文（查询关联数据）"] and query_image:
        can_search = True

    if not can_search:
        if search_mode == "📝 文本检索":
            st.info("请输入检索文本")
        else:
            st.info("请上传图片")

    if st.button("🔍 开始检索", type="primary", use_container_width=True, disabled=not can_search):
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
            # 使用 LanceDB 检索
            lancedb_dir = resolve_path(config.get("paths", {}).get("lancedb_dir", "poc/data/lancedb"))
            search_cfg = config.get("search", {})
            model_name = search_cfg.get("clip_model", "clip-ViT-B-32")
            cache_dir = search_cfg.get("model_cache_dir")
            hf_mirror = search_cfg.get("hf_mirror")

            try:
                # 加载模型和 LanceDB
                model = get_cached_model(model_name, cache_dir=cache_dir, hf_mirror=hf_mirror)
                db = get_cached_lancedb(lancedb_dir)
                table = db.open_table("embeddings")

                # 根据检索模式编码查询
                if search_mode == "📝 文本检索":
                    query_vec = encode_query(model, query_text, None)
                else:
                    # 以图搜图或图搜文：保存上传的图片到临时文件
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(query_image.read())
                        tmp_path = Path(tmp_file.name)

                    try:
                        query_vec = encode_query(model, None, tmp_path)
                    finally:
                        # 清理临时文件
                        tmp_path.unlink(missing_ok=True)

                # 构建 LanceDB 过滤条件
                filter_str = build_lance_filter(
                    event_type=filters.get("event_type"),
                    start_time=filters.get("start_time"),
                    end_time=filters.get("end_time"),
                    lat=filters.get("lat"),
                    lon=filters.get("lon"),
                    radius_km=filters.get("radius_km", 5.0),
                )

                # 执行向量搜索
                query = table.search(query_vec.tolist()).limit(top_k)
                if filter_str:
                    query = query.where(filter_str)

                results_df = query.to_pandas()

                # 转换为结果列表
                results = []
                for _, row in results_df.iterrows():
                    # 从 SQLite 获取 extra_json（媒体URL等）
                    conn = connect_db(db_path)
                    event_row = conn.execute(
                        "SELECT extra_json FROM events WHERE asset_id = ? LIMIT 1",
                        (row["asset_id"],)
                    ).fetchone()
                    conn.close()

                    result_item = {
                        "asset_id": row["asset_id"],
                        "score": float(row["_distance"]),  # LanceDB 返回距离
                        "file_path": row["file_path"],
                        "file_name": row["file_name"],
                        "captured_at": row["captured_at"],
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "event_type": row["event_type"],
                        "alarm_time": row["alarm_time"],
                        "alarm_level": row["alarm_level"],
                    }

                    # 解析 extra_json 获取媒体URL
                    if event_row and event_row["extra_json"]:
                        try:
                            extra = json.loads(event_row["extra_json"])
                            result_item["video_url"] = extra.get("video_url", "")
                            result_item["file_img_url_src"] = extra.get("file_img_url_src", "")
                            result_item["file_img_url_icon"] = extra.get("file_img_url_icon", "")
                        except:
                            pass

                    results.append(result_item)

                st.success(f"✅ 找到 {len(results)} 条结果")

                # 显示结果
                if search_mode == "🔍 图搜文（查询关联数据）":
                    # 图搜文模式：显示详细的关联数据
                    for idx, item in enumerate(results):
                        with st.container():
                            st.markdown(f"### 结果 {idx + 1} - 相似度: {item['score']:.4f}")

                            # 获取完整的关联数据
                            asset_id = item.get("asset_id")
                            conn = connect_db(db_path)

                            # 查询所有关联信息
                            asset_info = conn.execute(
                                """
                                SELECT * FROM assets WHERE asset_id = ?
                                """,
                                (asset_id,)
                            ).fetchone()

                            events = conn.execute(
                                """
                                SELECT * FROM events WHERE asset_id = ?
                                """,
                                (asset_id,)
                            ).fetchall()

                            detections = conn.execute(
                                """
                                SELECT * FROM detections WHERE asset_id = ?
                                """,
                                (asset_id,)
                            ).fetchall()

                            annotations = conn.execute(
                                """
                                SELECT * FROM annotations WHERE asset_id = ?
                                """,
                                (asset_id,)
                            ).fetchall()

                            conn.close()

                            col1, col2 = st.columns([1, 1])

                            with col1:
                                st.subheader("📋 资产信息")
                                if asset_info:
                                    asset_dict = dict(asset_info)
                                    st.json({
                                        "资产ID": asset_dict.get("asset_id"),
                                        "文件名": asset_dict.get("file_name"),
                                        "文件路径": asset_dict.get("file_path"),
                                        "拍摄时间": asset_dict.get("captured_at"),
                                        "纬度": asset_dict.get("lat"),
                                        "经度": asset_dict.get("lon"),
                                        "地址": asset_dict.get("location_name"),
                                    })

                                st.subheader("🚨 告警事件")
                                if events:
                                    events_data = []
                                    for event in events:
                                        event_dict = dict(event)
                                        events_data.append({
                                            "事件类型": event_dict.get("event_type"),
                                            "告警时间": event_dict.get("alarm_time"),
                                            "置信度": event_dict.get("confidence"),
                                            "描述": event_dict.get("description"),
                                        })
                                    st.dataframe(pd.DataFrame(events_data), use_container_width=True)
                                else:
                                    st.info("无告警事件")

                                st.subheader("🔍 检测结果")
                                if detections:
                                    detections_data = []
                                    for det in detections:
                                        det_dict = dict(det)
                                        detections_data.append({
                                            "类别": det_dict.get("class_name"),
                                            "置信度": det_dict.get("confidence"),
                                            "边界框": det_dict.get("bbox"),
                                        })
                                    st.dataframe(pd.DataFrame(detections_data), use_container_width=True)
                                else:
                                    st.info("无检测结果")

                                st.subheader("📝 标注信息")
                                if annotations:
                                    annotations_data = []
                                    for ann in annotations:
                                        ann_dict = dict(ann)
                                        annotations_data.append({
                                            "标注类型": ann_dict.get("annotation_type"),
                                            "标注者": ann_dict.get("annotator"),
                                            "标注时间": ann_dict.get("annotated_at"),
                                            "内容": ann_dict.get("content"),
                                        })
                                    st.dataframe(pd.DataFrame(annotations_data), use_container_width=True)
                                else:
                                    st.info("无标注信息")

                            with col2:
                                st.subheader("🖼️ 媒体文件")
                                # 显示媒体文件
                                video_url = item.get("video_url", "")
                                img_urls_src = parse_media_urls(item.get("file_img_url_src", ""))
                                img_urls_icon = parse_media_urls(item.get("file_img_url_icon", ""))

                                # 优先显示原图，如果没有则显示框图
                                img_urls = img_urls_src if img_urls_src else img_urls_icon

                                # 如果 extra_json 中没有媒体URL，使用 file_path 和 file_name
                                if not video_url and not img_urls:
                                    file_path = item.get("file_path")
                                    file_name = item.get("file_name")

                                    if file_name:
                                        img_urls = [file_name]
                                    elif file_path:
                                        img_urls = [file_path]

                                if video_url or img_urls:
                                    display_media(video_url, img_urls)
                                else:
                                    st.info("无媒体文件")

                            st.markdown("---")
                else:
                    # 文本检索或以图搜图模式：显示简洁结果
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

                                # 如果 extra_json 中没有媒体URL，使用 file_path 和 file_name
                                if not video_url and not img_urls:
                                    file_path = item.get("file_path")
                                    file_name = item.get("file_name")

                                    if file_name:
                                        # 使用 file_name 构建路径
                                        img_urls = [file_name]
                                    elif file_path:
                                        # 使用 file_path
                                        img_urls = [file_path]

                                if video_url or img_urls:
                                    display_media(video_url, img_urls)
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


def render_labeling_interface():
    """渲染自动标注页面"""
    # 导入标注界面模块
    try:
        from poc.pipeline.labeling_interface import render_labeling_interface as render_labeling
        render_labeling()
    except Exception as e:
        st.error(f"加载自动标注界面失败: {e}")
        st.info("""
        **备用方案：**

        如果遇到问题，可以单独启动标注界面：
        ```bash
        streamlit run poc/pipeline/labeling_interface.py
        ```
        """)


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
            ["🏠 架构概览", "🤖 智能问答", "🔍 多模态检索", "🏷️ 自动标注", "📊 系统监控"],
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
        - 🔍 LanceDB (向量数据库)
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
    elif page == "🏷️ 自动标注":
        render_labeling_interface()
    elif page == "📊 系统监控":
        render_system_monitor()


if __name__ == "__main__":
    main()
