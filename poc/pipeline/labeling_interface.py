"""
自动标注工具 - Streamlit界面
支持批量自动检测、标注结果浏览、手动画框标注
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import cv2
import numpy as np
import urllib.request
from PIL import Image

from pipeline.utils import load_yaml, resolve_path, ensure_parent_dir
from pipeline.label_auto import (
    discover_media,
    save_yolo_label,
    IMAGE_EXTS,
)
from streamlit_image_coordinates import streamlit_image_coordinates

DEFAULT_CLASSES = [
    'person', 'car', 'motorcycle', 'bus', 'truck', 'boat',
    'excavator', 'bulldozer', 'loader', 'crane', 'dump truck', 'concrete mixer',
    'trailer', 'van', 'tractor', 'forklift', 'ambulance', 'fire truck',
]

MODEL_URLS = {
    'yolov8x-worldv2.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt',
    'yolov8-worldv2.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8-worldv2.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt',
}

DEFAULT_MODEL_DIR = "poc/data/pretrainModel"


def get_model_path(model_name: str) -> Path:
    model_dir = resolve_path(DEFAULT_MODEL_DIR)
    return model_dir / model_name


def download_model(model_name: str, model_path: Path) -> bool:
    url = MODEL_URLS.get(model_name)
    if not url:
        st.error(f"未知模型: {model_name}")
        return False
    try:
        st.info(f"正在下载模型: {model_name}...")
        urllib.request.urlretrieve(url, str(model_path))
        st.success(f"模型已下载: {model_path}")
        return True
    except Exception as e:
        st.error(f"下载失败: {e}")
        return False


def ensure_model_available(model_name: str) -> Optional[Path]:
    model_path = get_model_path(model_name)
    if model_path.exists():
        return model_path
    if download_model(model_name, model_path):
        return model_path
    return None


def yolo_labels_from_result(result, class_names):
    labels = []
    if not hasattr(result, "boxes") or result.boxes is None:
        return labels
    h, w = result.orig_shape
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x_center = ((x1 + x2) / 2.0) / w
        y_center = ((y1 + y2) / 2.0) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        if isinstance(class_names, dict):
            label_name = class_names.get(cls_id, str(cls_id))
        elif isinstance(class_names, (list, tuple)) and cls_id < len(class_names):
            label_name = class_names[cls_id]
        else:
            label_name = str(cls_id)
        labels.append((cls_id, label_name, x_center, y_center, width, height, conf))
    return labels


def draw_annotations(
    image: np.ndarray,
    annotations: List[Dict],
) -> np.ndarray:
    img = image.copy()

    colors = {
        'person': (0, 255, 0),
        'truck': (255, 0, 0),
        'dump truck': (0, 128, 255),
        'excavator': (0, 0, 255),
        'car': (0, 255, 0),
        'bus': (0, 165, 255),
    }

    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann['class']
        confidence = ann.get('confidence', 1.0)
        is_manual = ann.get('manual', False)

        if is_manual:
            color = (0, 255, 0)
        else:
            color = colors.get(class_name, (255, 128, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        label = f"{class_name}: {confidence:.2f}" if confidence < 1.0 else class_name
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        if y1 - label_h - 10 < 0:
            bg_y1 = y1 + label_h + 10
            bg_y2 = y1
        else:
            bg_y1 = y1
            bg_y2 = y1 - label_h - 10

        cv2.rectangle(img, (x1, bg_y2), (x1 + label_w, bg_y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5 if bg_y2 < y1 else y1 + label_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


def get_image_files(raw_images_dir: str) -> List[str]:
    image_dir = resolve_path(raw_images_dir)
    if not image_dir.exists():
        return []
    image_files = discover_media(image_dir, IMAGE_EXTS)
    return [str(p) for p in sorted(image_files)]


def batch_detect(image_files: List[str], model, confidence: float) -> Dict[str, List[Dict]]:
    results = {}
    for i, img_path in enumerate(image_files):
        try:
            result = model(img_path, conf=confidence)[0]
            labels = yolo_labels_from_result(result, model.names)

            annotations = []
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            for cls_id, label_name, x_center, y_center, width, height, conf in labels:
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                annotations.append({
                    'class': label_name,
                    'confidence': round(conf, 4),
                    'bbox': [x1, y1, x2, y2],
                    'manual': False,
                })

            results[img_path] = annotations
        except Exception:
            results[img_path] = []

        if (i + 1) % 10 == 0:
            st.info(f"已处理 {i + 1}/{len(image_files)} 张...")

    return results


def render_labeling_interface():
    """渲染自动标注界面"""
    # 注意：不要在这里调用 st.set_page_config()，因为主应用已经设置过了

    # 初始化会话状态
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    if 'image_files' not in st.session_state:
        st.session_state.image_files = []
    if 'current_image_idx' not in st.session_state:
        st.session_state.current_image_idx = 0
    if 'detection_done' not in st.session_state:
        st.session_state.detection_done = False
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_path' not in st.session_state:
        st.session_state.model_path = ""
    if 'draw_start' not in st.session_state:
        st.session_state.draw_start = None
    if 'draw_end' not in st.session_state:
        st.session_state.draw_end = None
    if 'last_click' not in st.session_state:
        st.session_state.last_click = None

    annotations = st.session_state.annotations
    image_files = st.session_state.image_files

    st.header("🏷️ 自动标注工具")

    # 侧边栏配置（注意：在主应用中侧边栏已被占用，这里改为主区域配置）
    st.subheader("配置")

    col_config1, col_config2 = st.columns([1, 1])

    with col_config1:
        # 模型配置
        st.markdown("**模型设置**")
        config = load_yaml("poc/config/poc.yaml")
        model_name = st.selectbox(
            "选择模型",
            options=['yolov8x-worldv2.pt', 'yolov8-worldv2.pt', 'yolov8x.pt'],
            index=0
        )

        # 检查本地是否有模型
        model_path = get_model_path(model_name)
        local_exists = model_path.exists()

        # 自动加载本地模型
        if local_exists and not st.session_state.model_loaded:
            try:
                with st.spinner("正在加载本地模型..."):
                    from ultralytics import YOLO
                    st.session_state.model = YOLO(str(model_path))
                    st.session_state.model_loaded = True
                    st.session_state.model_path = str(model_path)
                st.success(f"模型已加载: {Path(model_path).name}")
            except Exception as e:
                st.error(f"加载失败: {e}")
        elif not local_exists and not st.session_state.model_loaded:
            st.warning("本地未找到模型，正在下载...")
            if ensure_model_available(model_name):
                st.rerun()

        # 显示当前模型状态
        if st.session_state.model_loaded:
            st.caption(f"当前模型: {Path(st.session_state.model_path).name}")

        # 置信度阈值
        confidence = st.slider("检测置信度", 0.1, 0.9, 0.25)

    with col_config2:
        # 路径配置
        st.markdown("**路径设置**")
        raw_images_dir = st.text_input(
            "图像目录",
            value=config.get("paths", {}).get("raw_images_dir", "warning_img")
        )

        labels_dir = st.text_input(
            "标签输出目录",
            value="poc/data/labels/auto"
        )
        ensure_parent_dir(resolve_path(labels_dir) / ".placeholder")

        if st.button("扫描图片", use_container_width=True):
            st.session_state.image_files = get_image_files(raw_images_dir)
            st.session_state.detection_done = False
            st.session_state.current_image_idx = 0
            st.session_state.draw_start = None
            st.session_state.draw_end = None
            st.rerun()

    st.divider()

    # 主内容区
    if not image_files:
        with st.spinner("扫描图片目录..."):
            st.session_state.image_files = get_image_files(raw_images_dir)
            image_files = st.session_state.image_files

    if not image_files:
        st.warning(f"未找到图片")
        st.info("请将图片放入指定目录。")
        return

    st.success(f"找到 {len(image_files)} 张图片")

    # 批量检测按钮
    if not st.session_state.detection_done:
        st.divider()
        col_start, col_progress = st.columns([1, 3])
        with col_start:
            if st.button("开始批量自动标注", type="primary", use_container_width=True, disabled=not st.session_state.model_loaded):
                if not st.session_state.model_loaded:
                    st.warning("请先加载模型")
                else:
                    with st.spinner("批量检测中，请稍候..."):
                        results = batch_detect(image_files, st.session_state.model, confidence)
                        st.session_state.annotations = results
                        st.session_state.detection_done = True
                    st.success(f"检测完成! 共 {len(image_files)} 张图片")
                    st.rerun()
        with col_progress:
            if st.session_state.model_loaded:
                st.info("点击按钮开始批量检测所有图片")
            else:
                st.warning("请先加载模型")

    # 检测完成后显示结果
    if st.session_state.detection_done:
        # 统计信息
        total_anns = sum(len(anns) for anns in annotations.values())
        detected_count = sum(1 for anns in annotations.values() if len(anns) > 0)
        class_counts = {}
        for anns in annotations.values():
            for ann in anns:
                cls = ann['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1

        st.divider()
        st.subheader("检测结果统计")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("总图片数", len(image_files))
        col_stat2.metric("有标注", detected_count)
        col_stat3.metric("无标注", len(image_files) - detected_count)
        col_stat4.metric("总标注数", total_anns)

        if class_counts:
            st.write("类别分布:")
            st.json(class_counts)

        # 图片浏览
        st.divider()
        st.subheader("标注结果浏览与编辑")

        current_idx = st.session_state.current_image_idx
        if current_idx >= len(image_files):
            current_idx = 0
            st.session_state.current_image_idx = 0

        col_nav1, col_nav2, col_nav3 = st.columns([2, 1, 1])
        with col_nav1:
            selected_image = st.selectbox(
                "选择图片",
                range(len(image_files)),
                index=current_idx,
                format_func=lambda i: Path(image_files[i]).name
            )
            st.session_state.current_image_idx = selected_image
            # 切换图片时清除点击状态
            if current_idx != selected_image:
                st.session_state.draw_start = None
                st.session_state.draw_end = None
                st.session_state.last_click = None
        with col_nav2:
            if st.button("上一张", disabled=selected_image == 0, use_container_width=True):
                st.session_state.current_image_idx = selected_image - 1
                st.session_state.draw_start = None
                st.session_state.draw_end = None
                st.session_state.last_click = None
                st.rerun()
        with col_nav3:
            if st.button("下一张", disabled=selected_image == len(image_files) - 1, use_container_width=True):
                st.session_state.current_image_idx = selected_image + 1
                st.session_state.draw_start = None
                st.session_state.draw_end = None
                st.session_state.last_click = None
                st.rerun()

        current_image_path = image_files[selected_image]
        current_image_name = Path(current_image_path).stem

        # 工具栏
        st.divider()
        col_tools1, col_tools2, col_tools3, col_tools4 = st.columns(4)

        with col_tools1:
            if st.button("重新检测当前", use_container_width=True):
                if st.session_state.model_loaded:
                    with st.spinner("检测中..."):
                        img = cv2.imread(current_image_path)
                        result = st.session_state.model(current_image_path, conf=confidence)[0]
                        labels = yolo_labels_from_result(result, st.session_state.model.names)

                        anns = []
                        h, w = img.shape[:2]
                        for cls_id, label_name, x_center, y_center, width, height, conf in labels:
                            x1 = int((x_center - width / 2) * w)
                            y1 = int((y_center - height / 2) * h)
                            x2 = int((x_center + width / 2) * w)
                            y2 = int((y_center + height / 2) * h)
                            anns.append({
                                'class': label_name,
                                'confidence': round(conf, 4),
                                'bbox': [x1, y1, x2, y2],
                                'manual': False,
                            })
                        annotations[current_image_path] = anns
                    st.session_state.draw_start = None
                    st.session_state.draw_end = None
                    st.rerun()

        with col_tools2:
            if st.button("清空当前标注", use_container_width=True):
                annotations[current_image_path] = []
                st.session_state.draw_start = None
                st.session_state.draw_end = None
                st.rerun()

        with col_tools3:
            if st.button("保存当前", use_container_width=True):
                output_dir = resolve_path(labels_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{current_image_name}.txt"

                img = cv2.imread(current_image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    yolo_labels = []
                    for ann in annotations.get(current_image_path, []):
                        x1, y1, x2, y2 = ann['bbox']
                        cls_id = DEFAULT_CLASSES.index(ann['class']) if ann['class'] in DEFAULT_CLASSES else 0
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        box_w = (x2 - x1) / w
                        box_h = (y2 - y1) / h
                        yolo_labels.append((cls_id, ann['class'], x_center, y_center, box_w, box_h, ann.get('confidence', 1.0)))
                    save_yolo_label(output_path, yolo_labels)
                    st.success("已保存")

        with col_tools4:
            if st.button("保存全部", use_container_width=True):
                output_dir = resolve_path(labels_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                saved_count = 0
                for img_path, anns in annotations.items():
                    img_stem = Path(img_path).stem
                    output_path = output_dir / f"{img_stem}.txt"
                    try:
                        temp_img = cv2.imread(str(img_path))
                        if temp_img is not None:
                            h, w = temp_img.shape[:2]
                            yolo_labels = []
                            for ann in anns:
                                x1, y1, x2, y2 = ann['bbox']
                                cls_id = DEFAULT_CLASSES.index(ann['class']) if ann['class'] in DEFAULT_CLASSES else 0
                                x_center = ((x1 + x2) / 2) / w
                                y_center = ((y1 + y2) / 2) / h
                                box_w = (x2 - x1) / w
                                box_h = (y2 - y1) / h
                                yolo_labels.append((cls_id, ann['class'], x_center, y_center, box_w, box_h, ann.get('confidence', 1.0)))
                            save_yolo_label(output_path, yolo_labels)
                            saved_count += 1
                    except Exception:
                        pass
                st.success(f"已保存 {saved_count} 张")

        # 手动标注区域
        st.divider()
        st.subheader("手动画框标注")

        # 加载图片
        img = cv2.imread(current_image_path)
        if img is None:
            st.error("无法读取图片")
            return

        h, w = img.shape[:2]

        # 选择类别
        col_class1, col_class2 = st.columns([2, 1])
        with col_class1:
            selected_class = st.selectbox("选择要标注的类别", DEFAULT_CLASSES, index=0, key="draw_class")
        with col_class2:
            st.markdown("""
            **操作步骤：**
            1. 先点击图片左上角
            2. 再点击图片右下角
            """)

        # 计算显示尺寸（用于坐标转换）
        max_display_width = 800
        display_width = min(w, max_display_width)
        display_height = int(h * display_width / w) if w > 0 else h

        # 计算缩放比例（显示坐标 -> 原始坐标）
        scale_x = w / display_width
        scale_y = h / display_height

        # 在原始图片上绘制所有框（自动识别 + 正在画的）
        annotated_img = draw_annotations(img, annotations.get(current_image_path, []))

        if st.session_state.draw_start and st.session_state.draw_end:
            x1 = min(st.session_state.draw_start[0], st.session_state.draw_end[0])
            y1 = min(st.session_state.draw_start[1], st.session_state.draw_end[1])
            x2 = max(st.session_state.draw_start[0], st.session_state.draw_end[0])
            y2 = max(st.session_state.draw_start[1], st.session_state.draw_end[1])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # 缩放到显示尺寸
        annotated_rgb_resized = cv2.resize(annotated_rgb, (display_width, display_height), interpolation=cv2.INTER_AREA)

        # 显示图片区域
        st.markdown("### 点击下方图片进行标注")
        st.info(f"图片尺寸: {w}×{h} | 显示尺寸: {display_width}×{display_height} | 缩放: {scale_x:.2f}×{scale_y:.2f}")

        # 获取点击坐标
        coords_value = streamlit_image_coordinates(
            annotated_rgb_resized,
            key="draw_coords",
            height=display_height,
            width=display_width,
        )

        # 处理点击坐标
        if coords_value is not None:
            click_x = coords_value['x']
            click_y = coords_value['y']

            # 如果是无效坐标，忽略
            if click_x == 0 and click_y == 0:
                pass
            else:
                last_click = st.session_state.get('last_click')

                if last_click != coords_value:
                    st.session_state.last_click = coords_value

                    # 转换到原始图片坐标
                    orig_x = int(click_x * scale_x)
                    orig_y = int(click_y * scale_y)

                    # 检查是否是第一次点击
                    if st.session_state.draw_start is None:
                        st.session_state.draw_start = (orig_x, orig_y)
                        st.session_state.draw_end = None
                        st.toast(f"起点已设置: ({orig_x}, {orig_y})，请点击右下角")
                        st.rerun()
                    elif st.session_state.draw_end is None:
                        # 防止重复点击同一点
                        if st.session_state.draw_start == (orig_x, orig_y):
                            st.toast("请点击不同的位置作为终点")
                        else:
                            st.session_state.draw_end = (orig_x, orig_y)
                            # 两点都获取后添加标注
                            x1 = min(st.session_state.draw_start[0], st.session_state.draw_end[0])
                            y1 = min(st.session_state.draw_start[1], st.session_state.draw_end[1])
                            x2 = max(st.session_state.draw_start[0], st.session_state.draw_end[0])
                            y2 = max(st.session_state.draw_start[1], st.session_state.draw_end[1])

                            if x2 > x1 and y2 > y1 and (x2 - x1) > 10 and (y2 - y1) > 10:
                                if current_image_path not in annotations:
                                    annotations[current_image_path] = []
                                annotations[current_image_path].append({
                                    'class': selected_class,
                                    'confidence': 1.0,
                                    'bbox': [x1, y1, x2, y2],
                                    'manual': True
                                })
                                st.toast(f"已添加标注: {selected_class}")
                                st.session_state.draw_start = None
                                st.session_state.draw_end = None
                                st.session_state.last_click = None
                                st.rerun()
                            else:
                                st.warning("框太小，请重新点击")
                                st.session_state.draw_start = None
                                st.session_state.draw_end = None

        # 显示当前画框状态
        if st.session_state.draw_start:
            sx, sy = st.session_state.draw_start
            st.warning(f"起点已设置: ({sx}, {sy})，请在图片上点击右下角")

        # 取消画框按钮
        if st.session_state.draw_start or st.session_state.draw_end:
            if st.button("取消画框"):
                st.session_state.draw_start = None
                st.session_state.draw_end = None
                st.rerun()

        # 标注列表
        st.divider()
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown("### 标注列表")
            current_annotations = annotations.get(current_image_path, [])

            if not current_annotations:
                st.info("暂无标注，请手动画框或等待自动检测")
            else:
                for idx, ann in enumerate(current_annotations):
                    is_manual = ann.get('manual', False)
                    prefix = "[手]" if is_manual else "[自动]"
                    conf_str = f" {ann.get('confidence', 1.0):.2f}" if ann.get('confidence', 1.0) < 1.0 else ""

                    with st.expander(f"{prefix} {ann['class']}{conf_str}", expanded=True):
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            new_cls = st.selectbox(
                                "类别",
                                DEFAULT_CLASSES,
                                index=DEFAULT_CLASSES.index(ann['class']) if ann['class'] in DEFAULT_CLASSES else 0,
                                key=f"class_{selected_image}_{idx}"
                            )
                            if new_cls != ann['class']:
                                annotations[current_image_path][idx]['class'] = new_cls
                                st.rerun()
                        with col_b:
                            if st.button("删除", key=f"del_{selected_image}_{idx}"):
                                annotations[current_image_path].pop(idx)
                                st.rerun()

                        x1, y1, x2, y2 = ann['bbox']
                        col_x1, col_y1 = st.columns(2)
                        with col_x1:
                            nx1 = st.number_input(f"x1##{idx}", value=x1, min_value=0, max_value=w, key=f"nx1_{idx}")
                        with col_y1:
                            ny1 = st.number_input(f"y1##{idx}", value=y1, min_value=0, max_value=h, key=f"ny1_{idx}")
                        col_x2, col_y2 = st.columns(2)
                        with col_x2:
                            nx2 = st.number_input(f"x2##{idx}", value=x2, min_value=0, max_value=w, key=f"nx2_{idx}")
                        with col_y2:
                            ny2 = st.number_input(f"y2##{idx}", value=y2, min_value=0, max_value=h, key=f"ny2_{idx}")

                        if [nx1, ny1, nx2, ny2] != [x1, y1, x2, y2]:
                            annotations[current_image_path][idx]['bbox'] = [nx1, ny1, nx2, ny2]
                            st.rerun()

        with col_right:
            st.markdown("### 使用说明")
            st.markdown("""
            **自动检测:**
            - 点击"开始批量自动标注"
            - 自动检测所有图片

            **手动标注:**
            1. 选择类别
            2. 点击图片左上角
            3. 点击图片右下角
            4. 框自动添加到列表

            **颜色说明:**
            - 绿色 = 手动标注
            - 橙色 = 自动检测

            **编辑标注:**
            - 可修改类别
            - 可调整坐标
            - 可删除标注
            """)


def main():
    render_labeling_interface()


if __name__ == "__main__":
    main()
