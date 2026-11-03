from __future__ import annotations

import tempfile
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ml.pipeline import EventEngine, annotate, assign_head_to_person, collect_boxes, model_path_or_pointer, region_summary

ROOT = Path(__file__).resolve().parent
VENDORED_ULTRALYTICS = ROOT / "ultralytics-8.3.163" / "ultralytics-8.3.163"
if VENDORED_ULTRALYTICS.is_dir():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
from ultralytics import YOLO

DEFAULT_HELMET = ROOT / "artifacts" / "weights" / "helmet_v0.1.0.pt"

st.set_page_config(page_title="筑安慧眼 | 施工安全监测", page_icon="🦺", layout="wide")
st.markdown("""
<style>
 .stApp {background:#f5f7fb;color:#17243a}
 .block-container {padding-top:1.5rem;max-width:1500px}
 [data-testid="stMetric"] {background:white;border:1px solid #e4eaf2;padding:15px 18px;border-radius:12px;box-shadow:0 3px 12px #18365b0a}
 .hero {padding:22px 26px;border-radius:16px;background:linear-gradient(115deg,#10243d,#174e62);color:#fff;margin-bottom:18px}
 .hero h1 {font-size:30px;margin:0 0 7px}.hero p{margin:0;color:#d4e4ee}
 .hint {padding:12px 16px;background:white;border-left:4px solid #1a8c73;border-radius:6px;color:#536477}
 </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>筑安慧眼 · 施工安全智能监测</h1><p>基于 YOLO11 与连续帧事件判定的本地视频分析演示</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("任务配置")
    weight_file = st.text_input("安全帽检测模型", str(DEFAULT_HELMET))
    person_weight = st.text_input("人员跟踪模型（COCO）", "yolo11s.pt")
    conf_head = st.slider("裸头阈值", .05, .8, .18, .01)
    conf_helmet = st.slider("安全帽阈值", .05, .8, .30, .01)
    frame_stride = st.slider("抽帧间隔", 1, 12, 2)
    region_names = st.text_input("监测区域（逗号分隔）", "东侧作业区,西侧作业区,材料区")
    vote_n = st.slider("连续观察帧数", 3, 30, 10)
    vote_ratio = st.slider("违规帧比例门槛", .3, 1.0, .6, .05)

tab_monitor, tab_events, tab_experiment = st.tabs(["视频监测", "事件与分区", "算法实验记录"])
with tab_monitor:
    left, right = st.columns([1.55, 1])
    with left:
        clip = st.file_uploader("上传授权的施工现场短视频", type=["mp4", "mov", "avi", "mkv"])
        region = st.selectbox("本次视频所属区域", [x.strip() for x in region_names.split(",") if x.strip()])
        run = st.button("开始分析", type="primary", disabled=clip is None, use_container_width=True)
        video_slot = st.empty()
    with right:
        st.subheader("检测说明")
        st.markdown('<div class="hint">系统将检测 head / helmet，并尝试用 COCO 人体模型关联到跟踪 ID。连续帧结果只生成“疑似违规”候选；人工复核后再确认。</div>', unsafe_allow_html=True)
        status = st.empty()

    if run and clip is not None:
        ok, message = model_path_or_pointer(weight_file)
        if not ok:
            st.error(message)
        else:
            with tempfile.TemporaryDirectory(prefix="zhuyan_demo_") as td:
                source = Path(td) / f"source{Path(clip.name).suffix or '.mp4'}"
                output = Path(td) / "annotated.mp4"
                source.write_bytes(clip.getvalue())
                cap = cv2.VideoCapture(str(source))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                head_model = YOLO(weight_file)
                people_model = YOLO(person_weight)
                engine = EventEngine(window=vote_n, violation_ratio=vote_ratio, min_observations=max(3, vote_n // 2))
                state_counts = {region: {"violation": 0, "compliant": 0} for region in
                                [x.strip() for x in region_names.split(",") if x.strip()]}
                progress = st.progress(0)
                frame_idx = 0
                snapshots = []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    if frame_idx % frame_stride == 0:
                        person_result = people_model.track(frame, persist=True, imgsz=640, conf=.35,
                                                           classes=[0], tracker="bytetrack.yaml", verbose=False)[0]
                        person_boxes = collect_boxes(person_result, {0: "person"}, {"person": .35})
                        helmet_result = head_model.predict(frame, imgsz=960, conf=.05, verbose=False)[0]
                        detected = collect_boxes(helmet_result, helmet_result.names,
                                                 {"head": conf_head, "helmet": conf_helmet})
                        timestamp = frame_idx / fps
                        frame_events = []
                        for head in [b for b in detected if b.name in ("head", "helmet")]:
                            person = assign_head_to_person(head.xyxy, person_boxes)
                            if person is None or person.track_id is None:
                                continue
                            state = "violation" if head.name == "head" else "compliant"
                            state_counts.setdefault(region, {"violation": 0, "compliant": 0})[state] += 1
                            item = engine.update(str(person.track_id), state, timestamp, region)
                            if item:
                                item["frame"] = frame_idx
                                engine.events[-1]["frame"] = frame_idx
                                frame_events.append(item)
                                snapshots.append({"事件时间（秒）": timestamp, "跟踪ID": person.track_id,
                                                  "区域": region, "状态": "待复核", "帧号": frame_idx})
                        shown = annotate(frame, person_boxes + detected, frame_events)
                        writer.write(shown)
                    else:
                        writer.write(frame)
                    frame_idx += 1
                    if total and frame_idx % max(1, total // 100) == 0:
                        progress.progress(min(1.0, frame_idx / total))
                cap.release(); writer.release(); progress.progress(1.0)
                if not output.is_file() or output.stat().st_size == 0:
                    st.error("未能生成检测视频。请检查输入视频格式和模型依赖。")
                else:
                    st.session_state["events"] = engine.events
                    st.session_state["event_rows"] = snapshots
                    st.session_state["regions"] = [x.strip() for x in region_names.split(",") if x.strip()]
                    old_scores = st.session_state.get("region_risk_scores", {})
                    risk = region_summary(engine.events, st.session_state["regions"], state_counts, old_scores)
                    st.session_state["risk_summary"] = risk
                    st.session_state["region_risk_scores"] = {x["区域"]: x["平滑风险"] for x in risk}
                    video_slot.video(output.read_bytes())
                    st.success(f"分析完成：处理 {frame_idx:,} 帧，形成 {len(engine.events)} 条连续帧候选事件。")
                    st.download_button("下载带检测框视频", output.read_bytes(), "zhuyan_annotated.mp4", "video/mp4")

with tab_events:
    events = st.session_state.get("event_rows", [])
    risk_data = st.session_state.get("risk_summary", [])
    names = st.session_state.get("regions", [x.strip() for x in region_names.split(",") if x.strip()])
    if risk_data:
        a, b, c = st.columns(3)
        a.metric("候选事件", len(events)); b.metric("监测区域", len(risk_data)); c.metric("人工复核", "待处理" if events else "暂无")
        st.subheader("分区风险概览")
        st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)
        if events:
            st.subheader("事件清单")
            table = pd.DataFrame(events)
            edited = st.data_editor(table, hide_index=True, use_container_width=True, disabled=["事件时间（秒）", "跟踪ID", "区域", "帧号"],
                                    column_config={"状态": st.column_config.SelectboxColumn(options=["待复核", "已确认", "误报", "已处理"])} )
            st.download_button("导出事件 CSV", edited.to_csv(index=False).encode("utf-8-sig"), "events.csv", "text/csv")
        else:
            st.info("当前片段没有达到连续帧违规门槛的事件。区域佩戴统计仍可供查看。")
    else:
        st.info("完成一次视频分析后，这里会显示去重事件、分区风险和人工复核状态。")

with tab_experiment:
    st.subheader("GitHub 当前基线（数据来自仓库实验记录）")
    st.caption("基线结果不是本次新训练结果。请以仓库 docs/实验记录.md 与对应产物为准。")
    col1, col2, col3 = st.columns(3)
    col1.metric("960 验证 mAP50", "0.8561")
    col2.metric("960 违规召回", "68.2%")
    col3.metric("960 单张推理", "18.2 ms")
    st.caption("640 基线为 mAP50 0.8207、违规召回 58.9%、推理 5.3 ms；上述均为 GitHub 记录中的实测值，不是本次运行数据。")
    st.markdown("**本版本的可复现改进策略**：960 高分辨率保留小头部细节；head / helmet 分别设置信心阈值；ByteTrack 绑定人员轨迹；滑动窗口投票压制单帧误报；事件按区域计数形成风险优先级。")
    st.warning("当前 GitHub checkout 不包含训练数据；安全帽模型文件目前是 133 字节的 Git LFS 指针。未下载到真实权重前，不能报告新训练提升，也无法运行模型推理。")

st.caption("筑安慧眼为安全管理辅助工具。画面遮挡、远距离小目标和复杂光照仍需人工复核；不要将单次模型输出作为处罚依据。")
