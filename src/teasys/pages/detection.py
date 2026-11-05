"""检测功能页：单张图像、批量图像、视频流检测。"""
from __future__ import annotations

import base64
import io
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from .. import config, db
from ..components import (
    alert_banner,
    badge,
    card,
    class_badge,
    data_table,
    empty_illustration,
    level_badge,
    media_url,
    page_header,
)
from ..detector import detect_image, detect_video


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _save_upload(contents: str, filename: str) -> Path:
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    suffix = Path(filename).suffix.lower() or ".bin"
    safe = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    path = config.UPLOAD_DIR / safe
    path.write_bytes(raw)
    return path


def _graph(fig: go.Figure, height: int = 260):
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=20, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, sans-serif", size=12, color="#2c3d35"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e9e3")
    fig.update_yaxes(gridcolor="#edf2ee", zeroline=False)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _result_view(task: dict[str, Any], timings: dict[str, Any] | None = None):
    dets = task.get("detections", [])
    rows = []
    for i, d in enumerate(dets, 1):
        rows.append(
            [
                i,
                class_badge(d["class_name"]),
                html.Span(f"{d['confidence']:.3f}", className="num"),
                html.Span(
                    f"{d['x1']:.0f}, {d['y1']:.0f}, {d['x2']:.0f}, {d['y2']:.0f}",
                    className="mono",
                ),
                html.Span(f"{d.get('area_ratio', 0) * 100:.2f}%", className="num"),
            ]
        )
    img_url = media_url(task.get("annotated_path"))
    timing_txt = ""
    if timings:
        timing_txt = (
            f"预处理 {timings.get('preprocess_ms', 0):.2f} ms · "
            f"推理 {timings.get('inference_ms', 0):.2f} ms · "
            f"后处理 {timings.get('postprocess_ms', 0):.2f} ms"
        )
    return html.Div(
        [
            html.Div(
                className="row-between",
                children=[
                    html.Div(
                        [
                            level_badge(task.get("alert_level", "none")),
                            html.Span("  "),
                            badge(f"病灶 {task.get('total_objects', 0)} 处", "gray"),
                            html.Span("  "),
                            badge(f"严重程度：{task.get('severity', '无')}", "gold"),
                        ]
                    ),
                    html.Span(f"任务 {task.get('id')}", className="mono muted small"),
                ],
            ),
            html.Div(
                className="preview-box mt-12",
                children=html.Img(src=img_url) if img_url else html.Div("无标注图", className="empty"),
            ),
            html.Div(
                className="flex mt-12",
                children=[
                    dcc.Link("🤖 生成 AI 诊断报告 →", href=f"/assistant?task={task.get('id')}", className="btn gold small"),
                    html.Span(f"处理耗时 {task.get('elapsed_ms', 0):.0f} ms", className="muted small"),
                    html.Span(timing_txt, className="muted small"),
                ],
            ),
            html.Div(className="divider"),
            alert_banner(task.get("alert_level", "none"), task.get("alert_message") or "未检出病害，继续保持监测。"),
            html.Div(className="mt-12", children=[
                html.Div("检测明细", className="card-title", style={"marginBottom": "8px"}),
                data_table(
                    ["#", "类别", "置信度", "边界框 (x1,y1,x2,y2)", "面积占比"],
                    rows,
                    empty_text="该图像未检出目标",
                    max_height="260px",
                ),
            ]),
        ]
    )


def layout() -> html.Div:
    s = db.summary()
    header = page_header(
        "病害检测",
        "上传茶园图像或视频，调用 YOLOv12n 实时识别茶藻斑病、茶褐枯病与茶灰枯病",
        [
            html.Span(className="chip", children=[html.Span("模型"), html.Strong(config.MODEL_VERSION)]),
            html.Span(className="chip", children=[html.Span("mAP@0.5"), html.Strong("0.9738")]),
            html.Span(className="chip", children=[html.Span("参数量"), html.Strong("2.56 M")]),
            html.Span(className="chip", children=[html.Span("终端"), html.Strong(config.TERMINAL_ID)]),
        ],
    )

    kpis = html.Div(
        className="grid grid-4",
        children=[
            html.Div(className="card kpi", children=[
                html.Div("今日检测任务", className="label"),
                html.Div([html.Span(str(s["today_tasks"]), id="det-kpi-today"), html.Small("次")], className="value"),
                html.Div("单张 / 批量 / 视频", className="foot"),
            ]),
            html.Div(className="card kpi", children=[
                html.Div("累计检出病灶", className="label"),
                html.Div([html.Span(str(s["objects"]), id="det-kpi-objects"), html.Small("处")], className="value"),
                html.Div("D1 病害数据库", className="foot"),
            ]),
            html.Div(className="card kpi red", children=[
                html.Div("待处理预警", className="label"),
                html.Div([html.Span(str(s["unacked_alerts"]), id="det-kpi-alerts"), html.Small("条")], className="value"),
                html.Div("橙色 / 红色预警", className="foot"),
            ]),
            html.Div(className="card kpi gold", children=[
                html.Div("端到端速度", className="label"),
                html.Div([html.Span("70.76"), html.Small("FPS")], className="value"),
                html.Div("batch=1 · RTX 4060 Laptop GPU", className="foot"),
            ]),
        ],
    )

    def single_tab():
        return html.Div(
            className="grid grid-1-2",
            children=[
                card(
                    [
                        dcc.Upload(
                            id="single-upload",
                            multiple=False,
                            className="upload-zone",
                            children=[
                                html.Span("🌿", className="ico"),
                                html.Div("点击或拖拽上传叶片图像"),
                                html.Div("支持 JPG / PNG，建议 640×640 以上、光照均匀", className="hint"),
                            ],
                        ),
                        html.Div(id="single-preview", className="preview-box mt-12",
                                 children=empty_illustration("🖼️", "尚未上传图像", "上传后在此预览")),
                        html.Div(id="single-filename", className="muted small mt-8"),
                        html.Div(className="divider"),
                        html.Div(className="field", children=[
                            html.Label("置信度阈值"),
                            dcc.Slider(id="single-conf", min=0.1, max=0.9, step=0.05, value=config.CONF_THRESHOLD,
                                       marks={0.25: "0.25", 0.5: "0.5", 0.75: "0.75"},
                                       tooltip={"placement": "bottom"}),
                        ]),
                        html.Div(className="field mt-12", children=[
                            html.Label("采集终端"),
                            dcc.Dropdown(
                                id="single-terminal",
                                options=[{"label": config.TERMINAL_ID, "value": config.TERMINAL_ID}],
                                value=config.TERMINAL_ID,
                                clearable=False,
                                className="select",
                            ),
                        ]),
                        html.Button("开始检测", id="single-run", className="btn wide mt-16"),
                        html.Div(id="single-status", className="small muted mt-8"),
                    ],
                    title="图像输入",
                    subtitle="单张图像检测：自动标注病灶、判定严重程度并写入 D1",
                ),
                card(
                    dcc.Loading(
                        id="single-loading",
                        type="circle",
                        color="#1f6f43",
                        children=html.Div(
                            id="single-result",
                            children=empty_illustration("🔬", "检测结果将在这里显示", "上传图像并点击“开始检测”"),
                        ),
                    ),
                    title="检测结果",
                    subtitle="标注图、置信度、严重程度与预警通报",
                ),
            ],
        )

    def batch_tab():
        return html.Div(
            className="grid grid-1-2",
            children=[
                card(
                    [
                        dcc.Upload(
                            id="batch-upload",
                            multiple=True,
                            className="upload-zone",
                            children=[
                                html.Span("🗂️", className="ico"),
                                html.Div("点击或拖拽上传多张图像"),
                                html.Div("单次建议不超过 30 张，按文件逐一推理", className="hint"),
                            ],
                        ),
                        html.Div(id="batch-preview", className="mt-12",
                                 children=empty_illustration("📁", "尚未选择文件", "支持多选")),
                        html.Div(className="field mt-12", children=[
                            html.Label("置信度阈值"),
                            dcc.Slider(id="batch-conf", min=0.1, max=0.9, step=0.05, value=config.CONF_THRESHOLD,
                                       marks={0.25: "0.25", 0.5: "0.5", 0.75: "0.75"}),
                        ]),
                        html.Button("开始批量检测", id="batch-run", className="btn wide mt-16"),
                        html.Div(id="batch-status", className="small muted mt-8"),
                        html.Div(className="flex mt-12", children=[
                            html.Button("下载标注结果 ZIP", id="batch-download-btn", className="btn ghost small", disabled=True),
                            dcc.Download(id="batch-download"),
                        ]),
                    ],
                    title="批量输入",
                    subtitle="批量图像检测：汇总各类病害数量并生成可下载的标注结果",
                ),
                card(
                    dcc.Loading(
                        id="batch-loading", type="circle", color="#1f6f43",
                        children=html.Div(id="batch-result",
                                          children=empty_illustration("📊", "批量结果将在这里显示", "上传多张图像并开始检测")),
                    ),
                    title="批量结果",
                    subtitle="逐文件结果、类别分布与预警统计",
                ),
            ],
        )

    def video_tab():
        return html.Div(
            className="grid grid-1-2",
            children=[
                card(
                    [
                        dcc.Upload(
                            id="video-upload",
                            multiple=False,
                            accept="video/*",
                            className="upload-zone",
                            children=[
                                html.Span("🎞️", className="ico"),
                                html.Div("点击或拖拽上传视频"),
                                html.Div("支持 MP4 / AVI / MOV，默认处理前 60 秒", className="hint"),
                            ],
                        ),
                        html.Div(id="video-preview", className="preview-box mt-12",
                                 children=empty_illustration("🎬", "尚未上传视频", "上传后在此预览")),
                        html.Div(id="video-filename", className="muted small mt-8"),
                        html.Div(className="field mt-12", children=[
                            html.Label("置信度阈值"),
                            dcc.Slider(id="video-conf", min=0.1, max=0.9, step=0.05, value=config.CONF_THRESHOLD,
                                       marks={0.25: "0.25", 0.5: "0.5", 0.75: "0.75"}),
                        ]),
                        html.Button("开始视频检测", id="video-run", className="btn wide mt-16"),
                        html.Div(id="video-status", className="small muted mt-8"),
                        html.Div(className="flex mt-12", children=[
                            html.Button("下载标注视频", id="video-download-btn", className="btn ghost small", disabled=True),
                            dcc.Download(id="video-download"),
                        ]),
                    ],
                    title="视频输入",
                    subtitle="视频流检测：逐帧推理、峰值病灶统计与时间线分析",
                ),
                card(
                    dcc.Loading(
                        id="video-loading", type="circle", color="#1f6f43",
                        children=html.Div(id="video-result",
                                          children=empty_illustration("📈", "视频检测结果将在这里显示", "上传短视频并开始检测")),
                    ),
                    title="视频结果",
                    subtitle="标注视频、逐秒检出曲线与类别峰值",
                ),
            ],
        )

    tabs = html.Div(
        className="tabs",
        children=[
            html.Button("单张图像", id="tab-btn-single", className="tab-btn active", n_clicks=0),
            html.Button("批量处理", id="tab-btn-batch", className="tab-btn", n_clicks=0),
            html.Button("视频流检测", id="tab-btn-video", className="tab-btn", n_clicks=0),
        ],
    )

    return html.Div(
        [
            header,
            kpis,
            html.Div(className="mt-16", children=tabs),
            html.Div(id="tab-single", children=single_tab()),
            html.Div(id="tab-batch", style={"display": "none"}, children=batch_tab()),
            html.Div(id="tab-video", style={"display": "none"}, children=video_tab()),
            dcc.Store(id="single-store"),
            dcc.Store(id="batch-store"),
            dcc.Store(id="batch-zip-store"),
            dcc.Store(id="video-store"),
            dcc.Store(id="video-result-store"),
            dcc.Interval(id="det-kpi-refresh", interval=4000, n_intervals=0),
        ]
    )


def register_callbacks(app) -> None:
    # ---- tabs ---------------------------------------------------------
    @app.callback(
        Output("tab-single", "style"),
        Output("tab-batch", "style"),
        Output("tab-video", "style"),
        Output("tab-btn-single", "className"),
        Output("tab-btn-batch", "className"),
        Output("tab-btn-video", "className"),
        Input("tab-btn-single", "n_clicks"),
        Input("tab-btn-batch", "n_clicks"),
        Input("tab-btn-video", "n_clicks"),
        prevent_initial_call=True,
    )
    def switch_tab(n1, n2, n3):
        from dash import ctx

        triggered = ctx.triggered_id
        show = {"tab-btn-single": 0, "tab-btn-batch": 1, "tab-btn-video": 2}.get(triggered, 0)
        styles = [{"display": "block"} if show == i else {"display": "none"} for i in range(3)]
        cls = ["tab-btn active" if show == i else "tab-btn" for i in range(3)]
        return styles[0], styles[1], styles[2], cls[0], cls[1], cls[2]

    # ---- single -------------------------------------------------------
    @app.callback(
        Output("single-preview", "children"),
        Output("single-filename", "children"),
        Output("single-store", "data"),
        Input("single-upload", "contents"),
        State("single-upload", "filename"),
        prevent_initial_call=True,
    )
    def single_upload(contents, filename):
        if not contents:
            return no_update, no_update, no_update
        path = _save_upload(contents, filename or "upload.jpg")
        return (
            html.Img(src=media_url(path)),
            f"已上传：{filename}（{path.stat().st_size / 1024:.0f} KB）",
            {"path": str(path), "filename": filename},
        )

    @app.callback(
        Output("single-result", "children"),
        Output("single-status", "children"),
        Input("single-run", "n_clicks"),
        State("single-store", "data"),
        State("single-conf", "value"),
        State("single-terminal", "value"),
        prevent_initial_call=True,
    )
    def single_run(n, store, conf, terminal):
        if not store or not store.get("path"):
            return no_update, "请先上传图像"
        task_id = db.create_task("single", store["filename"], terminal)
        try:
            result = detect_image(store["path"], conf=conf)
        except Exception as e:  # invalid/corrupt image
            db.finalize_task(task_id, severity="无", alert_level="none")
            return html.Div(alert_banner("danger", f"检测失败：{e}")), "检测失败"
        db.add_detections(task_id, result["detections"])
        task = db.finalize_task(
            task_id,
            annotated_path=result["annotated_path"],
            elapsed_ms=result["timings"]["wall_ms"],
            severity=result["severity"],
            alert_level=result["alert_level"],
            alert_message=result["alert_message"],
        )
        return _result_view(task, result["timings"]), f"检测完成 · 任务 {task_id}"

    # ---- batch --------------------------------------------------------
    @app.callback(
        Output("batch-preview", "children"),
        Output("batch-store", "data"),
        Output("batch-status", "children"),
        Input("batch-upload", "contents"),
        State("batch-upload", "filename"),
        prevent_initial_call=True,
    )
    def batch_upload(contents_list, filenames):
        if not contents_list:
            return no_update, no_update, no_update
        items = []
        thumbs = []
        for contents, filename in zip(contents_list, filenames):
            path = _save_upload(contents, filename)
            items.append({"path": str(path), "filename": filename})
            thumbs.append(
                html.Div(
                    style={"width": "110px"},
                    children=[
                        html.Img(src=media_url(path), style={"width": "110px", "height": "78px", "objectFit": "cover", "borderRadius": "8px"}),
                        html.Div(filename[:14], className="small muted", style={"marginTop": "4px"}),
                    ],
                )
            )
        return html.Div(thumbs, className="flex"), items, f"已选择 {len(items)} 张图像"

    @app.callback(
        Output("batch-result", "children"),
        Output("batch-status", "children"),
        Output("batch-zip-store", "data"),
        Output("batch-download-btn", "disabled"),
        Input("batch-run", "n_clicks"),
        State("batch-store", "data"),
        State("batch-conf", "value"),
        prevent_initial_call=True,
    )
    def batch_run(n, items, conf):
        if not items:
            return no_update, "请先选择图像", no_update, True
        batch_id = uuid.uuid4().hex[:10]
        rows, annotated, per_class = [], [], {}
        total = 0
        t0 = time.perf_counter()
        for it in items:
            task_id = db.create_task("batch", it["filename"], batch_id=batch_id)
            try:
                res = detect_image(it["path"], conf=conf)
            except Exception as e:
                db.finalize_task(task_id, severity="无", alert_level="none")
                rows.append([it["filename"][:22], f"读取失败：{e}"[:40], "0", "—", badge("失败", "red")])
                continue
            db.add_detections(task_id, res["detections"])
            task = db.finalize_task(
                task_id,
                annotated_path=res["annotated_path"],
                elapsed_ms=res["timings"]["wall_ms"],
                severity=res["severity"],
                alert_level=res["alert_level"],
                alert_message=res["alert_message"],
            )
            total += task["total_objects"]
            for k, v in task["class_counts"].items():
                per_class[k] = per_class.get(k, 0) + v
            if res["annotated_path"]:
                annotated.append(Path(res["annotated_path"]))
            rows.append([
                it["filename"][:22],
                html.Span(", ".join(f"{config.CLASS_NAMES_CN.get(k, k)}×{v}" for k, v in task["class_counts"].items()) or "—"),
                str(task["total_objects"]),
                f"{task['max_confidence']:.3f}",
                level_badge(task["alert_level"]),
            ])

        zip_path = config.OUTPUT_DIR / f"batch_{batch_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in annotated:
                zf.write(p, arcname=p.name)

        fig = go.Figure(
            go.Bar(
                x=[config.CLASS_NAMES_CN.get(k, k) for k in per_class],
                y=list(per_class.values()),
                marker_color=[config.CLASS_COLORS.get(k, "#2e8b57") for k in per_class],
                text=list(per_class.values()),
                textposition="outside",
            )
        )
        elapsed = time.perf_counter() - t0
        content = html.Div([
            html.Div(className="grid grid-3", children=[
                html.Div(className="card kpi", children=[html.Div("处理文件", className="label"),
                                                         html.Div([html.Span(str(len(items))), html.Small("张")], className="value")]),
                html.Div(className="card kpi", children=[html.Div("检出病灶", className="label"),
                                                         html.Div([html.Span(str(total)), html.Small("处")], className="value")]),
                html.Div(className="card kpi gold", children=[html.Div("总耗时", className="label"),
                                                             html.Div([html.Span(f"{elapsed:.1f}"), html.Small("s")], className="value")]),
            ]),
            html.Div(className="divider"),
            _graph(fig, 220),
            data_table(["文件", "检出类别", "病灶数", "最高置信度", "预警"], rows, max_height="260px"),
            html.Div(f"批次 {batch_id} 已写入 D1 数据库；标注图已打包，可下载。", className="small muted mt-8"),
        ])
        return content, f"批量检测完成：{len(items)} 张，{total} 处病灶", str(zip_path), False

    @app.callback(
        Output("batch-download", "data"),
        Input("batch-download-btn", "n_clicks"),
        State("batch-zip-store", "data"),
        prevent_initial_call=True,
    )
    def batch_download(n, zip_path):
        if not zip_path or not Path(zip_path).exists():
            return no_update
        return dcc.send_file(zip_path)

    # ---- video --------------------------------------------------------
    @app.callback(
        Output("video-preview", "children"),
        Output("video-filename", "children"),
        Output("video-store", "data"),
        Input("video-upload", "contents"),
        State("video-upload", "filename"),
        prevent_initial_call=True,
    )
    def video_upload(contents, filename):
        if not contents:
            return no_update, no_update, no_update
        path = _save_upload(contents, filename or "upload.mp4")
        return (
            html.Video(src=media_url(path), controls=True, style={"width": "100%", "borderRadius": "10px"}),
            f"已上传：{filename}（{path.stat().st_size / 1024 / 1024:.1f} MB）",
            {"path": str(path), "filename": filename},
        )

    @app.callback(
        Output("video-result", "children"),
        Output("video-status", "children"),
        Output("video-result-store", "data"),
        Output("video-download-btn", "disabled"),
        Input("video-run", "n_clicks"),
        State("video-store", "data"),
        State("video-conf", "value"),
        prevent_initial_call=True,
    )
    def video_run(n, store, conf):
        if not store:
            return no_update, "请先上传视频", no_update, True
        task_id = db.create_task("video", store["filename"])
        try:
            res = detect_video(store["path"], conf=conf)
        except Exception as e:
            db.finalize_task(task_id, severity="无", alert_level="none")
            return html.Div(alert_banner("danger", f"视频检测失败：{e}")), "视频检测失败", no_update, True
        db.add_detections(task_id, res["detections"])
        task = db.finalize_task(
            task_id,
            annotated_path=res["annotated_path"],
            elapsed_ms=res["elapsed_s"] * 1000,
            severity=res["severity"],
            alert_level=res["alert_level"],
            alert_message=res["alert_message"],
        )
        timeline = res["timeline"]
        fig = go.Figure(
            go.Scatter(
                x=[t["second"] for t in timeline],
                y=[t["count"] for t in timeline],
                mode="lines",
                line=dict(color="#1f6f43", width=2),
                fill="tozeroy",
                fillcolor="rgba(46,139,87,.12)",
            )
        )
        fig.update_layout(xaxis_title="时间 (s)", yaxis_title="病灶数")
        counts = res["class_peak"]
        content = html.Div([
            html.Video(
                src=media_url(res["annotated_path"]),
                poster=media_url(res.get("poster_path")),
                controls=True,
                style={"width": "100%", "borderRadius": "10px", "maxHeight": "420px", "background": "#000"},
            ),
            html.Div(className="flex mt-12", children=[
                level_badge(res["alert_level"]),
                badge(f"峰值病灶 {res['total_lesions']} 处", "gray"),
                badge(f"严重程度：{res['severity']}", "gold"),
                html.Span(f"处理 {res['frames']} 帧 / {res['elapsed_s']} s", className="muted small"),
            ]),
            html.Div(className="mt-12", children=_graph(fig, 230)),
            html.Div(className="mt-12", children=[
                html.Div("类别峰值统计", className="card-title", style={"marginBottom": "8px"}),
                data_table(
                    ["类别", "峰值检出"],
                    [[class_badge(k), str(v)] for k, v in counts.items()],
                    empty_text="视频中未检出目标",
                ),
            ]),
        ])
        return content, f"视频检测完成 · 任务 {task_id}", {"path": res["annotated_path"], "timeline": timeline}, False

    @app.callback(
        Output("video-download", "data"),
        Input("video-download-btn", "n_clicks"),
        State("video-result-store", "data"),
        prevent_initial_call=True,
    )
    def video_download(n, store):
        if not store or not store.get("path") or not Path(store["path"]).exists():
            return no_update
        return dcc.send_file(store["path"])


    @app.callback(
        Output("det-kpi-today", "children"),
        Output("det-kpi-objects", "children"),
        Output("det-kpi-alerts", "children"),
        Input("det-kpi-refresh", "n_intervals"),
    )
    def refresh_kpis(n):
        s = db.summary()
        return str(s["today_tasks"]), str(s["objects"]), str(s["unacked_alerts"])
