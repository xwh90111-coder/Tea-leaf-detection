"""人工智能助手分析页：本地 LLM + D2 知识库 RAG + 检测结果解读。"""
from __future__ import annotations

import base64
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import dash
from dash import Input, Output, State, dcc, html, no_update

from .. import config, db
from ..assistant import analyze_file, answer_question, generate_report
from ..components import badge, card, class_badge, level_badge, media_url, page_header
from ..knowledge import kb_meta
from ..llm import llm_status

WELCOME = (
    "你好，我是智慧茶园植保助手。我会结合 **D1 检测记录** 与 **D2 茶叶知识库**，"
    "为你解读病害检测结果并给出防治建议。\n\n"
    "你可以：\n"
    "- 在左侧选择一条检测任务，然后提问；\n"
    "- 直接描述症状或防治问题；\n"
    "- 上传检测报告文件让我解读；\n"
    "- 让我根据最近一周的数据做趋势分析。"
)

PROMPTS = [
    ("这株茶树的病害严重吗？", "结合检测结果判断严重程度并说明依据"),
    ("推荐哪些防治药剂？", "按病害类别给出药剂、剂量和安全间隔期"),
    ("如何区分三种病害？", "从病斑颜色、形状、位置和表面特征区分"),
    ("下一步应该怎么监测？", "给出复查周期和区域巡查建议"),
]


def _task_options():
    opts = []
    for t in db.recent_tasks(45):
        counts = t.get("class_counts") or {}
        labels = "/".join(config.CLASS_NAMES_CN.get(k, k) for k in counts) or "未检出"
        opts.append({
            "label": f"{t['created_at'][5:16]} · {t['source_name'][:16] if t['source_name'] else t['id']} · {labels}",
            "value": t["id"],
        })
    return opts


def _task_summary(task: dict | None):
    if not task:
        return html.Div(
            "尚未载入检测任务。可从下拉框选择，或在检测功能页完成后点击“生成 AI 诊断报告”跳转。",
            className="empty",
            style={"padding": "18px 4px"},
        )
    counts = task.get("class_counts") or {}
    rows = [
        ("任务ID", html.Span(task["id"], className="mono")),
        ("文件", task.get("source_name") or "—"),
        ("检测类型", {"single": "单张图像", "batch": "批量图像", "video": "视频流"}.get(task["task_type"], task["task_type"])),
        ("病灶数", str(task["total_objects"])),
        ("类别", "、".join(f"{config.CLASS_NAMES_CN.get(k, k)}×{v}" for k, v in counts.items()) or "—"),
        ("最高置信度", f"{task['max_confidence']:.3f}"),
        ("严重程度", task.get("severity", "无")),
        ("预警", level_badge(task.get("alert_level", "none"))),
        ("时间", task.get("created_at", "")),
    ]
    img = media_url(task.get("annotated_path"))
    return html.Div([
        html.Div([html.Div([html.Span(k, className="k"), html.Span(v, className="v", title=str(v) if isinstance(v, str) else None)], className="status-line") for k, v in rows]),
        html.Img(src=img, className="task-thumb") if img else None,
        dcc.Link("查看标注大图 ↗", href=img or "/detection", target="_blank", className="small"),
    ])


def _bubble(role: str, content: str, meta: str = ""):
    return html.Div(
        className=f"bubble {role}",
        children=[
            html.Div(meta or ("你" if role == "user" else "智慧茶园助手"), className="meta"),
            dcc.Markdown(content, link_target="_blank"),
        ],
    )


def _chat_view(history: list[dict]):
    if not history:
        return _bubble("assistant", WELCOME, "智慧茶园助手")
    return [_bubble(m["role"], m["content"], m.get("meta", "")) for m in history]


def _sources_view(sources: list[dict]):
    if not sources:
        return html.Div("检索到的知识库片段将显示在这里", className="empty", style={"padding": "18px 4px"})
    items = []
    for s in sources:
        items.append(
            html.Div(
                className="source-item",
                children=[
                    html.Div(f"[{s['idx']}] {s['title']}", className="src-title"),
                    html.Div(f"{s['section']} · 相似度 {s.get('score', 0)}", className="src-meta"),
                    html.Div(s["text"], className="src-text"),
                    html.Div(s["source"], className="src-meta", style={"marginTop": "4px"}),
                ],
            )
        )
    return items


def _sessions_view():
    sessions = db.all_chat_sessions(8)
    if not sessions:
        return html.Div("暂无历史对话", className="empty", style={"padding": "14px 4px"})
    items = []
    for item in sessions:
        q = (item.get("first_q") or "（无提问）")[:34]
        items.append(
            html.Button(
                [
                    html.Div(q, style={"fontWeight": 600, "color": "#33513f", "textAlign": "left"}),
                    html.Div(f"{item['started'][5:16]} · {item['msgs']} 条消息", className="src-meta"),
                ],
                id={"type": "session-item", "index": item["session_id"]},
                className="session-item session-btn",
                n_clicks=0,
            )
        )
    return items


def _status_view():
    llm = llm_status()
    kb = kb_meta()
    dot_cls = "" if llm.get("available") else "off"
    return html.Div([
        html.Div(className="status-line", children=[html.Span("模型服务", className="k"),
                                                   html.Span([html.Span(className=f"dot {dot_cls}"), llm.get("provider", "—")], className="v")]),
        html.Div(className="status-line", children=[html.Span("模型", className="k"),
                                                   html.Span(llm.get("model", "—"), className="v", title=llm.get("model", ""))]),
        html.Div(className="status-line", children=[html.Span("模型大小", className="k"),
                                                   html.Span(f"{llm.get('size_gb', 0)} GB" if llm.get("size_gb") else "—", className="v")]),
        html.Div(className="status-line", children=[html.Span("状态", className="k"),
                                                   html.Span(llm.get("detail", "—"), className="v")]),
        html.Div(className="status-line", children=[html.Span("服务地址", className="k"),
                                                   html.Span(llm.get("base_url", "—"), className="v mono")]),
        html.Div(className="status-line", children=[html.Span("知识库片段", className="k"),
                                                   html.Span(f"{kb.get('chunks', 0)} 条 / {kb.get('docs', 0)} 篇", className="v")]),
        html.Div(className="status-line", children=[html.Span("向量模型", className="k"),
                                                   html.Span(kb.get("backend", "—").split("(")[0], className="v")]),
    ])


def layout(search: str | None = None) -> html.Div:
    kb = kb_meta()
    header = page_header(
        "人工智能助手",
        "本地部署 DeepSeek-R1-14B，结合 D2 茶叶知识库对检测结果进行诊断与决策分析",
        [
            html.Span(className="chip", children=[html.Span("LLM"), html.Strong("DeepSeek-R1-14B")]),
            html.Span(className="chip", children=[html.Span("知识库"), html.Strong(f"{kb.get('chunks', 0)} 片段")]),
            html.Span(className="chip", children=[html.Span("部署"), html.Strong("本地 · 数据不出园")]),
        ],
    )

    left = html.Div(className="grid", children=[
        card(
            [
                html.Div(className="field", children=[
                    html.Label("选择检测任务"),
                    dcc.Dropdown(id="assistant-task", options=_task_options(), placeholder="选择历史检测任务"),
                ]),
                html.Div(className="flex mt-8", children=[
                    html.Button("载入任务", id="assistant-load", className="btn small"),
                    html.Button("最近一周趋势", id="assistant-trend", className="btn ghost small"),
                ]),
                html.Div(id="assistant-task-summary", className="mt-12", children=_task_summary(None)),
            ],
            title="检测上下文",
            subtitle="绑定任务后，回答将引用具体检测结果",
            className="side-card",
        ),
        card(
            [
                dcc.Upload(
                    id="assistant-upload",
                    multiple=False,
                    className="upload-zone",
                    children=[html.Span("📄", className="ico"),
                              html.Div("上传检测报告"),
                              html.Div("TXT / CSV / JSON / MD", className="hint")],
                ),
                html.Div(id="assistant-upload-status", className="small muted mt-8"),
            ],
            title="文件理解",
            subtitle="读取报告文本，结合知识库进行解读",
            className="side-card",
        ),
        card(
            [
                html.Div(id="assistant-status", children=_status_view()),
                html.Button("检测模型连接", id="assistant-refresh", className="btn ghost small mt-8"),
            ],
            title="模型与知识库",
            subtitle="本地服务运行状态",
            className="side-card",
        ),
    ])

    center = card(
        [
            dcc.Loading(
                id="assistant-thinking",
                type="dot",
                color="#1f6f43",
                children=html.Div(id="assistant-chat", className="chat-window", children=_chat_view([])),
            ),
            html.Div("本地 DeepSeek-R1-14B 推理中可能需要 1～4 分钟，请耐心等待…", className="small muted"),
            html.Div(className="prompt-chips mt-12", children=[
                html.Button(t, id={"type": "prompt-chip", "index": i}, className="prompt-chip")
                for i, (t, _) in enumerate(PROMPTS)
            ]),
            html.Div(className="chat-input-row", children=[
                dcc.Textarea(id="assistant-input", placeholder="输入你的问题，例如：这批茶叶的褐枯病严重吗？应该怎么防治？"),
                html.Button("发送", id="assistant-send", className="btn"),
            ]),
            html.Div(className="flex mt-8", children=[
                html.Button("生成综合分析报告", id="assistant-report", className="btn gold small"),
                html.Button("清空对话", id="assistant-clear", className="btn ghost small"),
                dcc.Download(id="assistant-download"),
            ]),
            dcc.Store(id="assistant-history", data=[]),
            dcc.Store(id="assistant-session", data=None),
            dcc.Store(id="assistant-active-task", data=None),
            dcc.Store(id="assistant-report-store", data=None),
        ],
        title="分析对话",
        subtitle="支持多轮问答；报告可导出为 Markdown",
    )

    right = html.Div(className="grid", children=[
        card(
            [html.Div(id="assistant-sources", children=_sources_view([]))],
            title="参考来源",
            subtitle="D2 知识库检索片段（RRF 融合）",
            className="side-card",
        ),
        card(
            _sessions_view(),
            title="历史对话",
            subtitle="最近的分析会话记录",
            className="side-card",
        ),
    ])

    return html.Div([
        header,
        html.Div(className="assistant-grid", children=[left, center, right]),
    ])


def register_callbacks(app) -> None:
    # load a task chosen by the user or handed over by the routing callback
    @app.callback(
        Output("assistant-task-summary", "children"),
        Output("assistant-active-task", "data"),
        Input("assistant-load", "n_clicks"),
        Input("selected-task", "data"),
        State("assistant-task", "value"),
        prevent_initial_call=True,
    )
    def load_task(n, handed_task, picked):
        task_id = handed_task or picked
        if not task_id:
            return _task_summary(None), None
        task = db.get_task(task_id)
        return _task_summary(task), task_id

    @app.callback(
        Output("assistant-status", "children"),
        Input("assistant-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def refresh_status(n):
        return _status_view()

    # prompt chips fill the input
    @app.callback(
        Output("assistant-input", "value"),
        Input({"type": "prompt-chip", "index": 0}, "n_clicks"),
        Input({"type": "prompt-chip", "index": 1}, "n_clicks"),
        Input({"type": "prompt-chip", "index": 2}, "n_clicks"),
        Input({"type": "prompt-chip", "index": 3}, "n_clicks"),
        prevent_initial_call=True,
    )
    def fill_prompt(a, b, c, d):
        from dash import ctx

        if not ctx.triggered_id:
            return no_update
        return PROMPTS[ctx.triggered_id["index"]][0]

    @app.callback(
        Output("assistant-chat", "children"),
        Output("assistant-history", "data"),
        Output("assistant-sources", "children"),
        Output("assistant-input", "value"),
        Output("assistant-session", "data"),
        Input("assistant-send", "n_clicks"),
        State("assistant-input", "value"),
        State("assistant-task", "value"),
        State("assistant-active-task", "data"),
        State("assistant-history", "data"),
        State("assistant-session", "data"),
        prevent_initial_call=True,
    )
    def send(n, question, picked, active, history, session_id):
        if not question or not question.strip():
            return no_update, no_update, no_update, no_update, no_update
        task_id = active or picked
        history = list(history or [])
        session_id = session_id or uuid.uuid4().hex[:12]
        task = db.get_task(task_id) if task_id else None
        history.append({"role": "user", "content": question.strip(), "meta": "你"})
        db.save_message(session_id, "user", question.strip(), task_id=task_id or "")
        result = answer_question(question.strip(), task=task, history=history[:-1])
        meta = f"智慧茶园助手 · {result['model']} · {result['latency_s']}s"
        if result.get("fallback"):
            meta += " · 离线知识库模式"
        history.append({"role": "assistant", "content": result["answer"], "meta": meta})
        db.save_message(
            session_id, "assistant", result["answer"],
            sources=json.dumps(result.get("sources", []), ensure_ascii=False),
            task_id=task_id or "",
        )
        return _chat_view(history), history, _sources_view(result.get("sources", [])), "", session_id

    @app.callback(
        Output("assistant-chat", "children", allow_duplicate=True),
        Output("assistant-history", "data", allow_duplicate=True),
        Output("assistant-sources", "children", allow_duplicate=True),
        Output("assistant-report-store", "data", allow_duplicate=True),
        Output("assistant-download", "data"),
        Input("assistant-report", "n_clicks"),
        State("assistant-task", "value"),
        State("assistant-active-task", "data"),
        State("assistant-history", "data"),
        State("assistant-session", "data"),
        prevent_initial_call=True,
    )
    def make_report(n, picked, active, history, session_id):
        task_id = active or picked
        if not task_id:
            return no_update, no_update, no_update, no_update, no_update
        task = db.get_task(task_id)
        if not task:
            return no_update, no_update, no_update, no_update, no_update
        result = generate_report(task)
        history = list(history or [])
        history.append({"role": "user", "content": "请生成本次检测的病害诊断与防治报告", "meta": "你"})
        meta = f"智慧茶园助手 · {result['model']} · 报告生成"
        history.append({"role": "assistant", "content": result["report"], "meta": meta})
        session_id = session_id or uuid.uuid4().hex[:12]
        db.save_message(session_id, "user", "请生成本次检测的病害诊断与防治报告", task_id=task_id)
        db.save_message(
            session_id,
            "assistant",
            result["report"],
            sources=json.dumps(result.get("sources", []), ensure_ascii=False),
            task_id=task_id,
        )
        out = config.REPORT_DIR / f"report_{task_id}_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(result["report"], encoding="utf-8")
        return (
            _chat_view(history),
            history,
            _sources_view(result.get("sources", [])),
            str(out),
            dcc.send_file(str(out)),
        )

    @app.callback(
        Output("assistant-chat", "children", allow_duplicate=True),
        Output("assistant-history", "data", allow_duplicate=True),
        Output("assistant-sources", "children", allow_duplicate=True),
        Output("assistant-upload-status", "children"),
        Input("assistant-upload", "contents"),
        State("assistant-upload", "filename"),
        State("assistant-history", "data"),
        prevent_initial_call=True,
    )
    def upload_report(contents, filename, history):
        if not contents:
            return no_update, no_update, no_update, no_update
        try:
            raw = base64.b64decode(contents.split(",", 1)[1])
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("gbk", errors="ignore")
        except Exception:
            return no_update, no_update, no_update, "文件解析失败"
        result = analyze_file(filename or "report.txt", text)
        history = list(history or [])
        history.append({"role": "user", "content": f"请解读文件《{filename}》", "meta": "你"})
        history.append({"role": "assistant", "content": result["answer"],
                        "meta": f"智慧茶园助手 · {result['model']}"})
        return _chat_view(history), history, _sources_view(result.get("sources", [])), f"已解析《{filename}》"

    @app.callback(
        Output("assistant-chat", "children", allow_duplicate=True),
        Output("assistant-history", "data", allow_duplicate=True),
        Output("assistant-sources", "children", allow_duplicate=True),
        Input("assistant-clear", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear(n):
        return _chat_view([]), [], _sources_view([])

    @app.callback(
        Output("assistant-chat", "children", allow_duplicate=True),
        Output("assistant-history", "data", allow_duplicate=True),
        Output("assistant-sources", "children", allow_duplicate=True),
        Input("assistant-trend", "n_clicks"),
        State("assistant-history", "data"),
        prevent_initial_call=True,
    )
    def trend(n, history):
        days = db.daily_trend(7)
        classes = db.class_distribution()
        s = db.summary()
        parts = [f"近7天检测任务：{sum(d['tasks'] for d in days)} 次，病灶 {sum(d['objects'] for d in days)} 处。"]
        parts += [f"{d['date']}：{d['tasks']} 次任务，{d['objects']} 处病灶。" for d in days if d["tasks"]]
        parts.append("类别累计：" + "、".join(
            f"{config.CLASS_NAMES_CN.get(c['class_name'], c['class_name'])} {c['c']} 处" for c in classes) or "无")
        parts.append(f"累计任务 {s['tasks']} 次，平均置信度 {s['avg_confidence']:.3f}，红色预警 {s['danger']} 次。")
        question = (
            "请根据以下历史检测数据做趋势分析，指出病害发生规律、风险区域和下一步监测建议：\n"
            + "\n".join(parts)
        )
        result = answer_question(question, task=None)
        history = list(history or [])
        history.append({"role": "user", "content": "最近一周病害趋势分析", "meta": "你"})
        history.append({"role": "assistant", "content": result["answer"],
                        "meta": f"智慧茶园助手 · {result['model']}"})
        return _chat_view(history), history, _sources_view(result.get("sources", []))


    @app.callback(
        Output("assistant-chat", "children", allow_duplicate=True),
        Output("assistant-history", "data", allow_duplicate=True),
        Output("assistant-sources", "children", allow_duplicate=True),
        Output("assistant-session", "data", allow_duplicate=True),
        Input({"type": "session-item", "index": dash.ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def load_session(n_clicks):
        from dash import ctx

        if not ctx.triggered_id:
            return no_update, no_update, no_update, no_update
        session_id = ctx.triggered_id["index"]
        rows = db.chat_history(session_id, limit=30)
        if not rows:
            return no_update, no_update, no_update, no_update
        history = []
        sources = []
        for row in rows:
            history.append({
                "role": row["role"],
                "content": row["content"],
                "meta": "你" if row["role"] == "user" else "历史对话 · 智慧茶园助手",
            })
            if row["role"] == "assistant" and row.get("sources"):
                try:
                    sources = json.loads(row["sources"])
                except Exception:
                    pass
        return _chat_view(history), history, _sources_view(sources), session_id
