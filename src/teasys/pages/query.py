"""数据查询功能页：多维检索、可视化统计、预警通报与 Excel 导出。"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from .. import config, db
from ..components import (
    badge,
    card,
    class_badge,
    data_table,
    level_badge,
    page_header,
)

TYPE_CN = {"single": "单张图像", "batch": "批量图像", "video": "视频流"}
PAGE_SIZE = 12


def _fig(fig: go.Figure, height: int = 250, legend: bool = False, gid: str | None = None):
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=26, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, sans-serif", size=12, color="#2c3d35"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e9e3")
    fig.update_yaxes(gridcolor="#edf2ee", zeroline=False)
    return dcc.Graph(id=gid, figure=fig, config={"displayModeBar": False})


def _default_dates():
    end = datetime.now().date()
    start = end - timedelta(days=13)
    return str(start), str(end)


def _filters():
    start, end = _default_dates()
    terminals = [t["terminal_id"] for t in db.terminal_distribution()] or [config.TERMINAL_ID]
    return start, end, terminals


def _kpis_html(s):
    return html.Div(
        className="grid grid-4",
        children=[
            html.Div(className="card kpi", children=[
                html.Div("检测任务", className="label"),
                html.Div([html.Span(str(s["tasks"])), html.Small("次")], className="value")]),
            html.Div(className="card kpi", children=[
                html.Div("检出病灶", className="label"),
                html.Div([html.Span(str(s["objects"])), html.Small("处")], className="value")]),
            html.Div(className="card kpi gold", children=[
                html.Div("平均置信度", className="label"),
                html.Div([html.Span(f"{s['avg_confidence']:.3f}")], className="value")]),
            html.Div(className="card kpi red", children=[
                html.Div("红色预警", className="label"),
                html.Div([html.Span(str(s["danger"])), html.Small("次")], className="value"),
                html.Div(f"待处理预警 {s['unacked_alerts']} 条", className="foot")]),
        ],
    )


def _trend_fig(days=14, classes=None, terminals=None, min_conf=None):
    data = db.daily_trend(days, classes=classes, terminals=terminals, min_conf=min_conf)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[d["date"] for d in data], y=[d["tasks"] for d in data],
        name="检测任务", marker_color="rgba(46,139,87,.35)",
    ))
    fig.add_trace(go.Scatter(
        x=[d["date"] for d in data], y=[d["objects"] for d in data],
        name="病灶数", mode="lines+markers", line=dict(color="#c8a24a", width=2.4),
    ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="date", tickformat="%m-%d")
    return _fig(fig, 260, legend=True, gid="query-trend-chart")


def _class_fig(classes=None, terminals=None, min_conf=None):
    data = db.class_distribution(classes=classes, terminals=terminals, min_conf=min_conf)
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", showarrow=False, font=dict(color="#93a79b"))
        return _fig(fig, 260, gid="query-class-chart")
    fig = go.Figure(go.Bar(
        x=[config.CLASS_NAMES_CN.get(d["class_name"], d["class_name"]) for d in data],
        y=[d["c"] for d in data],
        marker_color=[config.CLASS_COLORS.get(d["class_name"], "#2e8b57") for d in data],
        text=[d["c"] for d in data], textposition="outside",
    ))
    return _fig(fig, 260, gid="query-class-chart")


def _conf_fig(classes=None, terminals=None, min_conf=None):
    data = db.confidence_histogram(10, classes=classes, terminals=terminals, min_conf=min_conf)
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", showarrow=False, font=dict(color="#93a79b"))
        return _fig(fig, 260, gid="query-conf-chart")
    fig = go.Figure(go.Bar(
        x=[d["bin"] for d in data], y=[d["count"] for d in data],
        marker_color="#8ab89d",
    ))
    fig.update_xaxes(tickangle=-30, title="置信度区间")
    return _fig(fig, 260, gid="query-conf-chart")


def _terminal_fig(classes=None, min_conf=None):
    data = db.terminal_distribution(classes=classes, min_conf=min_conf)
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", showarrow=False, font=dict(color="#93a79b"))
        return _fig(fig, 260, gid="query-terminal-chart")
    fig = go.Figure(go.Pie(
        labels=[d["terminal_id"] or "未设置" for d in data],
        values=[d["c"] for d in data],
        hole=0.55,
        marker=dict(colors=["#1f6f43", "#c8a24a", "#6d4c41", "#8ab89d"]),
    ))
    return _fig(fig, 260, gid="query-terminal-chart")


def _table_html(tasks):
    rows = []
    for t in tasks:
        counts = t.get("class_counts") or {}
        cls = "、".join(f"{config.CLASS_NAMES_CN.get(k, k)}×{v}" for k, v in counts.items()) or "—"
        rows.append([
            html.Span(t["created_at"][5:16], className="mono"),
            dcc.Link(t["id"], href=f"/assistant?task={t['id']}", className="mono"),
            TYPE_CN.get(t["task_type"], t["task_type"]),
            html.Span(t["source_name"][:20] if t["source_name"] else "—", title=t["source_name"] or ""),
            cls,
            str(t["total_objects"]),
            f"{t['max_confidence']:.3f}",
            t["severity"],
            level_badge(t["alert_level"]),
            f"{t['elapsed_ms']:.0f}",
        ])
    return data_table(
        ["时间", "任务", "类型", "文件", "类别", "病灶数", "最高置信度", "严重度", "预警", "耗时(ms)"],
        rows,
        empty_text="没有符合条件的检测记录",
        max_height="540px",
    )


def layout() -> html.Div:
    start, end, terminals = _filters()
    tasks, total = db.query_tasks(date_from=start, date_to=end, limit=PAGE_SIZE, offset=0)
    s = db.summary(date_from=start, date_to=end)
    alerts = db.list_alerts(8)

    header = page_header(
        "数据查询",
        "检索 D1 病害数据库中的历史检测记录，分析病害发生规律并导出报表",
        [
            html.Span(className="chip", children=[html.Span("记录总数"), html.Strong(str(s["tasks"]))]),
            html.Span(className="chip", children=[html.Span("数据源"), html.Strong("D1 · SQLite")]),
        ],
    )

    filters = card(
        html.Div([
            html.Div(className="filters", children=[
                html.Div(className="field", children=[
                    html.Label("日期范围"),
                    dcc.DatePickerRange(
                        id="query-dates",
                        start_date=start,
                        end_date=end,
                        display_format="YYYY-MM-DD",
                        minimum_nights=0,
                        style={"width": "100%"},
                    ),
                ]),
                html.Div(className="field", children=[
                    html.Label("病害类别"),
                    dcc.Dropdown(
                        id="query-classes",
                        options=[{"label": config.CLASS_NAMES_CN[c], "value": c} for c in config.CLASS_NAMES],
                        multi=True,
                        placeholder="全部类别",
                    ),
                ]),
                html.Div(className="field", children=[
                    html.Label("采集终端"),
                    dcc.Dropdown(
                        id="query-terminal",
                        options=[{"label": t, "value": t} for t in terminals],
                        placeholder="全部终端",
                    ),
                ]),
                html.Div(className="field", children=[
                    html.Label("任务类型"),
                    dcc.Dropdown(
                        id="query-type",
                        options=[{"label": v, "value": k} for k, v in TYPE_CN.items()],
                        placeholder="全部类型",
                    ),
                ]),
                html.Div(className="field", children=[
                    html.Label("最低置信度"),
                    dcc.Slider(id="query-minconf", min=0, max=0.95, step=0.05, value=0,
                               marks={0: "0", 0.5: "0.5", 0.9: "0.9"}),
                ]),
                html.Div(className="field", children=[
                    html.Label("关键词"),
                    dcc.Input(id="query-keyword", type="text", placeholder="文件名 / 任务ID", className="input"),
                ]),
                html.Div(className="field", children=[
                    html.Label("操作"),
                    html.Div(className="flex", children=[
                        html.Button("查询", id="query-apply", className="btn"),
                        html.Button("导出 Excel", id="query-export", className="btn gold"),
                        dcc.Download(id="query-download"),
                    ]),
                ]),
            ]),
        ]),
        title="检索条件",
        subtitle="按时间、类别、终端、类型与置信度组合筛选",
    )

    kpis = html.Div(id="query-kpis", children=_kpis_html(s))

    charts = html.Div(className="grid grid-2 mt-16", children=[
        card(_trend_fig(14), title="检测趋势", subtitle="近 14 天任务量与病灶数"),
        card(_class_fig(), title="病害类别分布", subtitle="各类病灶累计检出数量"),
        card(_conf_fig(), title="置信度分布", subtitle="全部检测框的置信度直方图"),
        card(_terminal_fig(), title="终端分布", subtitle="按采集终端统计检测任务"),
    ])

    alerts_html = html.Div(
        [
            html.Div(
                className="flex",
                children=[
                    level_badge(a["level"]),
                    html.Span(a["created_at"][5:16], className="mono small muted"),
                    html.Span(a["source_name"] or a["task_id"], className="small"),
                    html.Span(a["message"], className="small muted"),
                ],
                style={"padding": "8px 0", "borderBottom": "1px solid #eef2ee"},
            )
            for a in alerts
        ] or [html.Div("暂无预警记录", className="empty")],
    )

    table_section = card(
        html.Div([
            html.Div(id="query-table", children=_table_html(tasks)),
            html.Div(
                className="row-between mt-12",
                children=[
                    html.Div(id="query-pageinfo", className="small muted",
                             children=f"共 {total} 条记录 · 第 1 / {max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)} 页"),
                    html.Div(className="flex", children=[
                        html.Button("上一页", id="query-prev", className="btn ghost small"),
                        html.Button("下一页", id="query-next", className="btn ghost small"),
                        dcc.Store(id="query-page-store", data={"page": 1, "total": total}),
                    ]),
                ],
            ),
        ]),
        title="检测记录",
        subtitle="点击任务ID可跳转至 AI 助手进行深度分析",
    )

    return html.Div([
        header,
        filters,
        html.Div(className="mt-16", children=kpis),
        charts,
        html.Div(className="mt-16", children=table_section),
        html.Div(className="mt-16", children=card(
            alerts_html,
            title="预警通报",
            subtitle="系统根据检测严重程度自动生成的预警记录",
        )),
    ])


def _read_filters(store):
    date_from = store.get("start")
    date_to = store.get("end")
    return dict(
        date_from=date_from,
        date_to=date_to,
        classes=store.get("classes") or None,
        terminals=store.get("terminals") or None,
        task_types=store.get("types") or None,
        min_conf=(store.get("min_conf") or 0) or None,
        keyword=(store.get("keyword") or "").strip() or None,
    ) if store else {}


def register_callbacks(app) -> None:
    @app.callback(
        Output("query-kpis", "children"),
        Output("query-table", "children"),
        Output("query-pageinfo", "children"),
        Output("query-page-store", "data"),
        Output("query-trend-chart", "figure"),
        Output("query-class-chart", "figure"),
        Output("query-conf-chart", "figure"),
        Output("query-terminal-chart", "figure"),
        Input("query-apply", "n_clicks"),
        State("query-dates", "start_date"),
        State("query-dates", "end_date"),
        State("query-classes", "value"),
        State("query-terminal", "value"),
        State("query-type", "value"),
        State("query-minconf", "value"),
        State("query-keyword", "value"),
        prevent_initial_call=True,
    )
    def apply_filters(n, start, end, classes, terminal, typ, minconf, keyword):
        tasks, total = db.query_tasks(
            date_from=start, date_to=end, classes=classes, terminals=[terminal] if terminal else None,
            task_types=[typ] if typ else None, min_conf=minconf or None, keyword=keyword,
            limit=PAGE_SIZE, offset=0,
        )
        s = db.summary(
            date_from=start, date_to=end, classes=classes, terminals=[terminal] if terminal else None,
            task_types=[typ] if typ else None, min_conf=minconf or None, keyword=keyword,
        )
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        return (
            _kpis_html(s),
            _table_html(tasks),
            f"共 {total} 条记录 · 第 1 / {pages} 页",
            {"page": 1, "total": total},
            _trend_fig(14, classes=classes, terminals=[terminal] if terminal else None, min_conf=minconf or None).figure,
            _class_fig(classes=classes, terminals=[terminal] if terminal else None, min_conf=minconf or None).figure,
            _conf_fig(classes=classes, terminals=[terminal] if terminal else None, min_conf=minconf or None).figure,
            _terminal_fig(classes=classes, min_conf=minconf or None).figure,
        )

    @app.callback(
        Output("query-table", "children", allow_duplicate=True),
        Output("query-pageinfo", "children", allow_duplicate=True),
        Output("query-page-store", "data", allow_duplicate=True),
        Input("query-prev", "n_clicks"),
        Input("query-next", "n_clicks"),
        State("query-dates", "start_date"),
        State("query-dates", "end_date"),
        State("query-classes", "value"),
        State("query-terminal", "value"),
        State("query-type", "value"),
        State("query-minconf", "value"),
        State("query-keyword", "value"),
        State("query-page-store", "data"),
        prevent_initial_call=True,
    )
    def paginate(prev_n, next_n, start, end, classes, terminal, typ, minconf, keyword, store):
        from dash import ctx

        page = (store or {}).get("page", 1)
        if ctx.triggered_id == "query-prev":
            page = max(1, page - 1)
        else:
            page = page + 1
        total = (store or {}).get("total", 0)
        pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        page = min(page, pages)
        tasks, total = db.query_tasks(
            date_from=start, date_to=end, classes=classes, terminals=[terminal] if terminal else None,
            task_types=[typ] if typ else None, min_conf=minconf or None, keyword=keyword,
            limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE,
        )
        return _table_html(tasks), f"共 {total} 条记录 · 第 {page} / {pages} 页", {
            "page": page, "total": total
        }

    @app.callback(
        Output("query-download", "data"),
        Input("query-export", "n_clicks"),
        State("query-dates", "start_date"),
        State("query-dates", "end_date"),
        State("query-classes", "value"),
        State("query-terminal", "value"),
        State("query-type", "value"),
        State("query-minconf", "value"),
        State("query-keyword", "value"),
        prevent_initial_call=True,
    )
    def export_excel(n, start, end, classes, terminal, typ, minconf, keyword):
        path = config.REPORT_DIR / f"tea_detection_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        db.export_excel(
            path,
            date_from=start,
            date_to=end,
            classes=classes,
            terminals=[terminal] if terminal else None,
            task_types=[typ] if typ else None,
            min_conf=minconf or None,
            keyword=keyword,
        )
        return dcc.send_file(str(path))
