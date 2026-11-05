"""Reusable UI building blocks for the Dash application."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

from dash import dcc, html

from . import config

LEVEL_CN = {
    "none": "正常",
    "info": "黄色提示",
    "warning": "橙色预警",
    "danger": "红色预警",
}
LEVEL_ICON = {"none": "🌿", "info": "🟡", "warning": "🟠", "danger": "🔴"}
LEVEL_CLASS = {"none": "none", "info": "info", "warning": "warning", "danger": "danger"}


def media_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path).resolve()
    for kind in ("uploads", "outputs", "reports"):
        base = (config.DATA_DIR / kind).resolve()
        try:
            rel = p.relative_to(base)
            return f"/media/{kind}/{quote(rel.as_posix())}"
        except ValueError:
            continue
    return None


def page_header(title: str, subtitle: str, chips: Sequence[Any] | None = None):
    return html.Div(
        className="topbar",
        children=[
            html.Div([html.H1(title), html.Div(subtitle, className="sub")]),
            html.Div(className="topbar-right", children=list(chips or [])),
        ],
    )


def chip(label: str, value: str | None = None, dot: bool = False):
    children = []
    if dot:
        children.append(html.Span(className="dot"))
    if value is None:
        children.append(html.Span(label))
    else:
        children.extend([html.Span(label), html.Strong(value)])
    return html.Span(className="chip", children=children)


def kpi_card(label: str, value: Any, unit: str = "", foot: str = "", tone: str = ""):
    return html.Div(
        className=f"card kpi {tone}",
        children=[
            html.Div(label, className="label"),
            html.Div([html.Span(str(value)), html.Small(unit) if unit else None], className="value"),
            html.Div(foot, className="foot") if foot else None,
        ],
    )


def badge(text: str, tone: str = "green"):
    return html.Span(text, className=f"badge {tone}")


def level_badge(level: str):
    return badge(f"{LEVEL_ICON.get(level, '')} {LEVEL_CN.get(level, level)}", {
        "danger": "red", "warning": "orange", "info": "gold", "none": "green"
    }.get(level, "gray"))


def class_badge(name: str):
    cn = config.CLASS_NAMES_CN.get(name, name)
    tone = {"algal-leaf-spot": "green", "brown-blight": "red", "grey-blight": "gray"}.get(name, "blue")
    return badge(cn, tone)


def card(children, title: str | None = None, subtitle: str | None = None, className: str = "", actions=None):
    head = None
    if title or actions:
        head = html.Div(
            className="card-head",
            children=[
                html.Div(
                    [html.Span(className="bar"), html.Span(title or "")],
                    className="card-title",
                ) if title else html.Div(),
                html.Div(actions) if actions else None,
            ],
        )
    # Dash 4 does not flatten a nested list passed as a single child entry;
    # flatten explicitly so every element becomes a direct child.
    body = list(children) if isinstance(children, (list, tuple)) else [children]
    inner = [head, html.Div(subtitle, className="card-sub") if subtitle else None]
    return html.Div(
        className=f"card {className}",
        children=[c for c in inner if c is not None] + body,
    )


def alert_banner(level: str, message: str):
    cls = LEVEL_CLASS.get(level, "none")
    return html.Div(
        className=f"alert-banner {cls}",
        children=[html.Span(LEVEL_ICON.get(level, "ℹ️"), className="ico"), html.Div(message)],
    )


def data_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    empty_text: str = "暂无数据",
    max_height: str | None = None,
):
    rows = list(rows)
    if not rows:
        return html.Div(empty_text, className="empty")
    wrap_style = {"maxHeight": max_height, "overflowY": "auto"} if max_height else None
    return html.Div(
        className="table-wrap",
        style=wrap_style,
        children=html.Table(
            className="data",
            children=[
                html.Thead(html.Tr([html.Th(h) for h in headers])),
                html.Tbody([html.Tr([html.Td(c) for c in r]) for r in rows]),
            ],
        ),
    )


def stat_row(items: Sequence[tuple[str, Any]]):
    return html.Div(
        className="grid grid-4",
        children=[
            html.Div(
                className="card",
                children=[
                    html.Div(k, className="label small muted"),
                    html.Div(str(v), style={"fontSize": "20px", "fontWeight": 700, "color": "#1b5138"}),
                ],
            )
            for k, v in items
        ],
    )


def empty_illustration(icon: str, title: str, hint: str):
    return html.Div(
        className="empty",
        children=[
            html.Div(icon, style={"fontSize": "42px", "marginBottom": "8px"}),
            html.Div(title, style={"fontWeight": 600, "color": "#5f7a6b", "fontSize": "14px"}),
            html.Div(hint, style={"fontSize": "12px", "marginTop": "4px"}),
        ],
    )
