"""Tea disease intelligent detection system - Dash entry point.

Run from the ``reproduction`` directory:

    python src/app.py

The three functional pages mirror the paper's system design:

    /detection  -> 01 检测功能（单张 / 批量 / 视频）
    /records    -> 02 数据查询功能
    /assistant  -> 03 人工智能助手分析
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dash
from dash import Input, Output, State, dcc, html
from flask import abort, jsonify, send_from_directory

from teasys import config, db
from teasys.components import media_url
from teasys.knowledge import kb_meta
from teasys.pages import assistant as assistant_page
from teasys.pages import detection as detection_page
from teasys.pages import query as query_page


def create_app() -> dash.Dash:
    db.init_db()
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
        title="智慧茶园 · 茶叶病害检测系统",
        update_title=None,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.server.config["JSON_AS_ASCII"] = False

    kb = kb_meta()

    def sidebar(pathname: str | None):
        def cls(target: str) -> str:
            return "nav-link active" if (pathname or "/").rstrip("/") == target.rstrip("/") else "nav-link"

        return html.Div(
            className="sidebar",
            children=[
                html.Div(
                    className="brand",
                    children=[
                        html.Div("🍃", className="brand-mark"),
                        html.Div([
                            html.Div("茶智检", className="brand-name"),
                            html.Div("TeaGuard · YOLOv12", className="brand-sub"),
                        ]),
                    ],
                ),
                html.Div(
                    className="nav",
                    children=[
                        dcc.Link([html.Span("🔬", className="nav-ico"), html.Span("病害检测")],
                                 href="/detection", className=cls("/detection"), id="nav-detection"),
                        dcc.Link([html.Span("📊", className="nav-ico"), html.Span("数据查询")],
                                 href="/records", className=cls("/records"), id="nav-records"),
                        dcc.Link([html.Span("🤖", className="nav-ico"), html.Span("人工智能助手")],
                                 href="/assistant", className=cls("/assistant"), id="nav-assistant"),
                    ],
                ),
                html.Div(
                    className="sidebar-foot",
                    children=[
                        html.Div([html.Span(className="dot"), "本地推理 · 数据不出园"], className="small"),
                        html.Div(f"YOLOv12n · DeepSeek-R1-14B", className="small", style={"marginTop": "4px"}),
                        html.Div(f"知识库 {kb.get('chunks', 0)} 片段", className="small"),
                        html.Div("v1.0 · 复现版", className="small", style={"marginTop": "4px", "opacity": .7}),
                    ],
                ),
            ],
        )

    app.layout = html.Div(
        className="app-shell",
        children=[
            dcc.Location(id="url"),
            html.Div(id="sidebar-container", children=sidebar("/detection")),
            html.Main(
                className="main",
                children=[
                    html.Div(id="page-content"),
                    html.Div(
                        className="small muted",
                        style={"marginTop": "28px", "textAlign": "center"},
                        children="基于 YOLOv12 + DeepSeek-R1-14B + Dash 的智慧化茶叶病害检测系统 · 研究复现",
                    ),
                ],
            ),
            dcc.Store(id="selected-task", data=None),
        ],
    )

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------
    @app.callback(
        Output("page-content", "children"),
        Output("sidebar-container", "children"),
        Output("selected-task", "data"),
        Input("url", "pathname"),
        State("url", "search"),
    )
    def route(pathname, search):
        pathname = (pathname or "/").rstrip("/") or "/"
        task_id = None
        if search:
            qs = parse_qs(search.lstrip("?"))
            task_id = (qs.get("task") or [None])[0]
        if pathname in ("/records", "/data"):
            content = query_page.layout()
        elif pathname in ("/assistant", "/ai"):
            content = assistant_page.layout(search)
        else:
            content = detection_page.layout()
        return content, sidebar("/detection" if pathname == "/" else pathname), task_id

    # ------------------------------------------------------------------
    # media / health
    # ------------------------------------------------------------------
    @app.server.route("/media/<kind>/<path:filename>")
    def media(kind: str, filename: str):
        if kind not in ("uploads", "outputs", "reports"):
            abort(404)
        base = config.DATA_DIR / kind
        return send_from_directory(base, filename)

    @app.server.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": config.MODEL_VERSION,
                "db": str(config.DB_PATH),
                "knowledge_chunks": kb.get("chunks", 0),
                "time": datetime.now().isoformat(timespec="seconds"),
            }
        )

    @app.server.errorhandler(404)
    def not_found(e):
        return "<h3 style='font-family:sans-serif'>页面不存在</h3>", 404

    detection_page.register_callbacks(app)
    query_page.register_callbacks(app)
    assistant_page.register_callbacks(app)

    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    print(f" * 智慧茶园系统启动: http://{config.APP_HOST}:{config.APP_PORT}")
    app.run(host=config.APP_HOST, port=config.APP_PORT, debug=False, use_reloader=False)
