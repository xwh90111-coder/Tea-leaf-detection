"""RAG assistant: retrieval + prompt construction + local LLM invocation.

Two entry points are used by the Dash assistant page:

* ``answer_question``  - multi-turn Q&A grounded in D1 detection data and D2
  knowledge snippets.
* ``generate_report``  - one-click structured diagnosis / control report.

If the local model service is unavailable the module falls back to a
transparent, deterministic knowledge-base answer so that the system remains
usable during demonstrations and offline deployment.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterable

from . import config
from .knowledge import get_kb
from .llm import get_client

SYSTEM_PROMPT = """你是“智慧茶园”茶叶病害检测系统的植保专家助手，服务于茶园管理者。
你的任务是把 YOLOv12 的检测结果与本地茶叶知识库结合，给出准确、可执行的诊断和防治建议。

必须遵守的规则：
1. 只依据【检测结果】和【知识库片段】作答；知识库没有的内容要明确说明，不得编造。
   当知识库片段给出病原、症状或药剂时，以片段为准，禁止使用与片段冲突的模型记忆（例如不得把茶褐枯病病原写成其他属种）。
2. 农药名称、剂量、安全间隔期必须以知识库片段或产品标签为准；不确定时提示用户咨询当地植保部门。
3. 区分“系统检测的严重程度/预警等级”与农学上的“病叶率、病情指数”，不要混为一谈。
4. 使用简体中文和 Markdown 组织答案，用分点列表，不要使用表格。
5. 引用知识库时在相应句子末尾标注片段编号，例如 [1]、[2]。
6. 对低置信度（<0.5）的检测结果要提示人工复核。
7. 回答控制在 400～700 字，重点突出，避免空话；如果时间或篇幅有限，优先保证“诊断结论 + 防治措施”两部分完整。"""

REPORT_PROMPT = """请基于下面的检测结果和知识库片段，生成一份茶园病害诊断与防治报告。
报告使用 Markdown，必须包含以下小节（不要使用表格）：
## 一、检测概况
## 二、结果解读
## 三、病害诊断
## 四、防治建议
## 五、预警与后续监测
## 六、参考依据

要求：结论明确、措施可执行；农药使用注明“以产品标签和当地登记为准”；引用片段用 [编号] 标注。"""


def _strip_code_fence(text: str) -> str:
    """Remove a leading ```markdown / ``` fence that the model sometimes adds."""
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _fmt_task(task: dict[str, Any] | None) -> str:
    if not task:
        return "（本次对话未绑定具体检测任务）"
    counts = task.get("class_counts") or {}
    if isinstance(counts, str):
        try:
            counts = json.loads(counts)
        except Exception:
            counts = {}
    cn = {config.CLASS_NAMES_CN.get(k, k): v for k, v in counts.items()}
    lines = [
        f"任务ID：{task.get('id')}",
        f"检测类型：{ {'single': '单张图像', 'batch': '批量图像', 'video': '视频流'}.get(task.get('task_type'), task.get('task_type')) }",
        f"文件：{task.get('source_name')}",
        f"终端：{task.get('terminal_id')}",
        f"时间：{task.get('created_at')}",
        f"病灶总数：{task.get('total_objects', 0)}",
        f"类别统计：{json.dumps(cn, ensure_ascii=False)}",
        f"最高置信度：{float(task.get('max_confidence') or 0):.3f}",
        f"平均置信度：{float(task.get('avg_confidence') or 0):.3f}",
        f"系统严重程度：{task.get('severity', '无')}",
        f"预警等级：{ {'none': '无', 'info': '黄色提示', 'warning': '橙色预警', 'danger': '红色预警'}.get(task.get('alert_level'), task.get('alert_level')) }",
    ]
    dets = task.get("detections") or []
    if dets:
        top = sorted(dets, key=lambda d: d.get("confidence", 0), reverse=True)[:5]
        lines.append(
            "主要病灶："
            + "；".join(
                f"{config.CLASS_NAMES_CN.get(d.get('class_name'), d.get('class_name'))}"
                f" 置信度{d.get('confidence', 0):.3f}"
                for d in top
            )
        )
    return "\n".join(lines)


def _format_contexts(contexts: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{i+1}] {c['title']} · {c['section']}（来源：{c['source']}）\n{c['text']}"
        for i, c in enumerate(contexts)
    )


def _sources_payload(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "idx": i + 1,
            "title": c["title"],
            "section": c["section"],
            "source": c["source"],
            "score": c.get("dense_score"),
            "text": c["text"],
        }
        for i, c in enumerate(contexts)
    ]


def _retrieve(question: str, task: dict[str, Any] | None, k: int | None = None) -> list[dict[str, Any]]:
    kb = get_kb()
    q = question
    if task:
        names = list((task.get("class_counts") or {}).keys())
        cn_names = [config.CLASS_NAMES_CN.get(n, n) for n in names]
        if cn_names:
            q = "、".join(cn_names) + " " + q
    return kb.search(q, k=k or config.TOP_K)


def _offline_answer(
    question: str, task: dict[str, Any] | None, contexts: list[dict[str, Any]]
) -> str:
    parts = [
        "> 当前未连接本地大模型服务，以下为**离线知识库检索模式**生成的参考建议（非大模型生成）。",
        "",
        "## 一、结论",
    ]
    if task and task.get("total_objects"):
        counts = task.get("class_counts") or {}
        cn = {config.CLASS_NAMES_CN.get(k, k): v for k, v in counts.items()}
        desc = "、".join(f"{k} {v} 处" for k, v in cn.items())
        parts.append(
            f"本次检测共发现病灶 {task.get('total_objects')} 处（{desc}），"
            f"系统判定严重程度为**{task.get('severity')}**，"
            f"预警等级为**{ {'info': '黄色提示', 'warning': '橙色预警', 'danger': '红色预警', 'none': '无'}.get(task.get('alert_level'), task.get('alert_level')) }**。"
        )
    else:
        parts.append("未绑定具体检测任务，以下建议基于知识库检索结果。")

    parts.append("\n## 二、知识库要点")
    if contexts:
        for i, c in enumerate(contexts[:3], 1):
            text = c["text"].split("：", 1)[-1]
            parts.append(f"{i}. **{c['title']} · {c['section']}**：{text[:260]}… [{i}]")
    else:
        parts.append("未检索到相关知识条目。")

    parts.append("\n## 三、建议措施")
    sev = (task or {}).get("severity", "")
    if sev == "重度":
        parts.append("- 立即复核检测区域，必要时采样送检，确认病原种类。")
        parts.append("- 尽快清除病叶、病枝并带出茶园处理，减少菌源。")
        parts.append("- 在植保人员指导下选用登记药剂应急防治，注意轮换用药和安全间隔期。")
    elif sev == "中度":
        parts.append("- 7 日内安排防治，优先农业措施（修剪、排水、平衡施肥）。")
        parts.append("- 发病初期可选用保护性或内吸性杀菌剂，7～10 天 1 次，连喷 2～3 次。")
    elif sev == "轻度":
        parts.append("- 持续监测，7 天后复查；加强通风透光和排水。")
    else:
        parts.append("- 保持常规巡查，做好清园和平衡施肥，预防病害发生。")
    parts.append("- 低置信度（<0.5）结果建议人工复核；单张图像结果仅代表当前视野。")

    parts.append("\n## 四、参考来源")
    for i, c in enumerate(contexts, 1):
        parts.append(f"- [{i}] {c['title']} · {c['section']}（{c['source']}）")
    if not contexts:
        parts.append("- 暂无")
    return "\n".join(parts)


def answer_question(
    question: str,
    task: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Run one RAG turn and return answer + sources + runtime metadata."""
    t0 = time.perf_counter()
    contexts = _retrieve(question, task, k)
    user_prompt = (
        f"【检测结果】\n{_fmt_task(task)}\n\n"
        f"【知识库片段】\n{_format_contexts(contexts) if contexts else '（无匹配片段）'}\n\n"
        f"【用户问题】\n{question}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})

    client = get_client()
    try:
        result = client.chat(messages)
        if not result.content:
            raise RuntimeError("空回复")
        return {
            "answer": _strip_code_fence(result.content),
            "sources": _sources_payload(contexts),
            "model": result.model,
            "provider": result.provider,
            "latency_s": result.latency_s,
            "fallback": False,
        }
    except Exception as e:
        return {
            "answer": _offline_answer(question, task, contexts),
            "sources": _sources_payload(contexts),
            "model": "offline-knowledge-base",
            "provider": "offline",
            "latency_s": round(time.perf_counter() - t0, 2),
            "fallback": True,
            "error": str(e)[:200],
        }


def generate_report(task: dict[str, Any], extra: str = "") -> dict[str, Any]:
    """Generate a structured diagnosis / control report for one detection task."""
    t0 = time.perf_counter()
    question = "请生成本次检测的病害诊断与防治报告"
    contexts = _retrieve(question + " 症状 防治 预警", task, k=max(config.TOP_K, 5))
    prompt = (
        f"【检测结果】\n{_fmt_task(task)}\n\n"
        f"【知识库片段】\n{_format_contexts(contexts)}\n\n"
        f"【补充说明】\n{extra or '无'}\n\n{REPORT_PROMPT}"
    )
    client = get_client()
    try:
        result = client.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        if not result.content:
            raise RuntimeError("空回复")
        answer = _strip_code_fence(result.content)
        meta = {
            "model": result.model,
            "provider": result.provider,
            "latency_s": result.latency_s,
            "fallback": False,
        }
    except Exception as e:
        # deterministic fallback report
        answer = _offline_answer(question, task, contexts)
        meta = {
            "model": "offline-knowledge-base",
            "provider": "offline",
            "latency_s": round(time.perf_counter() - t0, 2),
            "fallback": True,
            "error": str(e)[:200],
        }
    return {"report": answer, "sources": _sources_payload(contexts), **meta}


def analyze_file(filename: str, content: str, question: str = "") -> dict[str, Any]:
    """“文件理解”: use a text report as additional context for the assistant."""
    clipped = content[:6000]
    q = question or f"请解读文件《{filename}》中的检测结果，并给出诊断和防治建议"
    contexts = _retrieve(q + " " + clipped[:500], None, k=config.TOP_K)
    prompt = (
        f"【文件内容】《{filename}》\n{clipped}\n\n"
        f"【知识库片段】\n{_format_contexts(contexts)}\n\n"
        f"【用户问题】\n{q}"
    )
    client = get_client()
    try:
        result = client.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        return {
            "answer": result.content,
            "sources": _sources_payload(contexts),
            "model": result.model,
            "latency_s": result.latency_s,
            "fallback": False,
        }
    except Exception:
        return {
            "answer": _offline_answer(q, None, contexts),
            "sources": _sources_payload(contexts),
            "model": "offline-knowledge-base",
            "latency_s": 0,
            "fallback": True,
        }
