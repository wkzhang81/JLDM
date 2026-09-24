#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenJev · Choice<T> 决策命令行工具
==================================
对应 jev_like_test.html 的输入输出模式，通过 JSON 一次给出决策结果与后验概率。

输入 JSON 示例：
{
  "reference": "我在凌晨 3 点收到一封来自陌生 IP 的登录告警邮件……",
  "question": "将以下用户工单分类到正确的处理部门。",
  "options": ["账单计费", "技术支持", "安全事件", "人工审核"],
  "images": ["/path/to/screenshot.png"],          // 可选，本地图片路径列表
  "mode": "local",                                 // local=KoboldCpp / cloud=OpenAI兼容
  "base_url": "http://localhost:5001/v1",
  "model": "koboldcpp/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-Uncensored-MTP",
  "api_key": "",                                   // 本地可留空
  "temperature": 1.5                               // 概率校准温度
}

用法：
  python3 jev_decision.py --input input.json
  python3 jev_decision.py --input input.json --output result.json --debug
  echo '{"reference":"...","question":"...","options":["A","B"]}' | python3 jev_decision.py --input -
"""

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request


# --------------------------------------------------------------------------- #
# HTTP 工具
# --------------------------------------------------------------------------- #
def http_post(url, headers, body, timeout=180):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:1500]
        raise SystemExit(f"[HTTP {e.code}] {url}\n{detail}")
    except urllib.error.URLError as e:
        raise SystemExit(f"[连接失败] {url}: {e.reason}")


# --------------------------------------------------------------------------- #
# Prompt 构造（与前端 buildPrompt 完全一致）
# --------------------------------------------------------------------------- #
def build_prompt(reference, question, options):
    state = {}
    if reference:
        state["reference"] = reference
    if question:
        state["question"] = question
    letters = [chr(65 + i) for i in range(len(options))]
    option_lines = "\n".join(f"{letters[i]}. {options[i]}" for i in range(len(options)))
    return (
        "Given the following state:\n"
        + json.dumps(state, indent=2, ensure_ascii=False)
        + "\n\nEvaluation criteria:\n" + question
        + "\n\nSelect the single best option from the list below:\n"
        + option_lines
        + "\n\nReply with ONLY the option letter (e.g. A, B, C):"
    )


# --------------------------------------------------------------------------- #
# 图片 -> data URL（与前端 FileReader.readAsDataURL 行为一致）
# --------------------------------------------------------------------------- #
def to_data_url(path):
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# 概率数学
# --------------------------------------------------------------------------- #
def logaddexp(a, b):
    if a is None:
        return b
    if b is None:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def softmax(logits, temperature):
    keys = list(logits.keys())
    m = max(logits[k] for k in keys)
    exps = {k: math.exp((logits[k] - m) / temperature) for k in keys}
    s = sum(exps.values())
    return {k: exps[k] / s for k in keys}


def norm_tok(t):
    return str(t if t is not None else "").strip().upper()


# --------------------------------------------------------------------------- #
# logprobs 解析（云端 OpenAI 与本地 KoboldCpp 共用）
# --------------------------------------------------------------------------- #
def parse_positions(lp):
    positions = []
    if isinstance(lp, dict) and isinstance(lp.get("content"), list) and lp["content"]:
        for idx, step in enumerate(lp["content"]):
            flat = {}
            if isinstance(step, dict) and isinstance(step.get("top_logprobs"), list):
                for item in step["top_logprobs"]:
                    if isinstance(item, dict) and item.get("token") is not None \
                            and item.get("logprob") is not None:
                        flat[str(item["token"])] = float(item["logprob"])
            if isinstance(step, dict) and step.get("token") is not None \
                    and step.get("logprob") is not None \
                    and str(step["token"]) not in flat:
                flat[str(step["token"])] = float(step["logprob"])
            selected = str(step["token"]) if isinstance(step, dict) \
                and step.get("token") is not None else None
            positions.append({"idx": idx, "selected": selected, "top": flat})
    elif isinstance(lp, dict) and isinstance(lp.get("top_logprobs"), list):
        tokens = lp.get("tokens") or []
        for idx, pos_tokens in enumerate(lp["top_logprobs"]):
            flat = {}
            if isinstance(pos_tokens, dict):
                for k, v in pos_tokens.items():
                    try:
                        flat[str(k)] = float(v)
                    except (TypeError, ValueError):
                        pass
            selected = str(tokens[idx]) if idx < len(tokens) else None
            positions.append({"idx": idx, "selected": selected, "top": flat})
    return positions


def extract_decision(positions, letters, msg_content):
    """精确定位答题 token，把同字母的多个变体 logprob 做 logsumexp。"""
    answer_pos = None
    for p in positions:
        if p["selected"] is not None and norm_tok(p["selected"]) in letters:
            answer_pos = p
            break
    used_fallback = False
    if answer_pos is None:
        trimmed = str(msg_content or "").strip().upper()
        final = None
        for L in letters:
            if trimmed == L or trimmed.endswith("\n" + L) or trimmed.endswith(" " + L):
                final = L
                break
        if final:
            for p in reversed(positions):
                if any(norm_tok(k) == final for k in p["top"]):
                    answer_pos = p
                    used_fallback = True
                    break
    if answer_pos is None:
        raise RuntimeError(
            "模型未按 A/B/C… 格式生成候选字母。模型输出: " + repr(msg_content)
        )
    letter_logps = {}
    for tok, v in answer_pos["top"].items():
        n = norm_tok(tok)
        if n in letters and math.isfinite(v):
            letter_logps[n] = logaddexp(letter_logps.get(n), v)
    return answer_pos, letter_logps, used_fallback


# --------------------------------------------------------------------------- #
# 后端：本地 KoboldCpp（/api/v1/generate + grammar + /api/extra/last_logprobs）
# --------------------------------------------------------------------------- #
def run_local(cfg, prompt, images, letters, option_map, temperature):
    base = cfg["base_url"].rstrip("/")
    origin = base[:base.rfind("/v1")] if "/v1" in base else base
    gen_url = origin + "/api/v1/generate"
    lp_url = origin + "/api/extra/last_logprobs"
    headers = {}
    if cfg.get("api_key"):
        headers["Authorization"] = "Bearer " + cfg["api_key"]

    allowed = "".join(letters)
    grammar = f"root ::= [{allowed}]"
    local_prompt = (
        prompt
        + "\nDo not think or explain. Output exactly one letter from "
        + "/".join(letters) + ".\nAnswer:"
    )
    body = {
        "prompt": local_prompt,
        "max_length": 1,
        "temperature": 1.0,
        "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "typical": 1.0, "tfs": 1.0, "top_a": 0.0,
        "rep_pen": 1.0,
        "grammar": grammar,
        "grammar_retain_state": False,
        "logprobs": True,
        "stream": False,
    }
    if images:
        body["images"] = images

    gen = http_post(gen_url, headers, body)
    gen_text = ""
    try:
        gen_text = str(gen["results"][0].get("text", ""))
    except (KeyError, IndexError, TypeError):
        raise SystemExit("/api/v1/generate 返回异常: " + json.dumps(gen, ensure_ascii=False)[:800])

    lp_resp = http_post(lp_url, headers, {})
    lp = lp_resp.get("logprobs") if isinstance(lp_resp, dict) else None
    if not lp or (not isinstance(lp.get("content"), list)
                  and not isinstance(lp.get("top_logprobs"), list)):
        raise SystemExit(
            "KoboldCpp 未返回 last_logprobs。原始返回: "
            + json.dumps(lp_resp, ensure_ascii=False)[:1000]
        )
    positions = parse_positions(lp)
    return positions, gen_text, {"mode": "grammar", "grammar": grammar,
                                "request": body, "raw_generate": gen}


# --------------------------------------------------------------------------- #
# 后端：云端 OpenAI 兼容 /chat/completions（vision + logprobs）
# --------------------------------------------------------------------------- #
def run_cloud(cfg, prompt, images, letters, option_map, temperature):
    endpoint = cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {}
    if cfg.get("api_key"):
        headers["Authorization"] = "Bearer " + cfg["api_key"]

    if images:
        user_content = [{"type": "text", "text": prompt}]
        for b64 in images:
            user_content.append({"type": "image_url", "image_url": {"url": b64}})
    else:
        user_content = prompt

    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system",
             "content": "You are a precise decision engine. Reply with ONLY the single option letter, nothing else."},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 512,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    data = http_post(endpoint, headers, body)
    choice = (data.get("choices") or [{}])[0]
    lp = choice.get("logprobs") or {}
    msg_content = ""
    if choice.get("message") and isinstance(choice["message"].get("content"), str):
        msg_content = choice["message"]["content"]
    elif choice.get("text"):
        msg_content = choice["text"]
    positions = parse_positions(lp)
    return positions, msg_content, {"mode": "openai_chat", "request": body}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
# 单条决策：传入一条合并后的配置 dict，返回结果 dict；出错抛异常
# --------------------------------------------------------------------------- #
def decide(cfg, debug=False):
    reference = (cfg.get("reference") or "").strip()
    question = (cfg.get("question") or "").strip()
    options = [o.strip() for o in (cfg.get("options") or []) if str(o).strip()]
    if not options:
        raise ValueError("options 不能为空。")
    if not question:
        raise ValueError("question 不能为空。")

    mode = (cfg.get("mode") or "local").lower()
    if mode not in ("local", "cloud"):
        raise ValueError("mode 只能是 local 或 cloud。")
    if not cfg.get("base_url"):
        raise ValueError("缺少 base_url。")
    if mode == "cloud" and not cfg.get("model"):
        raise ValueError("cloud 模式需要 model 字段。")
    if mode == "local" and len(options) > 5:
        raise ValueError("Local JEV Grammar 模式当前最多支持 5 个候选"
                         "（KoboldCpp 1.121 last_logprobs 上限）。")

    temperature = float(cfg.get("temperature", 1.5))

    images = []
    for p in (cfg.get("images") or []):
        if not os.path.exists(p):
            raise FileNotFoundError(f"图片不存在: {p}")
        images.append(to_data_url(p))

    letters = [chr(65 + i) for i in range(len(options))]
    option_map = {letters[i]: options[i] for i in range(len(options))}
    prompt = build_prompt(reference, question, options)

    t0 = time.time()
    if mode == "local":
        positions, msg_content, meta = run_local(
            cfg, prompt, images, letters, option_map, temperature)
    else:
        positions, msg_content, meta = run_cloud(
            cfg, prompt, images, letters, option_map, temperature)
    elapsed_ms = (time.time() - t0) * 1000

    if not positions:
        raise RuntimeError("响应中未包含可解析的 logprobs。")

    answer_pos, letter_logps, used_fallback = extract_decision(
        positions, letters, msg_content)

    extracted = {}
    missing = []
    for L in letters:
        if letter_logps.get(L) is not None and math.isfinite(letter_logps[L]):
            extracted[option_map[L]] = letter_logps[L]
        else:
            missing.append(option_map[L])
    if not extracted:
        raise RuntimeError("已定位答题 token，但该位置没有候选字母 logprob。")

    probs_obs = softmax(extracted, temperature)
    probs = {opt: probs_obs.get(opt, 0.0) for opt in options}
    winner = max(probs_obs, key=lambda k: probs_obs[k])

    result = {
        "winner": winner,
        "confidence": round(probs_obs[winner], 4),
        "probabilities_pct": {k: round(v * 100, 2) for k, v in probs.items()},
        "elapsed_ms": round(elapsed_ms, 1),
        "mode": mode,
        "image_count": len(images),
        "answer_token_pos": answer_pos["idx"],
        "used_fallback": used_fallback,
    }
    if missing:
        result["missing_options"] = missing
    if debug:
        result["debug"] = {
            "prompt": prompt,
            "msg_content": msg_content,
            "grammar": meta.get("grammar"),
            "letter_logprobs": {k: round(v, 4) for k, v in letter_logps.items()},
            "candidate_logprobs": {k: round(v, 4) for k, v in extracted.items()},
            "positions": [
                {"idx": p["idx"], "selected": p["selected"],
                 "top": {k: round(v, 3) for k, v in p["top"].items()}}
                for p in positions
            ],
        }
    return result


# --------------------------------------------------------------------------- #
# 主流程：支持三种输入形态
#   1) 单条对象 {...}
#   2) 数组 [{...}, {...}]
#   3) 带共享连接配置的对象 {"base_url":..., "items":[{...}, ...]}
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="OpenJev Choice<T> 决策脚本（支持批量）")
    ap.add_argument("--input", "-i", required=True,
                    help="输入 JSON 文件路径，或 - 表示从 stdin 读取")
    ap.add_argument("--output", "-o",
                    help="把完整结果写入该 JSON 文件（同时仍打印到 stdout）")
    ap.add_argument("--debug", action="store_true", help="输出原始 logprobs 调试信息")
    args = ap.parse_args()

    if args.input == "-":
        payload = json.loads(sys.stdin.read())
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)

    # 解析出 (共享配置, 任务列表)
    if isinstance(payload, list):
        shared, items = {}, payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        shared = {k: v for k, v in payload.items() if k != "items"}
        items = payload["items"]
    elif isinstance(payload, dict):
        shared, items = {}, [payload]
    else:
        raise SystemExit("输入必须是 JSON 对象或数组。")

    results = []
    for idx, item in enumerate(items):
        merged = dict(shared)
        merged.update(item or {})
        try:
            r = decide(merged, debug=args.debug)
            r["index"] = idx
            if item and item.get("id") is not None:
                r["id"] = item["id"]
            results.append(r)
        except SystemExit as e:           # http_post 里的网络错误
            results.append({"index": idx, "error": str(e)})
        except Exception as e:
            results.append({"index": idx, "error": f"{type(e).__name__}: {e}"})

    # 单条对象输入时保持扁平输出，数组/items 批量输入时输出数组
    out_obj = results[0] if (isinstance(payload, dict) and not isinstance(payload.get("items"), list)) else results

    out = json.dumps(out_obj, ensure_ascii=False, indent=2)
    print(out)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)


if __name__ == "__main__":
    main()
