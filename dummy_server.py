"""OpenAI互換APIのダミーサーバー(LLM不使用)。

固定メッセージを返す。応答本文は ``generate_message()`` コールバックを差し替える
だけで任意のロジックに変更できる。TTFT / tok/s / ストリーミングの各シミュレーション
はCLIオプションで切り替える。

    python dummy_server.py                       # 全OFF: 即時応答
    python dummy_server.py --ttft 0.4            # TTFT 400ms
    python dummy_server.py --tps 25              # 25 tokens/sec
    python dummy_server.py --ttft 0.4 --tps 25   # 両方ON
    python dummy_server.py --stream force        # stream:falseでも常にSSEで返す
    python dummy_server.py --stream never        # stream:trueでも常にJSONで返す

    python dummy_server.py --help                # 全オプション

依存: pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("dummy_server")

app = FastAPI(title="dummy-llm-server")


# ---------------------------------------------------------------------------
# 設定(CLIオプションで上書きされる)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    ttft: float = 0.0          # TTFTシミュレーション: 最初のトークンまでの遅延(秒)。0 = OFF
    tps: float = 0.0           # tok/sシミュレーション: 送出レート(tokens/sec)。0 = OFF(即時)
    stream_mode: str = "auto"  # ストリーミング: auto=リクエスト準拠 / force=常にSSE / never=常にJSON
    chunk_size: int = 4        # 1トークンとみなす文字数
    model: str = "dummy-llm"   # 応答するモデル名


config = Config()


# ---------------------------------------------------------------------------
# 応答本文コールバック(ここを差し替えるだけで挙動を変えられる)
# ---------------------------------------------------------------------------
DUMMY_MESSAGE = (
    "これはOpenAI互換ダミーサーバーからの固定応答です。"
    "LLMは一切使用していません。"
)


def generate_message(request: dict[str, Any]) -> str:
    """応答本文を生成するコールバック。

    ``request`` は /v1/chat/completions のリクエストボディ(dict)。
    固定メッセージ以外にしたいときは、この関数の中身だけを書き換える。
    """
    return DUMMY_MESSAGE


# ---------------------------------------------------------------------------
# 共通パーツ
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """text を chunk_size 文字ずつの擬似トークン列に分割する。"""
    step = max(1, config.chunk_size)
    return [text[i : i + step] for i in range(0, len(text), step)]


def count_prompt_tokens(body: dict[str, Any]) -> int:
    """リクエストの messages から擬似プロンプトトークン数を数える。"""
    messages = body.get("messages")
    n = 0
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                n += len(tokenize(m["content"]))
    return n


def fake_usage(body: dict[str, Any], tokens: list[str]) -> dict[str, int]:
    prompt = count_prompt_tokens(body)
    completion = len(tokens)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def decide_stream(body: dict[str, Any]) -> bool:
    """ストリーミングモードに応じて SSE / JSON を決める。"""
    if config.stream_mode == "force":
        return True
    if config.stream_mode == "never":
        return False
    return bool(body.get("stream", False))


async def paced_tokens(tokens: list[str]) -> AsyncIterator[str]:
    """tok/sシミュレーション: 1/config.tps 秒間隔でトークンを送出する。

    絶対時刻ベースのスケジュールで送るため、sleep の丸め誤差が蓄積しない。
    tps <= 0 なら即時にすべて送出する。
    """
    interval = 1.0 / config.tps if config.tps > 0 else 0.0
    start = time.monotonic()
    for i, tok in enumerate(tokens):
        if interval:
            delay = start + i * interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        yield tok


def sse_line(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    """OpenAI形式のエラーレスポンス。"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


# ---------------------------------------------------------------------------
# 応答生成(ストリーミング / 非ストリーミング)
# ---------------------------------------------------------------------------
async def stream_chat(
    body: dict[str, Any], model: str, tokens: list[str]
) -> AsyncIterator[str]:
    """SSEで chat.completion.chunk を送出する。"""
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    stream_options = body.get("stream_options")
    include_usage = isinstance(stream_options, dict) and bool(
        stream_options.get("include_usage")
    )

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    # TTFTシミュレーション: 最初のトークンまで沈黙する
    if config.ttft > 0:
        await asyncio.sleep(config.ttft)

    yield sse_line(chunk({"role": "assistant", "content": ""}))
    async for tok in paced_tokens(tokens):
        yield sse_line(chunk({"content": tok}))
    yield sse_line(chunk({}, finish_reason="stop"))

    if include_usage:
        yield sse_line(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": fake_usage(body, tokens),
            }
        )
    yield "data: [DONE]\n\n"


async def complete_chat(
    body: dict[str, Any], model: str, tokens: list[str]
) -> dict[str, Any]:
    """非ストリーミング: 生成時間ぶん待ってから完全なJSONを返す。

    待ち時間 = TTFT + (最後のトークンが送出されるまでの時間) とし、
    ストリーミング時の完了タイミングと一致させる。
    """
    generation = (len(tokens) - 1) / config.tps if config.tps > 0 and len(tokens) > 1 else 0.0
    total = config.ttft + generation
    if total > 0:
        await asyncio.sleep(total)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(tokens)},
                "finish_reason": "stop",
            }
        ],
        "usage": fake_usage(body, tokens),
    }


# ---------------------------------------------------------------------------
# エンドポイント
# ---------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body: Any = await request.json()
    except Exception:
        return error_response("リクエストボディは有効なJSONである必要があります。")
    if not isinstance(body, dict):
        return error_response("リクエストボディはJSONオブジェクトである必要があります。")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return error_response("'messages' は空でない配列である必要があります。")

    model = body.get("model") or config.model
    tokens = tokenize(generate_message(body))
    stream = decide_stream(body)
    log.info(
        "chat/completions: mode=%s ttft=%s tps=%s chunks=%d",
        "sse" if stream else "json",
        f"{config.ttft:.3f}s" if config.ttft > 0 else "off",
        f"{config.tps:.1f}tok/s" if config.tps > 0 else "off",
        len(tokens),
    )

    if stream:
        return StreamingResponse(
            stream_chat(body, model, tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(await complete_chat(body, model, tokens))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dummy-llm-server",
            }
        ],
    }


# ---------------------------------------------------------------------------
# 起動
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI互換APIのダミーサーバー(LLM不使用)。応答は generate_message() コールバックで差し替え可能。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="バインドするホスト")
    parser.add_argument("--port", type=int, default=8000, help="バインドするポート")
    parser.add_argument(
        "--ttft", type=float, default=0.0,
        help="TTFTシミュレーション: 最初のトークンまでの遅延(秒)。0でOFF",
    )
    parser.add_argument(
        "--tps", type=float, default=0.0,
        help="tok/sシミュレーション: 送出レート(tokens/sec)。0でOFF(即時)",
    )
    parser.add_argument(
        "--stream", choices=("auto", "force", "never"), default="auto",
        help="ストリーミングシミュレーション: auto=リクエスト準拠 / force=常にSSE / never=常にJSON",
    )
    parser.add_argument("--chunk-size", type=int, default=4, help="1トークンとみなす文字数")
    parser.add_argument("--model", default="dummy-llm", help="応答するモデル名")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config.ttft = max(0.0, args.ttft)
    config.tps = max(0.0, args.tps)
    config.stream_mode = args.stream
    config.chunk_size = max(1, args.chunk_size)
    config.model = args.model

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
