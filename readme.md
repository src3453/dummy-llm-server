# dummy-llm-server

OpenAI互換APIのダミーサーバー(LLM不使用)。固定メッセージを返します。

実際のモデルに接続せずに、TTFT(最初のトークンまでの遅延)・tok/s(送出レート)・
ストリーミング/非ストリーミングといった応答タイミングの挙動をシミュレーションできます。
OpenAI SDK や既存のクライアントの `base_url` を向けて、ネットワーク周りや
ストリーミング処理の動作確認・負荷試験などに使います。

## セットアップ

```bash
pip install -r requirements.txt
```

依存は `fastapi` と `uvicorn` のみです(Python 3.10+、型ヒントに `from __future__ import annotations` を使用)。

## 起動

```bash
python dummy_server.py                       # 全OFF: 即時応答
python dummy_server.py --ttft 0.4            # TTFT 400ms
python dummy_server.py --tps 25              # 25 tokens/sec
python dummy_server.py --ttft 0.4 --tps 25   # 両方ON
python dummy_server.py --stream force        # stream:falseでも常にSSEで返す
python dummy_server.py --stream never        # stream:trueでも常にJSONで返す

python dummy_server.py --help                # 全オプション
```

デフォルトは `http://127.0.0.1:8000` で起動します。

### CLIオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--host` | `127.0.0.1` | バインドするホスト |
| `--port` | `8000` | バインドするポート |
| `--ttft` | `0.0` | 最初のトークンまでの遅延(秒)。`0` でOFF |
| `--tps` | `0.0` | 送出レート(tokens/sec)。`0` でOFF(即時) |
| `--stream` | `auto` | `auto`=リクエスト準拠 / `force`=常にSSE / `never`=常にJSON |
| `--chunk-size` | `4` | 1トークンとみなす文字数 |
| `--model` | `dummy-llm` | 応答するモデル名 |

## エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | チャット補完(ストリーミング/非ストリーミング両対応) |
| `GET` | `/v1/models` | モデル一覧(`--model` で指定した1件を返す) |

### 動作の詳細

- **応答本文**は `dummy_server.py` 内の `DUMMY_MESSAGE`(固定文字列)。
  疑似トークンは本文を `--chunk-size` 文字ずつに分割したもの(`tokenize()`)。
- **usage** は擬似値: `prompt_tokens` はリクエスト `messages` の文字数ベース、
  `completion_tokens` は疑似トークン数。
- **ストリーミング**は OpenAI形式の SSE(`chat.completion.chunk`)を送出し、
  最後に `data: [DONE]` を返します。リクエストに
  `"stream_options": {"include_usage": true}` があると、usageを含む最終chunkを送ります。
- **非ストリーミング**もストリーミング完了時のタイミングに合わせて待つため、
  両モードで応答完了までの所要時間が一致します(TTFT + 生成時間)。
  tok/s 送出は絶対時刻ベースのスケジュールで、sleepの丸め誤差が蓄積しません。
- **`model`** はリクエストの `model` フィールドを優先し、無指定時は `--model` の値を使います。
- **バリデーションエラー**は OpenAI形式の `error` オブジェクト付き `400` で返します
  (JSONでないボディ、`messages` が空配列・欠落など)。

## 使い方の例

### curl

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "dummy-llm", "messages": [{"role": "user", "content": "hello"}]}'
```

ストリーミング:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"stream": true, "messages": [{"role": "user", "content": "hello"}]}'
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# 非ストリーミング
resp = client.chat.completions.create(
    model="dummy-llm",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)

# ストリーミング
stream = client.chat.completions.create(
    model="dummy-llm",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## 応答本文をカスタマイズする

`dummy_server.py` の `generate_message()` が応答本文生成の単一ポイントです。
固定メッセージ以外にしたいときは、この関数の中身だけを書き換えます
(リクエストボディdictを受け取り、本文の文字列を返す)。

```python
def generate_message(request: dict[str, Any]) -> str:
    # 例: 最後のユーザーメッセージをそのまま返す
    return request["messages"][-1]["content"]
```

TTFT / tok/s / ストリーミングのシミュレーションは変更不要で、そのまま効きます。
