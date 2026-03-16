from faster_whisper import WhisperModel
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from notion_client import Client
from datetime import datetime
import tempfile
import os
import time
import json

# env 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("GROMIT_OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
NOTION_TOKEN = os.getenv("GROMIT_NOTION_TOKEN")
notion = Client(auth=NOTION_TOKEN)

app = Flask(__name__)

print("Loading Whisper model...", flush=True)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model loaded.", flush=True)


def summarize_text(text, speaker_list):

    prompt = f"""
다음 회의 내용을 발표자 기준으로 JSON 형식으로 정리하라.
발표자는 순서대로 {speaker_list}이다.


JSON 형식:

{{
  "발표": [
    {{
      "발표자": "",
      "주제": "",
      "메인 포인트": [],
      "질문 및 todo": []
    }}
  ]
}}

규칙:
- 반드시 JSON 형식만 출력하라. JSON 외의 텍스트나 설명을 절대 추가하지 마라.
- markdown 코드블록(```json 등)을 사용하지 마라.
- 발표자마다 주제가 다를 수 있으므로 발표자별로 주제를 따로 작성하라.
- 메인 포인트는 핵심 내용을 bullet 형태로 정리하라.
- 질문 및 todo는 질문이나 후속 작업을 정리하라.
- 질문자는 발표자가 아닌 회의 참여자일 수 있다. 문맥을 기반으로 발표자의 발언과 질문자의 발언을 구분하여 정리하라.
- 질문자가 없을 경우 질문 및 todo는 반드시 ["질문 없음"]으로 작성하라.
- STT 결과이므로 문장을 자연스럽게 보정하라.
- 회의와 관계없는 사설, 감탄사, 잡담은 모두 제거하라.
- 동일한 의미의 문장이 반복되면 하나로 정리하라.

회의 내용:
{text}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text
    
def append_blocks(blocks, notion_page_id):

    chunk_size = 100

    for i in range(0, len(blocks), chunk_size):

        notion.blocks.children.append(
            block_id=notion_page_id,
            children=blocks[i:i+chunk_size]
        )

def send_to_notion(summary_json, notion_page_id):

    today = datetime.now().strftime("%Y-%m-%d")

    blocks = []

    # 회의 제목
    blocks.append({
        "object": "block",
        "type": "heading_1",
        "heading_1": {
            "rich_text": [
                {"type": "text", "text": {"content": f"{today} 회의 요약"}}
            ]
        }
    })

    for pres in summary_json["발표"]:

        speaker = pres["발표자"]
        topic = pres["주제"]
        points = pres["메인 포인트"]
        todos = pres["질문 및 todo"]

        # 발표자 + 주제
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [
                    {"type": "text", "text": {"content": f"{speaker} - {topic}"}}
                ]
            }
        })

        # 메인 포인트
        for p in points:
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {"type": "text", "text": {"content": p}}
                    ]
                }
            })

        # TODO 제목
        blocks.append({
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [
                    {"type": "text", "text": {"content": "TODO / 질문"}}
                ]
            }
        })

        # TODO 체크박스
        for t in todos:
            blocks.append({
                "object": "block",
                "type": "to_do",
                "to_do": {
                    "rich_text": [
                        {"type": "text", "text": {"content": t}}
                    ],
                    "checked": False
                }
            })

        # 사람 구분선
        blocks.append({
            "object": "block",
            "type": "divider",
            "divider": {}
        })

    append_blocks(blocks, notion_page_id)


@app.route("/")
def root():
    return "STT + Summary Server Running"


@app.route("/v2/transcribe", methods=["POST"])
def transcribe():

    start_time = time.time()

    if "audio" not in request.files:
        return jsonify({"error": "audio file missing"}), 400

    audio = request.files["audio"]
    speakers = request.form.get("speakers")
    notion_page_id = request.form.get("notion_page_id")
    speaker_list = speakers.split(",") if speakers else []

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            audio.save(f.name)
            filepath = f.name

        segments, info = model.transcribe(filepath, language="ko")

        text = " ".join([s.text for s in segments])

        os.unlink(filepath)

        # GPT 요약
        summary = summarize_text(text,speaker_list)
        
        try:
            summary_json = json.loads(summary)
        except json.JSONDecodeError:
            return jsonify({"error": "GPT JSON parsing failed", "raw": summary}), 500

        send_to_notion(summary_json, notion_page_id)

        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "요약": summary_json,
            "응답 시간": elapsed
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
