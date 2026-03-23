#!/usr/bin/env python3
"""
AI Conversational Call Server v5
- ElevenLabs TTS (голос Артёма: blxHPCXhpXOsc7mCKk0P)
- Аудио отдаётся с локального Flask сервера через туннель
- GPT-4o диалог
"""

import os
import uuid
import threading
import requests
from pathlib import Path
from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse, Gather
from openai import OpenAI

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ARTEM_VOICE_ID = "blxHPCXhpXOsc7mCKk0P"

client = OpenAI(api_key=OPENAI_API_KEY)
CLIENT_NAME = os.environ.get("CLIENT_NAME", "Клиент")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "")

AUDIO_DIR = Path("/tmp/ai-call-audio")
AUDIO_DIR.mkdir(exist_ok=True)

prefetch_cache = {}
prefetch_lock = threading.Lock()

SYSTEM_PROMPT = f"""Ты — Доктор Амир Рашидов. Психолог-практик, 17 лет опыта. Лос-Анджелес, США.

ТВОЯ ЦЕЛЬ: Договориться на бесплатную Zoom-консультацию 30 минут. Любой ценой.

ТЕХНИКА "ЖЁСТКОГО ЗАКРЫТИЯ":
- Не спрашивай разрешения — предлагай и веди к "да"
- Создавай FOMO: "у меня 2 слота, один уже занят"
- Социальное доказательство: "мои клиентки из Ташкента за 3 недели меняют жизнь"
- Разрушай возражения немедленно: "я понимаю, но именно поэтому вам и нужно..."
- Альтернативное закрытие: "вам среда в 19:00 или четверг в 18:00?"
- Никогда не принимай "нет" как ответ с первого раза

ВАЖНО:
- Говори по-русски, УВЕРЕННО и НАПОРИСТО
- КОРОТКО — максимум 2 предложения за раз
- Не жди — сразу ведёшь к закрытию сделки
- Если согласился — скажи "Отлично! Пришлю ссылку на Zoom. До встречи!" и завершай

ЗАПРЕЩЕНО: Длинные монологи. Нерешительность. "Может быть". "Если хотите"."""

conversation_history = {}


def get_ai_text(call_sid: str, user_text: str) -> str:
    if call_sid not in conversation_history:
        conversation_history[call_sid] = []
    conversation_history[call_sid].append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[call_sid]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=80,
        temperature=0.8
    )
    ai_text = response.choices[0].message.content.strip()
    conversation_history[call_sid].append({"role": "assistant", "content": ai_text})
    print(f"[AI]: {ai_text}", flush=True)
    return ai_text


def generate_tts_elevenlabs(text: str) -> str:
    """ElevenLabs TTS — голос Артёма"""
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = AUDIO_DIR / filename

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ARTEM_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code == 200:
        filepath.write_bytes(resp.content)
        size = filepath.stat().st_size
        print(f"[EL TTS]: {filename} ({size}b)", flush=True)
    else:
        print(f"[EL TTS ERROR]: {resp.status_code} {resp.text[:200]}", flush=True)
        # Fallback на OpenAI TTS
        with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="onyx", input=text, speed=1.1
        ) as r:
            r.stream_to_file(str(filepath))
        print(f"[OAI TTS fallback]: {filename}", flush=True)

    return filename


def prefetch_next(call_sid: str, hint: str):
    def _run():
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
                       conversation_history.get(call_sid, []) + \
                       [{"role": "user", "content": hint}]
            resp = client.chat.completions.create(
                model="gpt-4o", messages=messages, max_tokens=80, temperature=0.8
            )
            ai_text = resp.choices[0].message.content.strip()
            filename = generate_tts_elevenlabs(ai_text)
            with prefetch_lock:
                prefetch_cache[call_sid] = (filename, ai_text)
            print(f"[PREFETCH OK]: {ai_text}", flush=True)
        except Exception as e:
            print(f"[PREFETCH ERR]: {e}", flush=True)
    threading.Thread(target=_run, daemon=True).start()


@app.route("/audio/<filename>")
def serve_audio(filename):
    filepath = AUDIO_DIR / filename
    if not filepath.exists():
        return "Not found", 404
    return send_file(str(filepath), mimetype="audio/mpeg")


@app.route("/call/start", methods=["POST"])
def call_start():
    call_sid = request.form.get("CallSid", "unknown")
    print(f"\n[СТАРТ] {call_sid}", flush=True)

    ai_text = get_ai_text(call_sid, "[Звонок соединился. Поздоровайся, представься кратко и сразу скажи зачем звонишь.]")
    filename = generate_tts_elevenlabs(ai_text)
    audio_url = f"{PUBLIC_URL}/audio/{filename}"

    prefetch_next(call_sid, "[Пауза после приветствия. Подтолкни к разговору одним предложением.]")

    resp = VoiceResponse()
    resp.play(audio_url)
    gather = Gather(input="speech", action="/call/respond", method="POST",
                    language="ru-RU", speechTimeout="auto", timeout=12)
    resp.append(gather)
    resp.redirect("/call/silence")
    return Response(str(resp), mimetype="text/xml")


@app.route("/call/respond", methods=["POST"])
def call_respond():
    call_sid = request.form.get("CallSid", "unknown")
    speech = request.form.get("SpeechResult", "").strip()
    print(f"[КЛИЕНТ]: '{speech}'", flush=True)

    if not speech:
        return do_silence(call_sid)

    ai_text = get_ai_text(call_sid, speech)
    filename = generate_tts_elevenlabs(ai_text)
    audio_url = f"{PUBLIC_URL}/audio/{filename}"

    prefetch_next(call_sid, "[Клиент сейчас ответит на предложение. Готовь контраргумент на возражение или закрытие.]")

    resp = VoiceResponse()
    resp.play(audio_url)

    end_phrases = ["до свидания", "до встречи", "договорились", "пришлю ссылку"]
    if any(p in ai_text.lower() for p in end_phrases):
        resp.hangup()
        print("[ЗАВЕРШЕНО]", flush=True)
    else:
        gather = Gather(input="speech", action="/call/respond", method="POST",
                        language="ru-RU", speechTimeout="auto", timeout=12)
        resp.append(gather)
        resp.redirect("/call/silence")
    return Response(str(resp), mimetype="text/xml")


@app.route("/call/silence", methods=["GET", "POST"])
def call_silence():
    call_sid = request.form.get("CallSid", "unknown")
    return do_silence(call_sid)


def do_silence(call_sid: str):
    with prefetch_lock:
        cached = prefetch_cache.pop(call_sid, None)

    if cached:
        filename, ai_text = cached
        conversation_history.setdefault(call_sid, [])
        conversation_history[call_sid].append({"role": "assistant", "content": ai_text})
        print(f"[AI prefetch]: {ai_text}", flush=True)
    else:
        ai_text = get_ai_text(call_sid, "[Клиент молчит. Переспроси одним коротким предложением.]")
        filename = generate_tts_elevenlabs(ai_text)

    audio_url = f"{PUBLIC_URL}/audio/{filename}"
    resp = VoiceResponse()
    resp.play(audio_url)
    gather = Gather(input="speech", action="/call/respond", method="POST",
                    language="ru-RU", speechTimeout="auto", timeout=12)
    resp.append(gather)
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"🚀 AI Call Server v5 (ElevenLabs Артём) | port={port} | client={CLIENT_NAME}", flush=True)
    print(f"🌐 {PUBLIC_URL}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
