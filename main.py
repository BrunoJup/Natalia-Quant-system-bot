import os
import base64
import requests
from flask import Flask, request, jsonify, abort
from concurrent.futures import ThreadPoolExecutor

# ================= CONFIG =================
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
API_KEY = os.environ["OPENROUTER_API_KEY"]

TEXT_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
VISION_MODELS = [
    os.getenv("VISION_MODEL_1", "google/gemini-2.0-flash-001"),
    os.getenv("VISION_MODEL_2", "anthropic/claude-3.5-haiku"),
    os.getenv("VISION_MODEL_3", "meta-llama/llama-3.2-11b-vision-instruct"),
]
VISION_MODELS = [m for m in VISION_MODELS if m]

AGG_MODEL = os.getenv("AGGREGATOR_MODEL", TEXT_MODEL)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

TG = f"https://api.telegram.org/bot{TOKEN}"
OR = "https://openrouter.ai/api/v1/chat/completions"

app = Flask(__name__)

# ================= ENGINE =================
CONF = {1:52,2:58,3:64,4:71,5:78,6:84,7:89}

def avg(x): return sum(x)/len(x)

def evaluate(f):
    a_for, a_against, b_for, b_against = f

    total = avg(a_for)+avg(a_against)+avg(b_for)+avg(b_against)
    ht = (avg(a_for)+avg(b_for))/2

    ft = 3 if total>=7.5 else 2 if total>=6.5 else 1 if total>=5.8 else 0
    ht_s = 2 if ht>=3.2 else 1 if ht>=2.6 else 0
    d = 1 if avg(a_against)>=2.5 and avg(b_against)>=2.5 else 0

    score = ft+ht_s+d

    verdict = "🔥 VERY STRONG" if score>=6 else "✅ STRONG" if score>=4 else "🟡 LEAN" if score>=3 else "❌ NO BET"

    market = "—" if verdict=="❌ NO BET" else (
        "Over 7.5" if score>=6 and total>=8 else
        "Over 6.5" if score>=4 and total>=7 else
        "Over 5.5" if score>=2 and total>=5.8 else
        "HT Over 2.5" if ht>=2.6 else "No Market"
    )

    return verdict, market, CONF.get(score,45)

# ================= TELEGRAM =================
def send(chat_id, text):
    url = f"{TG}/sendMessage"
    for i in range(0, len(text), 4096):
        chunk = text[i:i+4096]
        requests.post(url, json={
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML"
        }, timeout=10)

def file_url(file_id):
    r = requests.get(f"{TG}/getFile", params={"file_id": file_id}, timeout=10).json()
    return f"https://api.telegram.org/file/bot{TOKEN}/{r['result']['file_path']}"

def img_b64(url):
    r = requests.get(url, timeout=20)
    return base64.b64encode(r.content).decode()

# ================= AI =================
def call(model, messages):
    r = requests.post(
        OR,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 900
        },
        timeout=40
    )
    return r.json()["choices"][0]["message"]["content"]

# ================= VISION =================
def vision_pipeline(image):
    def run(model):
        return call(model, [{
            "role": "user",
            "content": [
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image}"}},
                {"type":"text","text":"Extract football fixtures and stats clearly"}
            ]
        }])

    with ThreadPoolExecutor(max_workers=len(VISION_MODELS)) as ex:
        results = list(ex.map(run, VISION_MODELS))

    combined = "\n\n".join(results)

    return call(AGG_MODEL, [
        {"role":"system","content":"You are a football betting analyst. Be structured and accurate."},
        {"role":"user","content":combined}
    ])

# ================= ROUTES =================
@app.route("/")
def home():
    return jsonify({"status":"ok"})

@app.route(f"/webhook/{TOKEN}", methods=["POST"])
def webhook():
    # Optional security
    if WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token","")
        if token != WEBHOOK_SECRET:
            abort(403)

    data = request.get_json(silent=True)
    if not data:
        return "bad request", 400

    msg = data.get("message", {})
    chat = msg.get("chat", {}).get("id")

    if not chat:
        return "ok"

    # PHOTO
    if "photo" in msg:
        try:
            url = file_url(msg["photo"][-1]["file_id"])
            result = vision_pipeline(img_b64(url))
            send(chat, result)
        except Exception:
            send(chat, "❌ Vision processing failed")
        return "ok"

    text = msg.get("text","")

    # ANALYZE
    if text.startswith("/analyze"):
        try:
            _, raw = text.split(" ",1)
            parts = raw.split("|")
            f = [list(map(float,p.split(","))) for p in parts[1:]]
            verdict, market, conf = evaluate(f)
            send(chat, f"{verdict} | {market} | {conf}%")
        except Exception:
            send(chat, "Format: /analyze Match | 1,2 | 2,3 | 3,4 | 1,2")

    # AI
    elif text.startswith("/ai"):
        q = text.replace("/ai","",1).strip()
        if not q:
            send(chat, "Ask something after /ai")
        else:
            try:
                send(chat, call(TEXT_MODEL,[{"role":"user","content":q}]))
            except Exception:
                send(chat, "❌ AI request failed")

    else:
        send(chat, "Send image or use /analyze or /ai")

    return "ok"

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)))
