import os
import json
import base64
import logging
import numpy as np
import aiohttp
import asyncpg
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from fastapi import FastAPI, Request
from telegram import Bot, Update

# ================= CONFIG ================= #

logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

# ================= DATABASE ================= #

class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(DATABASE_URL)
            print("✅ Database connected")
        except Exception as e:
            print("❌ DB ERROR:", e)

db = Database()

# ================= MEMORY ================= #

last_prediction = {}

# ================= ML ENGINE ================= #

class AutoML:
    def __init__(self):
        self.model = None

    def load(self):
        try:
            if os.path.exists("model.pkl"):
                self.model = joblib.load("model.pkl")
                print("✅ Model loaded")
            else:
                self.model = GradientBoostingClassifier()
                print("⚠️ New model initialized")
        except Exception as e:
            print("❌ Model load error:", e)

    def train(self, df):
        try:
            if len(df) < 10:
                return

            df = df.dropna()

            df["target"] = ((df["score1"] + df["score2"]) > 2.5).astype(int)

            X = df[["score1", "score2"]]
            y = df["target"]

            model = GradientBoostingClassifier()
            model.fit(X, y)

            self.model = model
            joblib.dump(model, "model.pkl")

            print("🧠 Model trained")
        except Exception as e:
            print("❌ Training error:", e)

    def predict(self, x):
        try:
            if not self.model:
                return 0.5
            return self.model.predict_proba(x)[0][1]
        except:
            return 0.5

ml = AutoML()

# ================= VISION AI ================= #

async def vision(image_bytes):
    try:
        encoded = base64.b64encode(image_bytes).decode()

        payload = {
            "model": "google/gemini-2.0-flash-001",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract JSON: team1,team2,p1,p2,score"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                    }
                ]
            }]
        }

        async with aiohttp.ClientSession() as s:
            async with s.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json=payload
            ) as r:
                res = await r.json()

        raw = res["choices"][0]["message"]["content"]
        return json.loads(raw.replace("```json", "").replace("```", ""))

    except Exception as e:
        print("❌ Vision error:", e)
        return {}

# ================= DATABASE SAVE ================= #

async def save_prediction(match, market, prob, odds):
    try:
        async with db.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO matches(team1,team2,player1,player2,
                predicted_market,predicted_prob,odds)
                VALUES($1,$2,$3,$4,$5,$6,$7)
            """,
            match.get("team1"),
            match.get("team2"),
            match.get("p1"),
            match.get("p2"),
            market,
            prob,
            odds)
    except Exception as e:
        print("❌ Save prediction error:", e)

async def save_result(match, score1, score2, status):
    try:
        async with db.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO matches(team1,team2,player1,player2,
                score1,score2,result_status)
                VALUES($1,$2,$3,$4,$5,$6,$7)
            """,
            match.get("team1"),
            match.get("team2"),
            match.get("p1"),
            match.get("p2"),
            score1,
            score2,
            status)
    except Exception as e:
        print("❌ Save result error:", e)

# ================= PROCESS ================= #

async def process_prediction(chat_id, match):
    odds = 1.85
    prob = ml.predict(np.array([[1, 1]]))  # safe fallback

    market = "Over 2.5" if prob > 0.5 else "Under 2.5"

    last_prediction[chat_id] = {
        "market": market,
        "prob": prob
    }

    await save_prediction(match, market, prob, odds)

    msg = f"""
🧠 PREDICTION
━━━━━━━━━━━━
🏟 {match.get("team1","?")} vs {match.get("team2","?")}
👤 {match.get("p1","?")} vs {match.get("p2","?")}

📊 {market}
📈 {round(prob*100,2)}%
💸 {odds}
━━━━━━━━━━━━
"""

    await bot.send_message(chat_id, msg)

async def process_result(chat_id, match):
    try:
        score = match.get("score", "0-0")
        s1, s2 = map(int, score.split("-"))

        pred = last_prediction.get(chat_id)

        if not pred:
            await bot.send_message(chat_id, "⚠️ No previous prediction")
            return

        actual = "Over 2.5" if (s1 + s2) > 2.5 else "Under 2.5"
        status = "WIN" if actual == pred["market"] else "LOSS"

        await save_result(match, s1, s2, status)

        msg = f"""
📊 RESULT
━━━━━━━━━━━━
📊 {s1}-{s2}
🎯 {actual}
📡 {pred["market"]}
💰 {status}
━━━━━━━━━━━━
"""

        await bot.send_message(chat_id, msg)

    except Exception as e:
        print("❌ Result error:", e)

# ================= WEBHOOK ================= #

@app.post("/webhook")
async def webhook(req: Request):
    try:
        data = await req.json()
        update = Update.de_json(data, bot)

        if update.message and update.message.photo:
            photo = await update.message.photo[-1].get_file()
            img = await photo.download_as_bytearray()

            await bot.send_message(update.message.chat_id, "🔎 Processing...")

            match = await vision(img)

            if "score" in match and "-" in match.get("score", ""):
                await process_result(update.message.chat_id, match)
            else:
                await process_prediction(update.message.chat_id, match)

    except Exception as e:
        print("❌ Webhook error:", e)

    return {"ok": True}

# ================= STARTUP ================= #

@app.on_event("startup")
async def startup():
    await db.connect()
    ml.load()
    print("🚀 App started successfully")

# ================= HEALTH CHECK ================= #

@app.get("/")
def home():
    return {"status": "running"}
