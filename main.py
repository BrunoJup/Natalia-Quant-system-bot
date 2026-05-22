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
        self.pool = await asyncpg.create_pool(DATABASE_URL)

db = Database()

# ================= MEMORY (FOR MATCH PAIRING) ================= #

last_prediction = {}

# ================= ML ENGINE ================= #

class AutoML:
    def __init__(self):
        self.model = None

    def load(self):
        if os.path.exists("model.pkl"):
            self.model = joblib.load("model.pkl")

    def train(self, df):
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

    def predict(self, x):
        if not self.model:
            return 0.5
        return self.model.predict_proba(x)[0][1]

ml = AutoML()

# ================= VISION AI ================= #

async def vision(image_bytes):
    encoded = base64.b64encode(image_bytes).decode()

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Return JSON: team1,team2,p1,p2,score. If final result, only score."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
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
    return json.loads(raw.replace("```json","").replace("```",""))

# ================= SAVE ================= #

async def save_match(match, pred=None, odds=None, status=None, img_type="prediction"):
    async with db.pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO matches(
                team1,team2,player1,player2,
                predicted_market,predicted_prob,odds,
                score1,score2,result_status,image_type
            )
            VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        """,
        match.get("team1"),
        match.get("team2"),
        match.get("p1"),
        match.get("p2"),
        pred.get("market") if pred else None,
        pred.get("prob") if pred else None,
        odds,
        match.get("score1"),
        match.get("score2"),
        status,
        img_type
        )

# ================= PROCESS PREDICTION ================= #

async def process_prediction(chat_id, match):

    odds = 1.85
    prob = 0.63  # placeholder (replace with ML later)

    market = "Over 2.5" if prob > 0.5 else "Under 2.5"

    last_prediction[chat_id] = {
        "match": match,
        "market": market,
        "prob": prob,
        "odds": odds
    }

    await save_match(match, {"market": market, "prob": prob}, odds, None, "prediction")

    msg = f"""
🧠 PREDICTION GENERATED
━━━━━━━━━━━━━━━━━━
🏟 {match.get("team1")} vs {match.get("team2")}
👤 {match.get("p1")} vs {match.get("p2")}

📊 Market: {market}
📈 Probability: {round(prob*100,2)}%
💸 Odds: {odds}
━━━━━━━━━━━━━━━━━━
"""

    await bot.send_message(chat_id, msg)

# ================= PROCESS RESULT ================= #

async def process_result(chat_id, match):

    score = match.get("score", "0-0")
    s1, s2 = map(int, score.split("-"))

    pred = last_prediction.get(chat_id)

    if not pred:
        await bot.send_message(chat_id, "⚠️ No previous prediction found")
        return

    actual = "Over 2.5" if (s1 + s2) > 2.5 else "Under 2.5"

    status = "WIN" if actual == pred["market"] else "LOSS"

    # save result
    await save_match(match, None, None, status, "result")

    msg = f"""
📊 MATCH RESULT
━━━━━━━━━━━━━━━━━━
🏟 {match.get("team1","?")} vs {match.get("team2","?")}
👤 {match.get("p1","?")} vs {match.get("p2","?")}

📊 Score: {s1}-{s2}
🎯 Actual: {actual}
📡 Prediction: {pred["market"]}

💰 RESULT: {status}
━━━━━━━━━━━━━━━━━━
"""

    await bot.send_message(chat_id, msg)

# ================= MAIN HANDLER ================= #

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, bot)

    if update.message and update.message.photo:
        photo = await update.message.photo[-1].get_file()
        img = await photo.download_as_bytearray()

        match = await vision(img)

        # detect if result or prediction
        if "score" in match and len(match.get("score","")) <= 3:
            await process_result(update.message.chat_id, match)
        else:
            await process_prediction(update.message.chat_id, match)

    return {"ok": True}

# ================= START ================= #

@app.on_event("startup")
async def startup():
    await db.connect()
    ml.load()
    logging.info("🚀 FULL CLOSED LOOP AI LIVE")

@app.get("/")
def home():
    return {"status": "closed loop system running"}
