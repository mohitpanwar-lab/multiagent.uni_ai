#!/usr/bin/env python
# coding: utf-8

"""
app_flask.py — Flask web UI for Uni Assistant, with MongoDB-backed,
multi-conversation chat history (a sidebar list of past conversations,
like ChatGPT — not just one continuous chat).

Setup:
    pip install -r requirements.txt
    Put these in a .env file next to this script:
        GROQ_API_KEY=gsk_...
        MONGODB_URI=mongodb://localhost:27017   (or your Atlas connection string)

Run:
    python app_flask.py
Then open http://127.0.0.1:5000 in your browser.

Data model (MongoDB, database "uni_assistant", collection "conversations"):
    {
        conversation_id: "<uuid>",
        user_id: "<uuid>",       # identifies this browser, permanent cookie
        title: "<first message, truncated>",
        history: [...],          # full message list incl. system/tool messages
        updated_at: <ISO timestamp>
    }
"""

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, session
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError
from dotenv import load_dotenv

from engine import run_turn, SYSTEM_PROMPT

load_dotenv()

app = Flask(__name__)
app.secret_key = "uni-assistant-dev-secret-change-this-in-production"

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    mongo_client.admin.command("ping")
    db = mongo_client["uni_assistant"]
    conversations = db["conversations"]
    print(f"[app_flask.py] Connected to MongoDB at {MONGODB_URI}")
except ServerSelectionTimeoutError as e:
    print(
        f"[app_flask.py] WARNING: could not connect to MongoDB at {MONGODB_URI}.\n"
        f"  -> Make sure MongoDB is installed and running, or set MONGODB_URI in .env "
        f"to a valid Atlas connection string.\n  -> Error: {e}"
    )
    conversations = None


def _get_user_id():
    """One persistent id per browser — groups all of that browser's conversations."""
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())
        session.permanent = True
    return session["user_id"]


def _visible_messages(history):
    visible = []
    for msg in history:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            visible.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant" and msg.get("content"):
            visible.append({"role": "assistant", "content": msg["content"]})
    return visible


def _make_title(first_message):
    title = first_message.strip().replace("\n", " ")
    return title[:48] + ("…" if len(title) > 48 else "")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    """List this browser's past conversations, most recent first."""
    user_id = _get_user_id()
    if conversations is None:
        return jsonify({"conversations": [], "mongo_connected": False})

    docs = conversations.find(
        {"user_id": user_id}, {"conversation_id": 1, "title": 1, "updated_at": 1}
    ).sort("updated_at", DESCENDING)

    result = [
        {
            "conversation_id": d["conversation_id"],
            "title": d.get("title") or "New conversation",
            "updated_at": d.get("updated_at"),
        }
        for d in docs
    ]
    return jsonify({"conversations": result, "mongo_connected": True})


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get messages for one conversation by id (falls back to a fresh one)."""
    conversation_id = request.args.get("conversation_id")
    if not conversation_id or conversations is None:
        return jsonify({"messages": [], "conversation_id": None, "mongo_connected": conversations is not None})

    user_id = _get_user_id()
    doc = conversations.find_one({"conversation_id": conversation_id, "user_id": user_id})
    if not doc:
        return jsonify({"messages": [], "conversation_id": None, "mongo_connected": True})

    return jsonify(
        {
            "messages": _visible_messages(doc["history"]),
            "conversation_id": conversation_id,
            "mongo_connected": True,
        }
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    user_id = _get_user_id()

    is_new = False
    if conversation_id and conversations is not None:
        doc = conversations.find_one({"conversation_id": conversation_id, "user_id": user_id})
        history = doc["history"] if doc else [{"role": "system", "content": SYSTEM_PROMPT}]
    else:
        conversation_id = str(uuid.uuid4())
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        is_new = True

    try:
        answer = run_turn(history, user_message)
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {e}"}), 500

    if conversations is not None:
        update = {"history": history, "updated_at": datetime.now(timezone.utc).isoformat()}
        if is_new:
            update["title"] = _make_title(user_message)
            update["user_id"] = user_id
            update["conversation_id"] = conversation_id
        conversations.update_one(
            {"conversation_id": conversation_id, "user_id": user_id},
            {"$set": update},
            upsert=True,
        )

    return jsonify({"answer": answer, "conversation_id": conversation_id})


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    user_id = _get_user_id()
    if conversations is not None:
        conversations.delete_one({"conversation_id": conversation_id, "user_id": user_id})
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
