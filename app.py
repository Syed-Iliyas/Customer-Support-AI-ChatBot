"""Flask API + web UI for the ShopEase support chatbot.

Run:   python app.py            (dev)      gunicorn -w 2 "app:create_app()"   (production)
"""
from __future__ import annotations

import csv
import hmac
import io
import logging
import os
import re
import secrets
import uuid

from flask import Flask, Response, jsonify, render_template, request

from bot import Chatbot, Database

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def create_app(db_path: str | None = None, backend: str | None = None, admin_token: str | None = None) -> Flask:
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db = Database(db_path)
    bot = Chatbot(db=db, backend=backend)
    token = admin_token or os.environ.get("ADMIN_TOKEN") or secrets.token_urlsafe(9)
    if not (admin_token or os.environ.get("ADMIN_TOKEN")):
        app.logger.warning("ADMIN_TOKEN not set. Generated one for this run: %s", token)
    app.config.update(BOT=bot, DB=db, ADMIN_TOKEN=token)

    def is_admin() -> bool:
        supplied = request.headers.get("X-Admin-Token") or request.args.get("token", "")
        return hmac.compare_digest(supplied.encode(), token.encode())

    def admin_only(fn):
        def wrapper(*a, **kw):
            if not is_admin():
                return jsonify(error="unauthorized"), 401
            return fn(*a, **kw)
        wrapper.__name__ = fn.__name__
        return wrapper

    # ------------------------------------------------------------ public
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok", backend=bot.backend, intents=len(bot.intents))

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify(error="message is required"), 400
        session_id = str(data.get("session_id") or "")
        if not SESSION_RE.match(session_id):
            session_id = uuid.uuid4().hex
        reply = bot.respond(session_id, message)
        return jsonify(
            session_id=session_id,
            reply=reply.text,
            intent=reply.intent,
            confidence=round(reply.confidence, 3),
            suggestions=reply.suggestions,
            fallback=reply.fallback,
            log_id=reply.log_id,
        )

    @app.post("/api/feedback")
    def feedback():
        data = request.get_json(silent=True) or {}
        try:
            log_id, value = int(data["log_id"]), int(data["value"])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="log_id and value (1 or -1) are required"), 400
        if value not in (1, -1):
            return jsonify(error="value must be 1 or -1"), 400
        return (jsonify(ok=True) if db.set_feedback(log_id, value) else (jsonify(error="unknown log_id"), 404))

    # ------------------------------------------------------------- admin
    @app.get("/admin")
    def admin_page():
        return render_template("admin.html")

    @app.get("/api/admin/stats")
    @admin_only
    def admin_stats():
        return jsonify(db.stats())

    @app.get("/api/admin/logs")
    @admin_only
    def admin_logs():
        limit = min(max(request.args.get("limit", 100, type=int), 1), 500)
        offset = max(request.args.get("offset", 0, type=int), 0)
        return jsonify(db.recent_logs(limit, offset, request.args.get("session_id") or None))

    @app.get("/api/admin/export.csv")
    @admin_only
    def admin_export():
        rows = db.recent_logs(limit=100000)
        buf = io.StringIO()
        cols = ["id", "created_at", "session_id", "user_message", "bot_response", "intent", "confidence", "is_fallback", "latency_ms", "feedback"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # prefix formula characters so spreadsheets don't execute user-typed text
            for k in ("user_message", "bot_response"):
                if r[k] and r[k][0] in "=+-@":
                    r[k] = "'" + r[k]
            w.writerow(r)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=chat_logs.csv"})

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
