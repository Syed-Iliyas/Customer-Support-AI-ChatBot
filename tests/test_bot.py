import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app  # noqa: E402
from bot import Chatbot, Database  # noqa: E402
from bot.nlp import extract_entities, preprocess  # noqa: E402


@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


@pytest.fixture()
def bot(db):
    return Chatbot(db=db, backend="tfidf")


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "api.db"), backend="tfidf", admin_token="secret")
    return app.test_client()


# ------------------------------------------------------------------ NLP
def test_preprocess_stems_and_drops_stopwords():
    assert preprocess("Where is my package?") == ["where", "packag"]
    assert "ordernum" in preprocess("track order #10001")


def test_entity_extraction():
    assert extract_entities("my order is #10002 please")["order_id"] == "10002"
    assert extract_entities("My name is sam")["name"] == "Sam"
    assert "name" not in extract_entities("this is not working")
    assert "order_id" not in extract_entities("call 123")  # too short to be an order id


# --------------------------------------------------------- classification
HELD_OUT = [  # none of these sentences appear in intents.json
    ("hello there", "greeting"),
    ("when do you close", "business_hours"),
    ("i would like my money refunded", "refund_policy"),
    ("how can i pay for my purchase", "payment_methods"),
    ("i forgot my login password", "reset_password"),
    ("is there any coupon code", "promo_codes"),
    ("what is the status of order 10003", "order_status"),
    ("can i send an item back", "return_process"),
    ("i need to speak with a real person", "human_agent"),
    ("my headphones are broken, what about warranty", "warranty"),
    ("wher is my pacakge", "order_status"),  # typos
    ("goodbye for now", "goodbye"),
    ("thank you very much", "thanks"),
    ("what are the delivery charges", "shipping_info"),
    ("i entered the wrong delivery address", "change_address"),
]


def test_intent_accuracy_on_unseen_phrases(bot):
    hits = [bot.classify(text)[0][0] == tag for text, tag in HELD_OUT]
    assert sum(hits) / len(hits) >= 0.85, [t for (t, _), h in zip(HELD_OUT, hits) if not h]


PAYMENT_AND_MORE = [  # real-world style messages, typos included
    ("i already made a payment but did not received the conformation", "payment_issue"),
    ("money got deducted from my account but i didnt get any email", "payment_issue"),
    ("where is my money, refund not yet received", "refund_status"),
    ("my order is late", "delayed_delivery"),
    ("you sent me a different product", "wrong_or_missing_item"),
    ("can i get the bill for my purchase", "invoice"),
    ("is this available in stock", "stock_availability"),
    ("i want to edit my order", "modify_order"),
    ("how do i delete my account", "account_update"),
    ("how to use a gift card", "gift_cards"),
]


@pytest.mark.parametrize("text,tag", PAYMENT_AND_MORE)
def test_support_topics(bot, text, tag):
    assert bot.respond("sess-topics", text).intent == tag


def test_unsure_bot_offers_did_you_mean_options(bot):
    r = bot.respond("sess-dym", "my thing is broken somehow")
    assert "Talk to a human" in r.suggestions and len(r.suggestions) >= 2
    # tapping a suggested option must land on an intent (never loop back to the fallback)
    for chip in r.suggestions:
        assert not bot.respond("sess-chip", chip).fallback, chip


def test_every_suggestion_chip_maps_to_an_answer(bot):
    chips = {c for i in bot.intents.values() for c in i.get("suggestions", [])} - {"Yes", "No"}
    chips |= {bot._label(t) for t in bot.intents if bot.intents[t].get("contextual_responses") is None}
    bad = [c for c in sorted(chips) if bot.respond("sess-allchips", c).fallback]
    assert not bad, bad


@pytest.mark.parametrize("text", ["banana pancakes", "xyzzy qwerty", "the moon is made of cheese"])
def test_gibberish_falls_back(bot, text):
    r = bot.respond("sess-fallback", text)
    assert r.fallback and r.intent is None


def test_repeated_fallback_offers_human_handoff(bot):
    bot.respond("sess-handoff", "xyzzy")
    assert "human agent" in bot.respond("sess-handoff", "qwerty plugh").text


# -------------------------------------------------------------- context
def test_followup_uses_previous_topic(bot):
    bot.respond("sess-ctx", "what is your refund policy")
    assert "5-7 business days" in bot.respond("sess-ctx", "how long will it take?").text
    bot.respond("sess-ctx", "tell me about shipping times")
    assert "3-5 business days" in bot.respond("sess-ctx", "how long will it take?").text


def test_followup_survives_small_talk(bot):
    bot.respond("sess-talk", "what is your refund policy")
    bot.respond("sess-talk", "thanks")
    assert "5-7 business days" in bot.respond("sess-talk", "how long?").text


def test_followup_without_context_asks_clarifying_question(bot):
    assert "Which process" in bot.respond("sess-new", "how long will it take").text


def test_sessions_are_isolated(bot):
    bot.respond("sess-a", "what is your refund policy")
    assert "Which process" in bot.respond("sess-b", "how long will it take").text


def test_name_is_remembered(bot):
    bot.respond("sess-name", "my name is Priya")
    assert "Priya" in bot.respond("sess-name", "hello").text


def test_order_slot_filling(bot):
    assert "order number" in bot.respond("sess-slot", "where is my order").text
    assert "BlueDart" in bot.respond("sess-slot", "10001").text


def test_unknown_order_number(bot):
    bot.respond("sess-miss", "track my order")
    assert "couldn't find order #99999" in bot.respond("sess-miss", "99999").text


def test_cancel_requires_confirmation(bot, db):
    bot.respond("sess-cancel", "cancel order 10002")
    assert db.get_order("10002")["status"] == "Processing"  # nothing changed yet
    assert "refund" in bot.respond("sess-cancel", "yes").text.lower()
    assert db.get_order("10002")["status"] == "Cancelled"


def test_cancel_declined_changes_nothing(bot, db):
    bot.respond("sess-no", "cancel order 10002")
    bot.respond("sess-no", "no")
    assert db.get_order("10002")["status"] == "Processing"


def test_shipped_order_cannot_be_cancelled(bot, db):
    assert "can't be cancelled" in bot.respond("sess-ship", "cancel order 10001").text
    assert db.get_order("10001")["status"] == "Shipped"


# -------------------------------------------------------------- logging
def test_every_turn_is_logged(bot, db):
    bot.respond("sess-log", "hello")
    r = bot.respond("sess-log", "xyzzy")
    logs = db.recent_logs(session_id="sess-log")
    assert len(logs) == 2 and logs[0]["id"] == r.log_id
    assert logs[0]["is_fallback"] == 1 and logs[1]["intent"] == "greeting"
    assert logs[1]["latency_ms"] is not None


# ------------------------------------------------------------------ API
def test_api_chat_and_session(client):
    r = client.post("/api/chat", json={"message": "what is your refund policy"})
    body = r.get_json()
    assert r.status_code == 200 and body["intent"] == "refund_policy" and body["log_id"]
    r2 = client.post("/api/chat", json={"message": "how long will it take", "session_id": body["session_id"]})
    assert "5-7" in r2.get_json()["reply"]


def test_api_rejects_empty_message(client):
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/chat", data="not json").status_code == 400


def test_api_feedback(client):
    log_id = client.post("/api/chat", json={"message": "hello"}).get_json()["log_id"]
    assert client.post("/api/feedback", json={"log_id": log_id, "value": 1}).status_code == 200
    assert client.post("/api/feedback", json={"log_id": log_id, "value": 5}).status_code == 400
    assert client.post("/api/feedback", json={"log_id": 9999, "value": 1}).status_code == 404
    stats = client.get("/api/admin/stats", headers={"X-Admin-Token": "secret"}).get_json()
    assert stats["thumbs_up"] == 1 and stats["total_messages"] == 1


def test_admin_requires_token(client):
    assert client.get("/api/admin/logs").status_code == 401
    assert client.get("/api/admin/logs?token=wrong").status_code == 401
    assert client.get("/api/admin/export.csv").status_code == 401
    client.post("/api/chat", json={"message": "hello"})
    rows = client.get("/api/admin/logs", headers={"X-Admin-Token": "secret"}).get_json()
    assert rows[0]["user_message"] == "hello"


def test_csv_export_neutralises_formulas(client):
    client.post("/api/chat", json={"message": "=HYPERLINK(1)"})
    csv_text = client.get("/api/admin/export.csv", headers={"X-Admin-Token": "secret"}).get_data(as_text=True)
    assert "'=HYPERLINK(1)" in csv_text and csv_text.startswith("id,created_at")


def test_pages_render(client):
    assert b"ShopEase Assistant" in client.get("/").data
    assert client.get("/admin").status_code == 200
    assert client.get("/health").get_json()["status"] == "ok"


# ------------------------------------------------- topic tracking (regressions)
def _say(bot, sid, *msgs):
    return [bot.respond(sid, m) for m in msgs][-1]


def test_question_that_names_a_topic_is_not_answered_with_the_previous_topic(bot):
    _say(bot, "t1", "what is your refund policy")
    r = _say(bot, "t1", "how long does delivery take")
    assert r.intent == "shipping_info" and "3-5" in r.text
    r = _say(bot, "t1", "how long does a refund take")
    assert "5-7" in r.text and r.intent in ("refund_status", "refund_policy")


def test_return_duration_question_gets_return_timing_not_generic_steps(bot):
    r = _say(bot, "t2", "how long does a return take")
    assert "Pickup" in r.text


def test_bare_followup_still_uses_current_topic(bot):
    _say(bot, "t3", "do you ship internationally")
    assert "free over" in _say(bot, "t3", "is it free").text
    assert "3-5" in _say(bot, "t3", "and how long").text


def test_followup_names_topic_after_small_talk_and_switches(bot):
    _say(bot, "t4", "what is your refund policy", "thanks")
    assert "5-7" in _say(bot, "t4", "how long will it take").text
    _say(bot, "t4", "do you have any coupons")
    assert _say(bot, "t4", "how do i reset my password").intent == "reset_password"


def test_cancel_confirmation_survives_a_side_question(bot, db):
    _say(bot, "t5", "cancel order 10002")
    r = _say(bot, "t5", "do you accept upi")
    assert r.intent == "payment_methods" and "still want me to cancel" in r.text
    assert db.get_order("10002")["status"] == "Processing"
    assert "Done" in _say(bot, "t5", "yes").text and db.get_order("10002")["status"] == "Cancelled"


def test_ok_inside_a_question_never_confirms_a_cancellation(bot, db):
    _say(bot, "t6", "cancel order 10002")
    _say(bot, "t6", "ok what is your refund policy")
    assert db.get_order("10002")["status"] == "Processing"


def test_bare_order_number_looks_the_order_up(bot):
    _say(bot, "t7", "track my order", "what payment methods do you accept")
    assert "Smart Watch" in _say(bot, "t7", "10004").text


def test_payment_trouble_then_money_back_is_answered_for_that_context(bot):
    _say(bot, "t8", "my payment failed")
    assert "automatically" in _say(bot, "t8", "will i get my money back").text


def test_name_introduction_always_wins(bot):
    assert bot.respond("t9", "my name is Priya").intent == "provide_name"
    assert "Priya" in bot.respond("t9", "hello").text


@pytest.mark.parametrize("text", ["what's the weather today", "tell me a joke", "who won the cricket match", "play some music",
                                  "who is the prime minister", "can you recommend a pizza place"])
def test_off_topic_questions_are_declined_not_guessed(bot, text):
    assert bot.respond("t10", text).fallback


def test_greeting_followed_by_a_request_answers_the_request(bot):
    assert bot.respond("t11", "hey, can you check on my delivery").intent == "order_status"


def test_evaluation_sets_stay_above_the_bar():
    """Runs tests/run_eval.py: tuning, held-out, unseen and off-topic sets plus every multi-turn scenario."""
    import io, contextlib
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import run_eval
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        misses = run_eval.main(verbose=False)
    assert misses <= 12, buf.getvalue()
