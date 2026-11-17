# AI-Powered Customer Support Chatbot

An NLP chatbot for customer support / FAQs, demonstrated on a fictional store ("ShopEase").
It understands free-text questions, keeps conversational context, can look up and cancel orders,
and logs every interaction to SQLite for review.

**Stack:** Python · NLTK · scikit-learn · Transformers (optional) · Flask · SQLite

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                           # http://127.0.0.1:5000
```

- Chat UI: `http://127.0.0.1:5000/`
- Log dashboard: `http://127.0.0.1:5000/admin` (token is printed in the console, or set `ADMIN_TOKEN`)
- Tests: `python -m pytest -q`

Try this conversation: `hi` → `my name is Sam` → `what is your refund policy` → `how long will it take?` →
`is it free?` → `where is my order` → `10001` → `cancel order 10002` → `yes`.
Demo orders are `10001`–`10005`.

## How it works

```
user message ──► NLP pipeline ──► intent classifier ──► dialogue manager ──► response
                 (NLTK)           (TF-IDF [+ BERT])     (context in SQLite)      │
                                                                                  ▼
                                                                  chat_logs table + feedback
```

| Stage | What it does | File |
|---|---|---|
| Preprocessing | Contraction expansion, NLTK tokenizer, stopword removal, Porter stemming, order-number masking, edit-distance typo correction | `bot/nlp.py`, `bot/engine.py` |
| Intent matching | Synonym/phrase normalisation, then TF-IDF (uni+bigrams) cosine similarity against 527 example phrases for 31 intents. Optional blend with sentence embeddings from a Transformers model | `bot/engine.py` |
| Context | Per-session state in SQLite: last topic, remembered name/order id, pending question. Follow-ups like *"how long will it take?"* are answered according to the previous topic | `bot/engine.py`, `bot/db.py` |
| Actions | Slot filling ("which order?") against an `orders` table; cancellation asks for explicit confirmation before changing anything | `bot/engine.py` |
| Fallback | Below the confidence threshold it offers "Did you mean...?" buttons for its closest topics; after two misses in a row it offers a human hand-off | `bot/engine.py` |
| Logging | Every turn: session, message, reply, intent, confidence, latency, fallback flag, 👍/👎 feedback | `bot/db.py` |

## HTTP API

| Method & path | Purpose |
|---|---|
| `POST /api/chat` `{message, session_id?}` | Returns `reply, intent, confidence, suggestions, fallback, log_id, session_id` |
| `POST /api/feedback` `{log_id, value: 1 \| -1}` | Thumbs up/down on a reply |
| `GET /health` | Liveness check |
| `GET /api/admin/stats`, `/api/admin/logs`, `/api/admin/export.csv` | Dashboard data (header `X-Admin-Token`) |

## Customising

- **Your own FAQ/intents:** edit `data/intents.json`. Add a `tag`, 6-10 example `patterns` and `responses`; no retraining step, the index is rebuilt at startup.
- **Context-dependent answers:** add `contextual_responses` keyed by the previous intent (see `followup_duration`).
- **Sensitivity:** `CHATBOT_THRESHOLD` (default `0.32` for TF-IDF). Raise to ask for rephrasing more often.
- **Unanswered questions:** the dashboard lists fallbacks. Add each frequent one as example phrases (existing intent) or a new intent. This is the main way to make the bot smarter, since matching is based on your examples. More varied phrasings = better recall.
- **Real order data:** replace `Database.get_order` / `set_order_status` with calls to your order system.

## How the bot decides what you mean

1. **Normalise:** lower-case, expand contractions, map synonyms/phrases (`"billed"` -> `"charged"`, `"sign in"` -> `"login"`), fix typos.
2. **Score every topic** by similarity to its example phrases, plus a bonus when the message contains one of the topic's `keywords`. A message that is mostly words the bot doesn't know scores low, so off-topic questions get "I'm not sure" instead of a wrong answer.
3. **Resolve follow-ups from the message first.** "how long does *shipping* take?" names its topic and is answered about shipping. Only a bare "how long will it take?" uses the previous topic. Small talk ("thanks") never changes the topic.
4. **Answer the question type.** `cues` ("how long", "is it free", "tell me more") pick the matching `contextual_responses` entry for the named topic.

Fields you can edit per topic in `data/intents.json`: `patterns` (example phrases), `keywords` (words that identify the topic), `weight`
(raise above 1 when a request word like *cancel* should beat a topic word like *wrong*), `responses`, `contextual_responses` (answer by previous topic),
`context_variants` (answer differently when the previous topic was X), `suggestions` (buttons). Top-level `synonyms`, `phrases` and `neutral_words` apply to all topics.

## Measuring accuracy

```bash
python tests/run_eval.py     # prints every question the bot gets wrong, per evaluation set
```

On a set of 77 questions written after the first round of fixes and never used for tuning, the first score was 82% (63/77), with 9 of 14
off-topic questions wrongly answered. The fixes above brought every set to 97-100%, but those sets were then used for tuning, so real accuracy
on customers' own wording will be lower (roughly 80-90%). The reliable way to improve it is the admin page's *Unanswered questions* list: add
those sentences to the right topic's `patterns`, then re-run `tests/run_eval.py`. For paraphrases that share no words with your examples,
use the optional Transformers backend.

## Transformers backend (optional, semantic matching)

```bash
pip install -r requirements-transformers.txt
CHATBOT_BACKEND=transformers python app.py
```

Embeds the example phrases and each user message with `sentence-transformers/all-MiniLM-L6-v2`
(override with `CHATBOT_MODEL`) and blends that similarity with TF-IDF, which helps with paraphrases that share
no words with your examples. The model is downloaded from Hugging Face on first start; if torch/transformers or the
model are unavailable the app logs a warning and continues with TF-IDF only. The default threshold becomes `0.40`
in this mode; tune it on your own data.

## Deployment

```bash
ADMIN_TOKEN=choose-a-long-secret gunicorn -w 2 -b 0.0.0.0:5000 "app:create_app()"
# or
docker build -t chatbot . && docker run -p 5000:5000 -e ADMIN_TOKEN=... -v chatdata:/data chatbot
```

Conversation state and logs live in SQLite (WAL mode), so several workers share context. Put the app behind HTTPS
(nginx, Caddy or your platform's proxy) and persist the `/data` volume.

## Before using with real customers

- The demo order lookup/cancel has **no customer authentication**: anyone who knows an order number can query it.
  Require login or an order-number + email/phone check first.
- Chat logs contain whatever users type (possibly personal data). Decide on a retention period and who may open `/admin`.
- No rate limiting is included; add one (e.g. Flask-Limiter or the reverse proxy) for public deployments.

## Project layout

```
app.py              Flask app factory, API and admin routes
bot/nlp.py          tokenising, stemming, entity extraction
bot/engine.py       classifier, dialogue manager, actions, optional Transformers embedder
bot/db.py           SQLite: sessions, chat_logs, demo orders
data/intents.json   intents, example phrases, responses
templates/, static/ chat UI and admin dashboard
tests/test_bot.py   53 tests: NLP, topic tracking, off-topic handling, accuracy sets, context, actions, logging, API
```
