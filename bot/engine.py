"""Intent classification, dialogue management and response generation."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field

import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

from .db import Database
from nltk.tokenize import wordpunct_tokenize

from .nlp import STOPWORDS, extract_entities, normalize, stem, tokenize

log = logging.getLogger("chatbot")

DEFAULT_INTENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "intents.json")

# Small-talk intents never replace the conversation topic, so
# "refund policy" -> "thanks" -> "how long does it take?" still means refunds.
NON_TOPIC = {"greeting", "goodbye", "thanks", "bot_identity", "provide_name"}
KEYWORD_BONUS = 0.45
SMALL_TALK = {"greeting", "goodbye", "thanks"}
YES = {"yes", "y", "yeah", "yep", "yup", "sure", "confirm", "confirmed", "ok", "okay", "please", "proceed"}
NO = {"no", "n", "nope", "nah", "dont", "stop", "wait"}


class TransformerEmbedder:
    """Sentence embeddings from a Hugging Face model (mean pooling)."""

    def __init__(self, model_name: str):
        import torch  # imported lazily: only needed for the transformers backend
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=1).numpy()


@dataclass
class Reply:
    text: str
    intent: str | None = None
    confidence: float = 0.0
    suggestions: list[str] = field(default_factory=list)
    fallback: bool = False
    log_id: int | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)


def _edit_distance(a: str, b: str) -> int:
    """Optimal-string-alignment distance (insert, delete, substitute, swap neighbours)."""
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                val = min(val, prev2[j - 2] + 1)
            cur.append(val)
        prev2, prev = prev, cur
    return prev[-1]


class _Safe(dict):
    def __missing__(self, key):
        return ""


class Chatbot:
    def __init__(
        self,
        db: Database | None = None,
        intents_path: str = DEFAULT_INTENTS,
        backend: str | None = None,
        threshold: float | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.db = db or Database()
        with open(intents_path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.fallback_cfg = data["fallback"]
        self.intents = {i["tag"]: i for i in data["intents"]}
        self.action_tag = {i["action"]: i["tag"] for i in data["intents"] if i.get("action")}

        # Wording normalisation shared by training phrases and user messages: multi-word phrases and
        # single-word synonyms are mapped to one canonical form ("billed" -> "charged", "sign in" -> "login").
        self.synonyms = {k.lower(): v.lower() for k, v in data.get("synonyms", {}).items()}
        self.phrase_res = [(re.compile(r"\b" + re.escape(a.lower()) + r"\b"), b.lower()) for a, b in data.get("phrases", [])]

        # ---- training corpus: every pattern is one document --------------
        self.docs: list[str] = []
        self.ranges: dict[str, tuple[int, int]] = {}
        for tag, intent in self.intents.items():
            start = len(self.docs)
            self.docs += intent["patterns"]
            if intent.get("label") and intent["label"] not in intent["patterns"]:
                self.docs.append(intent["label"])  # the "Did you mean" chip text must map back to its intent
            self.ranges[tag] = (start, len(self.docs))

        # Strong topic words per intent (stemmed) and an exact-phrase table. Keywords make "refund" count for
        # more than filler words like "want" or "offer"; an exact phrase always beats fuzzy matches.
        self.kw = {tag: {stem(self.synonyms.get(w.lower(), w.lower())) for w in it.get("keywords", [])} for tag, it in self.intents.items()}
        self.kw_weight = {tag: float(it.get("weight", 1.0)) for tag, it in self.intents.items()}
        self._kw_all = set().union(*self.kw.values())
        self._neutral = {stem(w.lower()) for w in data.get("neutral_words", [])}
        self.cues = {t: [self._tok(c) for c in it.get("cues", [])] for t, it in self.intents.items() if it.get("cues")}
        self.intent_vocab = {t: {stem(w) for p in it["patterns"] for w in self._tok(p) if w not in STOPWORDS} for t, it in self.intents.items()}
        self.followup_tags = {t for t, it in self.intents.items() if it.get("contextual_responses") is not None}
        self.exact: dict[str, set[str]] = {}
        for tag, it in self.intents.items():
            for phrase in it["patterns"] + ([it["label"]] if it.get("label") else []):
                self.exact.setdefault(" ".join(self._tok(phrase)), set()).add(tag)

        # vocabulary for typo correction (raw, non-stopword tokens)
        self.vocab = sorted({t for d in self.docs for t in self._tok(d) if t not in STOPWORDS and not t.isdigit()})
        self._vocab_set = set(self.vocab)
        self._pool = self.vocab + sorted(STOPWORDS)  # typo correction may also land on a filler word, which is then dropped

        self.vectorizer = TfidfVectorizer(analyzer=self._features, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.docs)

        # ---- optional transformer backend --------------------------------
        backend = (backend or os.environ.get("CHATBOT_BACKEND", "tfidf")).lower()
        self.embedder: TransformerEmbedder | None = None
        self.emb_matrix = None
        if backend == "transformers":
            try:
                self.embedder = TransformerEmbedder(os.environ.get("CHATBOT_MODEL", model_name))
                self.emb_matrix = self.embedder.encode([self._embed_text(d) for d in self.docs])
                log.info("Transformer backend enabled")
            except Exception as exc:  # missing torch/transformers, offline, ...
                log.warning("Transformer backend unavailable (%s); using TF-IDF only", exc)
                self.embedder = None
        self.backend = "transformers+tfidf" if self.embedder else "tfidf"
        if threshold is None:
            threshold = float(os.environ.get("CHATBOT_THRESHOLD", 0.40 if self.embedder else 0.32))
        self.threshold = threshold

    # ------------------------------------------------------------------ NLP
    def _tok(self, text: str) -> list[str]:
        """Normalised tokens: contractions expanded, phrases and synonyms mapped to one canonical word."""
        t = normalize(text)
        for pat, rep in self.phrase_res:
            t = pat.sub(rep, t)
        return [self.synonyms.get(x, x) for x in wordpunct_tokenize(t) if x.isalnum()]

    def _prep_query(self, text: str) -> list[str]:
        out = []
        for tok in self._tok(text):
            if tok in STOPWORDS:
                continue
            if tok not in self._vocab_set and len(tok) >= 4 and not tok.isdigit():
                tok = self._correct(tok)
                if tok in STOPWORDS:
                    continue
            out.append(stem(tok))
        return out

    def _correct(self, tok: str) -> str:
        """Fix a typo only when a vocabulary word is within 1 edit (2 for long words).

        Edit distance counts a swap of neighbouring letters as one edit, so
        "pacakge" -> "package" is fixed, while real but unknown words such as
        "cheese" are not forced onto a lookalike ("cheers").
        """
        limit = 1 if len(tok) <= 5 else 2
        best, best_d = tok, limit + 1
        for word in self._pool:
            if abs(len(word) - len(tok)) > limit or word[0] != tok[0]:
                continue
            if len(tok) <= 4 and not (len(word) == len(tok) + 1 or sorted(word) == sorted(tok)):
                continue  # short words: only fix a missing letter or a swap ("oder"), never a substitution ("made" -> "make")
            d = _edit_distance(tok, word)
            if d == 2 and word[-1] != tok[-1]:
                continue  # two edits are only trusted when the word ending agrees ("shipng" -> "shipping", not "cheese" -> "cheers")
            if d < best_d:
                best, best_d = word, d
        return best if best_d <= limit else tok

    def _features(self, text: str) -> list[str]:
        toks = self._prep_query(text)
        return toks + [f"{a}_{b}" for a, b in ngrams(toks, 2)]

    @staticmethod
    def _embed_text(text: str) -> str:
        return text.replace("ordernum", "12345")

    def classify(self, text: str) -> list[tuple[str, float]]:
        """Return intents ranked by confidence (0-1)."""
        sims = (self.vectorizer.transform([text]) @ self.matrix.T).toarray()[0]
        if self.embedder is not None:
            esims = (self.embedder.encode([self._embed_text(text)]) @ self.emb_matrix.T)[0]
            esims = np.clip((esims - 0.3) / 0.7, 0.0, 1.0)  # cosine baseline of unrelated text is ~0.2-0.3
            sims = 0.5 * sims + 0.5 * esims

        # Coverage = share of the message's meaningful words the bot knows. Context words such as "last week"
        # are not counted against it. With a topic word in the message, unknown words are probably harmless
        # context ("ship to dubai"); without one, a mostly-unknown message must not be answered.
        tokens = self._prep_query(text)
        tokset = set(tokens)
        is_known = lambda t: t in self.vectorizer.vocabulary_ or t in self._kw_all
        counted = [t for t in tokens if is_known(t) or t not in self._neutral]
        f = sum(1 for t in tokens if is_known(t)) / len(counted) if counted else 0.0
        exact = self.exact.get(" ".join(self._tok(text)), ())

        ranked = []
        for tag, (s, e) in self.ranges.items():
            raw = float(sims[s:e].max())
            if tokset & self.kw[tag]:
                score = raw * (0.4 + 0.6 * f) + KEYWORD_BONUS * (0.6 + 0.4 * f) * self.kw_weight[tag]
            else:
                score = raw * f * f
            if tag in SMALL_TALK and tokens:
                score *= len(tokset & self.intent_vocab[tag]) / len(tokens)  # "hey, check my delivery" is not a greeting
            if tag in exact:
                score = max(score, 1.0) + 1.0  # the user typed one of the known phrases word for word
            ranked.append((tag, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [(tag, min(score, 1.0)) for tag, score in ranked]

    # ------------------------------------------------------------- dialogue
    def respond(self, session_id: str, message: str) -> Reply:
        t0 = time.perf_counter()
        msg = (message or "").strip()[:500]
        state = self.db.get_session(session_id)
        state["turns"] += 1

        ents = extract_entities(msg)
        for key in ("name", "order_id"):  # remembered across turns (email is never stored)
            if key in ents:
                state["entities"][key] = ents[key]

        reply = (
            self._handle_pending(state, msg, ents)
            or self._handle_bare(state, msg, ents)
            or self._handle_intent(state, msg, ents)
        )

        reply.text = self._fmt(reply.text, state)
        self.db.save_session(session_id, state)
        latency = (time.perf_counter() - t0) * 1000
        reply.log_id = self.db.log_chat(
            session_id, msg, reply.text, reply.intent, round(reply.confidence, 4), reply.fallback, round(latency, 2)
        )
        return reply

    # -- pending slot / confirmation ------------------------------------
    def _handle_pending(self, state: dict, msg: str, ents: dict) -> Reply | None:
        pending = state.get("pending")
        if not pending:
            return None

        if pending["slot"] == "confirm":
            words = set(tokenize(msg))
            short = len(words) <= 4  # "yes", "yes please cancel it": a long sentence is a new question, not an answer
            if short and words & NO and not words & YES:
                state["pending"] = None
                return Reply("No problem. I haven't changed anything on your order.", pending["tag"], 1.0, ["Track my order"])
            if short and words & YES:
                state["pending"] = None
                return self._execute_cancel(state, pending["order_id"], pending["tag"])
            # A different question: answer it, then keep waiting for the yes/no for a couple of turns.
            pending["off"] = pending.get("off", 0) + 1
            state["pending"] = None
            if pending["off"] > 2:
                return None
            reply = self._handle_intent(state, msg, ents)
            if state["pending"] is None:
                state["pending"] = pending
                reply.text += f"\n\nBy the way, do you still want me to cancel order #{pending['order_id']}? (yes/no)"
                reply.suggestions = ["Yes", "No"]
            return reply

        if pending["slot"] == "order_id":
            ranked = self.classify(msg)
            top_tag, top_score = ranked[0]
            if "order_id" in ents:
                state["pending"] = None
                action = pending["action"]
                top = self.intents[top_tag]
                has_words = any(t not in STOPWORDS and t != "ordernum" for t in tokenize(msg))
                # A bare number answers the question we asked. Only if the user also typed
                # words ("actually, track order 10001") can the new request replace the old action.
                if has_words and top_score >= self.threshold and top.get("action") in ("order_lookup", "cancel_order"):
                    action = top["action"]
                return self._run_action(action, state, ents, from_pending=True)
            if top_score >= self.threshold:  # user moved on to a different question
                state["pending"] = None
                return None
            pending["asks"] = pending.get("asks", 0) + 1
            if pending["asks"] >= 2:
                state["pending"] = None
                return None
            return Reply(
                "I still need a valid order number (5-8 digits, e.g. 10001) to continue, or you can ask me something else.",
                pending["tag"], 1.0, ["Contact support"],
            )
        state["pending"] = None
        return None

    # -- one-word answers with nothing pending -----------------------------
    def _handle_bare(self, state: dict, msg: str, ents: dict) -> Reply | None:
        words = tokenize(msg)
        if words and all(w in YES or w in NO for w in words):
            positive = any(w in YES for w in words) and not any(w in NO for w in words)
            text = "Great! What would you like help with?" if positive else "Okay, no problem. Is there anything else I can help with?"
            return Reply(text, "acknowledge", 1.0, ["Track my order", "Return policy", "Talk to a human"])
        if "order_id" in ents and not any(t not in STOPWORDS and t != "ordernum" for t in words):
            # just an order number: continue a cancellation if that was the topic, otherwise look the order up
            action = "cancel_order" if state.get("last_intent") == "cancel_order" else "order_lookup"
            reply = self._run_action(action, state, ents)
            state["last_intent"] = self.action_tag[action]
            return reply
        return None

    # -- normal intent flow ----------------------------------------------
    def _handle_intent(self, state: dict, msg: str, ents: dict) -> Reply:
        ranked = self.classify(msg) if msg else []
        if "name" in ents and "provide_name" in self.intents:  # "my name is Sam" is an introduction, whatever else matches
            ranked = [("provide_name", 1.0)] + [r for r in ranked if r[0] != "provide_name"]
        if not ranked or ranked[0][1] < self.threshold:
            return self._fallback(state, ranked)

        tag, score = ranked[0]
        intent = self.intents[tag]
        state["fallback_streak"] = 0
        suggestions = list(intent.get("suggestions", []))

        action = intent.get("action")
        if action == "remember_name":
            name = state["entities"].get("name")
            text = random.choice(intent["responses"]) if name else "Sure, just tell me, e.g. 'my name is Sam'."
            reply = Reply(text, tag, score, suggestions, candidates=ranked[:3])
        elif action in ("order_lookup", "cancel_order"):
            reply = self._run_action(action, state, ents)
            reply.confidence, reply.candidates = score, ranked[:3]
        elif tag not in self.followup_tags and tag not in NON_TOPIC and (text := self._answer_by_cue(msg, tag)):
            state["last_intent"] = tag
            return Reply(text, tag, score, suggestions, candidates=ranked[:3])
        elif intent.get("contextual_responses") is not None:
            # "how long will it take?" refers to the current topic, but "how long does shipping take?"
            # names its own topic and must be answered about that topic.
            topic = self._explicit_topic(msg, ranked)
            ref = topic or state.get("last_intent")
            text = intent["contextual_responses"].get(ref)
            if text is None and topic:
                text = random.choice(self.intents[topic]["responses"])
            if text is None:
                text = random.choice(intent["responses"])
            if topic:
                state["last_intent"] = topic
            return Reply(text, topic or tag, score, suggestions, candidates=ranked[:3])  # a bare follow-up keeps the topic
        else:
            variants = intent.get("context_variants") or {}
            text = variants.get(state.get("last_intent")) or random.choice(intent["responses"])
            reply = Reply(text, tag, score, suggestions, candidates=ranked[:3])

        if tag not in NON_TOPIC:
            state["last_intent"] = tag
        return reply

    def _answer_by_cue(self, msg: str, topic: str) -> str | None:
        """A message that names a topic AND asks a question type ("how long", "is it free", "tell me more")
        gets the answer to that question for that topic, instead of the topic's generic answer."""
        words = self._tok(msg)
        for ftag, cue_list in self.cues.items():
            ctx = self.intents[ftag].get("contextual_responses") or {}
            if topic not in ctx:
                continue
            for cue in cue_list:
                if cue and any(words[i:i + len(cue)] == cue for i in range(len(words) - len(cue) + 1)):
                    return ctx[topic]
        return None

    def _explicit_topic(self, msg: str, ranked: list[tuple[str, float]]) -> str | None:
        """Best non-follow-up intent whose topic word appears in the message, if any."""
        tokset = set(self._prep_query(msg))
        for tag, _ in ranked:
            if tag not in self.followup_tags and tag not in NON_TOPIC and tokset & self.kw[tag]:
                return tag
        return None

    def _fallback(self, state: dict, ranked: list[tuple[str, float]]) -> Reply:
        state["fallback_streak"] += 1
        cfg = self.fallback_cfg
        top = ranked[0] if ranked else (None, 0.0)
        if state["fallback_streak"] >= 2:
            return Reply(cfg["handoff"], None, top[1], list(cfg["suggestions"]), fallback=True, candidates=ranked[:3])

        # Not confident enough to answer, but offer the closest topics as one-tap options.
        guesses = [
            t for t, s in ranked[:3]
            if s >= 0.10 and t not in NON_TOPIC and self.intents[t].get("contextual_responses") is None
        ]
        if guesses:
            labels = [self._label(t) for t in guesses[:3]]
            return Reply("I'm not completely sure what you mean. Is it one of these?", None, top[1],
                         labels + ["Talk to a human"], fallback=True, candidates=ranked[:3])
        return Reply(random.choice(cfg["responses"]), None, top[1], list(cfg["suggestions"]),
                     fallback=True, candidates=ranked[:3])

    def _label(self, tag: str) -> str:
        intent = self.intents[tag]
        text = intent.get("label") or intent["patterns"][0]
        return text[0].upper() + text[1:]

    # -- actions (slot filling against SQLite) ------------------------------
    def _run_action(self, action: str, state: dict, ents: dict, from_pending: bool = False) -> Reply:
        tag = self.action_tag[action]
        suggestions = list(self.intents[tag].get("suggestions", []))
        order_id = state["entities"].get("order_id")

        if not order_id:  # missing slot -> ask and remember what we are waiting for
            state["pending"] = {"slot": "order_id", "action": action, "tag": tag, "asks": 0}
            return Reply(random.choice(self.intents[tag]["responses"]), tag, 1.0, suggestions)

        order = self.db.get_order(order_id)
        if not order:
            state["entities"].pop("order_id", None)
            return Reply(
                f"I couldn't find order #{order_id}. Please check the number (try 10001-10005 in this demo) or contact support.",
                tag, 1.0, ["Contact support"],
            )
        remembered = "order_id" not in ents and not from_pending
        prefix = f"Using the order you mentioned earlier (#{order_id}). " if remembered else ""

        if action == "order_lookup":
            return Reply(prefix + self._describe_order(order), tag, 1.0, suggestions)

        # cancel_order: ask for confirmation before changing anything
        status = order["status"]
        if status == "Processing":
            state["pending"] = {"slot": "confirm", "action": action, "tag": tag, "order_id": order_id}
            return Reply(
                f"{prefix}Order #{order_id} ({order['item']}) hasn't shipped yet, so I can cancel it. Do you want me to cancel it? (yes/no)",
                tag, 1.0, ["Yes", "No"],
            )
        if status in ("Shipped", "Out for delivery"):
            return Reply(
                f"{prefix}Order #{order_id} has already {'shipped' if status == 'Shipped' else 'left for delivery'}, so it can't be cancelled. "
                "You can refuse the delivery or start a return once it arrives.",
                tag, 1.0, ["How to return an item", "Refund policy"],
            )
        if status == "Delivered":
            return Reply(f"{prefix}Order #{order_id} was already delivered. I can't cancel it, but you can return it within 30 days.",
                         tag, 1.0, ["How to return an item"])
        return Reply(f"{prefix}Order #{order_id} is already cancelled. Your refund takes 3-5 business days.", tag, 1.0, ["Refund policy"])

    def _execute_cancel(self, state: dict, order_id: str, tag: str) -> Reply:
        order = self.db.get_order(order_id)
        if not order or order["status"] != "Processing":
            return Reply("That order can no longer be cancelled, so I haven't made any changes.", tag, 1.0)
        self.db.set_order_status(order_id, "Cancelled")
        return Reply(
            f"Done. Order #{order_id} ({order['item']}) is cancelled and your refund will arrive in 3-5 business days.",
            tag, 1.0, ["Refund policy", "Track my order"],
        )

    @staticmethod
    def _describe_order(o: dict) -> str:
        head = f"Order #{o['order_id']} ({o['item']})"
        status = o["status"]
        if status == "Shipped":
            return f"{head} has shipped via {o['carrier']} and should arrive in about {o['eta']}."
        if status == "Processing":
            return f"{head} is being prepared and should ship soon (estimated delivery {o['eta']}). You can still cancel it or change the address."
        if status == "Out for delivery":
            return f"{head} is out for delivery with {o['carrier']} and should reach you {o['eta']}."
        if status == "Delivered":
            return f"{head} was delivered by {o['carrier']}. If something is wrong, I can help with a return or a warranty claim."
        return f"{head} was cancelled. Any refund takes 3-5 business days."

    @staticmethod
    def _fmt(text: str, state: dict) -> str:
        name = state["entities"].get("name", "")
        return text.format_map(_Safe(name=name, name_comma=f", {name}" if name else ""))
