(() => {
  const $ = (id) => document.getElementById(id);
  const messages = $("messages"), chips = $("chips"), form = $("composer"), input = $("input");
  const sendBtn = form.querySelector("button");

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const newId = () => (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, "") : String(Date.now()) + Math.random().toString(16).slice(2));
  let sessionId = store.get("chat_session") || newId();
  store.set("chat_session", sessionId);

  function addMessage(text, who, extra = {}) {
    const wrap = document.createElement("div");
    wrap.className = `msg ${who}${extra.fallback ? " fallback" : ""}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text; // textContent: never render user/bot text as HTML
    wrap.appendChild(bubble);
    if (who === "bot" && extra.logId) {
      const rate = document.createElement("div");
      rate.className = "rate";
      [["👍", 1], ["👎", -1]].forEach(([icon, value]) => {
        const b = document.createElement("button");
        b.type = "button"; b.textContent = icon; b.title = value > 0 ? "Helpful" : "Not helpful";
        b.addEventListener("click", async () => {
          rate.querySelectorAll("button").forEach((x) => (x.disabled = true));
          b.classList.add("on");
          fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
                                   body: JSON.stringify({ log_id: extra.logId, value }) }).catch(() => {});
        });
        rate.appendChild(b);
      });
      wrap.appendChild(rate);
    }
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
    return wrap;
  }

  const toBottom = () => { messages.scrollTop = messages.scrollHeight; };

  function setChips(items) {
    chips.replaceChildren();
    (items || []).forEach((label) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label;
      b.addEventListener("click", () => send(label));
      chips.appendChild(b);
    });
    toBottom(); // the chip row changes the height of the message pane, so re-pin to the newest message
  }

  function typing() {
    const wrap = document.createElement("div");
    wrap.className = "msg bot typing";
    wrap.innerHTML = '<div class="bubble"><i></i><i></i><i></i></div>';
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
    return wrap;
  }

  async function send(text) {
    text = text.trim();
    if (!text) return;
    addMessage(text, "user");
    setChips([]);
    input.value = "";
    sendBtn.disabled = true;
    const dots = typing();
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
                                             body: JSON.stringify({ message: text, session_id: sessionId }) });
      if (!res.ok) throw new Error(res.status);
      const data = await res.json();
      sessionId = data.session_id; store.set("chat_session", sessionId);
      dots.remove();
      addMessage(data.reply, "bot", { logId: data.log_id, fallback: data.fallback });
      setChips(data.suggestions);
    } catch {
      dots.remove();
      addMessage("Sorry, I couldn't reach the server. Please try again.", "bot");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function welcome() {
    messages.replaceChildren();
    addMessage("Hi! I'm the ShopEase assistant. I can track or cancel orders and answer questions about returns, refunds, shipping and payments.", "bot");
    setChips(["Track my order", "Return policy", "Payment methods", "Talk to a human"]);
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
  $("new-chat").addEventListener("click", () => { sessionId = newId(); store.set("chat_session", sessionId); welcome(); });
  welcome();
})();
