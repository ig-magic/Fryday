# api/bot.py — Fryday single-file (full features, RTDB, Vercel)
import os, html, time, json, asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Header, HTTPException
import httpx
import google.generativeai as genai

# Google OAuth for Firebase RTDB
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

# ========= Config =========
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
BOT_ID = int(BOT_TOKEN.split(":")[0])
BOT_NAME = os.getenv("FRYDAY_BOT_NAME", "Fryday")
BOT_USERNAME = os.getenv("FRYDAY_BOT_USERNAME", "").lower()  # optional, e.g., fryday_bot
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")

# AI (Gemini)
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_FUN = os.getenv("GEMINI_MODEL_FUN", "gemini-1.5-flash")
MODEL_ANALYZE = os.getenv("GEMINI_MODEL_ANALYZE", "gemini-1.5-pro")
genai.configure(api_key=GEMINI_API_KEY)

# Media / Search
TENOR_API_KEY = os.getenv("TENOR_API_KEY", "")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY", "")
GOOGLE_CSE_CX  = os.getenv("GOOGLE_CSE_CX", "")

# Firebase Realtime Database
# Example: https://your-project-id-default-rtdb.firebaseio.com
RTDB_URL = os.environ["FIREBASE_DB_URL"].rstrip("/")
FIREBASE_SERVICE_ACCOUNT = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])

# Safety level
SAFETY_LEVEL = os.getenv("FRYDAY_SAFETY", "PG-13")

# ========= HTTP clients / App =========
tg = httpx.AsyncClient(timeout=20)
web = httpx.AsyncClient(timeout=20)
app = FastAPI()

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ========= Utils =========
def mention_html(name: str, user_id: int) -> str:
    safe = html.escape(name or "user", quote=False)
    return f'<a href="tg://user?id={user_id}">{safe}</a>'

def parse_args(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""

def parse_duration_to_seconds(s: str) -> int:
    if not s: return 0
    s = s.strip().lower()
    num = int("".join([ch for ch in s if ch.isdigit()]) or 0)
    if s.endswith(("m","min","mins")): return num * 60
    if s.endswith(("h","hr","hrs")): return num * 3600
    if s.endswith(("d","day","days")): return num * 86400
    return num

def now_ts() -> int:
    return int(time.time())

def target_id_from_reply_or_text(reply: Optional[dict], text: str) -> Optional[int]:
    if reply and "from" in reply:
        return reply["from"]["id"]
    arg = parse_args(text)
    if arg.isdigit():
        return int(arg)
    return None  # Bots can't resolve @username -> id reliably; ask to reply or give id.

# ========= Telegram helpers =========
async def tg_send_message(chat_id: int, text: str, reply_to: Optional[int] = None, disable_preview: bool = True):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": disable_preview}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
        payload["allow_sending_without_reply"] = True
    await tg.post(f"{TELEGRAM_API}/sendMessage", json=payload)

async def tg_send_animation(chat_id: int, url: str, caption: str = "", reply_to: Optional[int] = None):
    payload = {"chat_id": chat_id, "animation": url, "caption": caption, "parse_mode": "HTML"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
        payload["allow_sending_without_reply"] = True
    await tg.post(f"{TELEGRAM_API}/sendAnimation", json=payload)

async def tg_delete_message(chat_id: int, message_id: int):
    await tg.post(f"{TELEGRAM_API}/deleteMessage", json={"chat_id": chat_id, "message_id": message_id})

async def tg_get_chat_member(chat_id: int, user_id: int) -> Dict[str, Any]:
    r = await tg.get(f"{TELEGRAM_API}/getChatMember", params={"chat_id": chat_id, "user_id": user_id})
    return r.json().get("result", {}) or {}

async def tg_restrict(chat_id: int, user_id: int, until_ts: int = 0, mute: bool = True):
    if mute:
        perms = { "can_send_messages": False }
    else:
        perms = {  # allow typical permissions
            "can_send_messages": True,
            "can_send_audios": True,
            "can_send_documents": True,
            "can_send_photos": True,
            "can_send_videos": True,
            "can_send_video_notes": True,
            "can_send_voice_notes": True,
            "can_send_polls": True,
            "can_send_other_messages": True,
            "can_add_web_page_previews": True
        }
    await tg.post(f"{TELEGRAM_API}/restrictChatMember", json={
        "chat_id": chat_id, "user_id": user_id,
        "permissions": perms, "until_date": until_ts
    })

async def tg_ban(chat_id: int, user_id: int):
    await tg.post(f"{TELEGRAM_API}/banChatMember", json={"chat_id": chat_id, "user_id": user_id})

async def tg_unban(chat_id: int, user_id: int):
    await tg.post(f"{TELEGRAM_API}/unbanChatMember", json={"chat_id": chat_id, "user_id": user_id, "only_if_banned": False})

async def tg_promote(chat_id: int, user_id: int, promote: bool):
    await tg.post(f"{TELEGRAM_API}/promoteChatMember", json={
        "chat_id": chat_id, "user_id": user_id,
        "can_manage_chat": promote,
        "can_delete_messages": promote,
        "can_restrict_members": promote,
        "can_promote_members": promote,
        "can_manage_topics": promote,
        "can_invite_users": promote
    })

async def is_admin(chat_id: int, user_id: int) -> bool:
    cm = await tg_get_chat_member(chat_id, user_id)
    return cm.get("status") in ("administrator", "creator")

# ========= Firebase RTDB (OAuth via service account) =========
_creds = service_account.Credentials.from_service_account_info(
    FIREBASE_SERVICE_ACCOUNT,
    scopes=[
        "https://www.googleapis.com/auth/firebase.database",
        "https://www.googleapis.com/auth/userinfo.email"
    ],
)

async def _ensure_token():
    # Refresh token if needed (blocking -> thread)
    if not _creds.valid or _creds.expired:
        await asyncio.to_thread(_creds.refresh, GoogleAuthRequest())

async def rtdb_request(method: str, path: str, data: Optional[dict] = None):
    await _ensure_token()
    url = f"{RTDB_URL}{path}.json"
    headers = {"Authorization": f"Bearer {_creds.token}"}
    if method == "GET":
        r = await web.get(url, headers=headers)
    elif method == "PUT":
        r = await web.put(url, headers=headers, json=data)
    elif method == "PATCH":
        r = await web.patch(url, headers=headers, json=data)
    elif method == "POST":
        r = await web.post(url, headers=headers, json=data)
    else:
        raise ValueError("Unsupported method")
    r.raise_for_status()
    if r.text:
        try:
            return r.json()
        except Exception:
            return None
    return None

async def db_get(path: str, default: Any = None) -> Any:
    val = await rtdb_request("GET", path)
    return default if val is None else val

async def db_put(path: str, obj: dict):
    return await rtdb_request("PUT", path, obj)

async def db_patch(path: str, obj: dict):
    return await rtdb_request("PATCH", path, obj)

async def db_post(path: str, obj: dict):
    return await rtdb_request("POST", path, obj)

# Warnings
async def warn_add(chat_id: int, user_id: int, by_id: int, reason: str = "") -> int:
    path = f"/warnings/{chat_id}/{user_id}"
    cur = await db_get(path, default={})
    count = int(cur.get("count", 0)) + 1
    hist = cur.get("history", [])
    hist.append({"by": by_id, "reason": reason, "at": now_ts()})
    hist = hist[-10:]
    await db_put(path, {"count": count, "history": hist, "updated_at": now_ts()})
    # Log admin action
    await db_post(f"/admin_actions/{chat_id}", {"action": "warn", "by": by_id, "target": user_id, "reason": reason, "at": now_ts()})
    return count

async def warn_clear(chat_id: int, user_id: int):
    await db_put(f"/warnings/{chat_id}/{user_id}", {"count": 0, "history": [], "updated_at": now_ts()})

# Rate limiter (persisted)
async def rl_allow(key: str, interval_s: int) -> bool:
    path = f"/rate_limits/{key}"
    doc = await db_get(path, default={})
    last = int(doc.get("last", 0))
    now = now_ts()
    if now - last < interval_s:
        return False
    await db_put(path, {"last": now})
    return True

# ========= External APIs =========
async def tenor_gif(q: str) -> Optional[str]:
    if not TENOR_API_KEY:
        return None
    url = "https://tenor.googleapis.com/v2/search"
    params = {"q": q, "key": TENOR_API_KEY, "limit": 1, "media_filter": "gif", "random": True}
    r = await web.get(url, params=params)
    try:
        return r.json()["results"][0]["media_formats"]["gif"]["url"]
    except Exception:
        return None

async def web_search(q: str, n: int = 3):
    if not (GOOGLE_CSE_KEY and GOOGLE_CSE_CX):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    r = await web.get(url, params={"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "q": q})
    items = r.json().get("items", [])[:n]
    return [{"title": it.get("title",""), "url": it.get("link",""), "source": it.get("displayLink","")} for it in items]

async def image_search(q: str, n: int = 3):
    if not (GOOGLE_CSE_KEY and GOOGLE_CSE_CX):
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    r = await web.get(url, params={"key": GOOGLE_CSE_KEY, "cx": GOOGLE_CSE_CX, "q": q, "searchType": "image"})
    items = r.json().get("items", [])[:n]
    return [{"title": it.get("title",""), "url": it.get("link","")} for it in items]

async def wiki_summary(topic: str) -> Optional[dict]:
    slug = topic.replace(" ", "_")
    r = await web.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
    if r.status_code != 200:
        return None
    j = r.json()
    return {
        "title": j.get("title", topic),
        "extract": j.get("extract", ""),
        "url": j.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{slug}")
    }

# ========= AI =========
PERSONA = (
    "You are Fryday, a witty, friendly Telegram bot. Keep replies short (1–3 lines) in group chats. "
    "Tone playful/helpful, PG-13 only. No slurs, no identity-based insults. Use mild, safe humor. "
    "Mentions are provided as HTML links."
)

FUN_TEMPLATES = {
    "slap": "Write 1–2 playful, cartoony lines where {A} ‘slaps’ {B} in a silly way. One emoji. No profanity/gore.",
    "hug": "Write 1 warm line where {A} gives {B} a cozy hug. One wholesome emoji.",
    "kick": "Write 1 playful, fictional ‘kick’ from {A} to {B}, like a video game gag. One emoji.",
    "insult": "Write a light PG-13 roast from {A} to {B}, tease habits (late replies, messy desk), never identity/appearance.",
    "laugh": "Write 1 short laughing reaction from {A} aimed at {B}, with a funny sound and one emoji.",
    "cry": "Write 1 short dramatic ‘crying’ line from {A} about {B}, playful and harmless. One emoji.",
    "bonk": "Write 1 meme-y ‘bonk! go to horny jail’ gag from {A} to {B}. One emoji.",
    "shrug": "Write 1 witty ‘I don’t know’ line from {A}, referencing {B}, with a shrug tone. One emoji.",
    "kys": "Interpret ‘kys’ as ‘Keep Yourself Safe’. One supportive, non-judgmental line to {B} with a gentle emoji."
}

async def ai_generate(prompt: str, which: str = "fun") -> str:
    model_name = MODEL_FUN if which == "fun" else MODEL_ANALYZE
    model = genai.GenerativeModel(model_name)
    resp = await asyncio.to_thread(model.generate_content, f"{PERSONA}\n\n{prompt}")
    return (getattr(resp, "text", "") or "✨").strip()

async def ai_fun(cmd: str, a_html: str, b_html: str) -> str:
    tmpl = FUN_TEMPLATES[cmd].format(A=a_html, B=b_html)
    return await ai_generate(tmpl, which="fun")

async def ai_short_reply(user_html: str, text: str) -> str:
    prompt = (
        f"The user {user_html} said: “{text}”. "
        "Reply in 1–2 short lines, match tone, be helpful or playful. PG-13."
    )
    return await ai_generate(prompt, which="analyze")

async def ai_story(theme: str) -> str:
    return await ai_generate(f"Write a short creative story (120–180 words) about: {theme}. Add a playful twist and one emoji.", "fun")

async def ai_poem(style: str, topic: str) -> str:
    return await ai_generate(f"Write a brief poem in the style of {style} about {topic}. 6–10 lines, PG-13.", "fun")

async def ai_code(lang: str, task: str) -> str:
    return await ai_generate(f"Provide a concise {lang} code snippet to: {task}. Include brief comments. Fit for chat.", "analyze")

async def ai_debug(code: str) -> str:
    return await ai_generate(f"Find issues and propose fixes. Return corrected snippet and 1–2 bullet notes.\n\nCode:\n{code}", "analyze")

async def ai_roast(target: str) -> str:
    return await ai_generate(f"PG-13 roast aimed at {target}. Tease habits only. No identity/appearance/profanity.", "fun")

async def ai_analyze_topic(topic: str) -> str:
    return await ai_generate(f"Analyze: {topic}. Return a tight summary with 3–5 bullets.", "analyze")

async def ai_rewrite(style: str, text: str) -> str:
    return await ai_generate(f"Rewrite the following in the style of {style}:\n\n{text}", "fun")

async def ai_discuss(topic: str) -> str:
    return await ai_generate(f"Start an engaging discussion about: {topic}. Ask one open question at the end.", "fun")

# ========= Health =========
@app.get("/api/health")
async def health():
    return {"ok": True, "bot": BOT_NAME}

# ========= Webhook =========
@app.post("/api/bot")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str = Header(None)):
    if WEBHOOK_SECRET and x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="bad secret")
    update = await request.json()
    try:
        await handle_update(update)
    except Exception as e:
        # Avoid 500 to keep webhook healthy
        print("Error:", e)
    return {"ok": True}

# ========= Core Handler =========
async def handle_update(update: dict):
    msg = update.get("message") or update.get("edited_message")
    if not msg or "text" not in msg:
        return
    chat = msg["chat"]; chat_id = chat["id"]
    text = msg["text"].strip()
    from_user = msg.get("from", {})
    msg_id = msg["message_id"]
    reply = msg.get("reply_to_message")

    # Slash commands
    if text.startswith("/"):
        cmd = text.split()[0].split("@")[0].lower()
        if cmd == "/start":
            user_html = mention_html(from_user.get("first_name","User"), from_user["id"])
            return await tg_send_message(chat_id, f"Hey {user_html}! I’m {html.escape(BOT_NAME)} — fun, chatty, and a tiny bit smart. Try /help ✨", reply_to=msg_id)
        if cmd == "/help":
            txt = (
                "⚡ Basic\n"
                "/search <q> — Google results\n"
                "/imagesearch <q> — Images\n"
                "/gif <q> — Tenor GIF\n"
                "/wiki <topic> — Wikipedia summary\n\n"
                "🎭 Fun (reply to someone)\n"
                "/slap /hug /kick /insult /laugh /cry /bonk /shrug /kys\n\n"
                "🤖 AI\n"
                ".story | .poem | .code | .debug | .roast | .analyze | .rewrite | .discuss\n\n"
                "👮 Admin\n"
                ".del (reply) | .warn/.unwarn (reply) | .mute/.unmute (reply) | .ban/.unban/.kick (reply)\n"
                ".promote/.demote (reply or numeric id)"
            )
            return await tg_send_message(chat_id, txt)

        if cmd == "/search":
            q = parse_args(text)
            if not q: return await tg_send_message(chat_id, "Usage: /search <query>")
            hits = await web_search(q)
            if not hits: return await tg_send_message(chat_id, "No results.")
            lines = [f"• <b>{html.escape(h['title'])}</b>\n{h['url']} ({html.escape(h['source'])})" for h in hits]
            return await tg_send_message(chat_id, "\n\n".join(lines), disable_preview=False)

        if cmd == "/imagesearch":
            q = parse_args(text)
            if not q: return await tg_send_message(chat_id, "Usage: /imagesearch <query>")
            imgs = await image_search(q)
            if not imgs: return await tg_send_message(chat_id, "No images found.")
            lines = [f"• {html.escape(img['title'])}\n{img['url']}" for img in imgs]
            return await tg_send_message(chat_id, "\n\n".join(lines), disable_preview=False)

        if cmd == "/gif":
            q = parse_args(text) or "funny"
            url = await tenor_gif(q)
            if url: return await tg_send_animation(chat_id, url, caption=f"GIF: {html.escape(q)}")
            return await tg_send_message(chat_id, "No GIF found. Try a different query.")

        if cmd == "/wiki":
            q = parse_args(text)
            if not q: return await tg_send_message(chat_id, "Usage: /wiki <topic>")
            s = await wiki_summary(q)
            if not s: return await tg_send_message(chat_id, "No summary found.")
            return await tg_send_message(chat_id, f"🧠 <b>{html.escape(s['title'])}</b>\n{s['extract']}\n\n🔗 {s['url']}", disable_preview=False)

        # Fun (reply required)
        FUNS = {"/slap":"slap","/hug":"hug","/kick":"kick","/insult":"insult","/laugh":"laugh","/cry":"cry","/bonk":"bonk","/shrug":"shrug","/kys":"kys"}
        if cmd in FUNS:
            if not reply or "from" not in reply:
                return await tg_send_message(chat_id, "Reply to someone to use this command.")
            a = mention_html(from_user.get("first_name","User"), from_user["id"])
            b_from = reply["from"]
            b = mention_html(b_from.get("first_name","User"), b_from["id"])
            out = await ai_fun(FUNS[cmd], a, b)
            return await tg_send_message(chat_id, out, reply_to=msg_id)

    # Dot commands (Admin + AI)
    if text.startswith("."):
        cmd = text.split()[0].lower()

        # Admin (require actor admin)
        if cmd in (".del",".warn",".unwarn",".mute",".unmute",".ban",".unban",".kick",".promote",".demote"):
            if not await is_admin(chat_id, from_user["id"]):
                return await tg_send_message(chat_id, "Admin only.")
            # .del
            if cmd == ".del":
                if reply:
                    await tg_delete_message(chat_id, reply["message_id"])
                    return
                return await tg_send_message(chat_id, "Reply to a message to delete it.")
            # .warn
            if cmd == ".warn":
                if not reply: return await tg_send_message(chat_id, "Reply to the user to warn.")
                reason = parse_args(text)
                count = await warn_add(chat_id, reply["from"]["id"], from_user["id"], reason)
                # Auto-mute at 3 warnings for 10 minutes
                if count >= 3:
                    await tg_restrict(chat_id, reply["from"]["id"], until_ts=now_ts()+600, mute=True)
                    await tg_send_message(chat_id, f"⚠️ Warning issued. Count: {count}. Auto-muted for 10m.", reply_to=msg_id)
                else:
                    await tg_send_message(chat_id, f"⚠️ Warning issued. Count: {count}", reply_to=msg_id)
                return
            # .unwarn
            if cmd == ".unwarn":
                if not reply: return await tg_send_message(chat_id, "Reply to the user to clear warnings.")
                await warn_clear(chat_id, reply["from"]["id"])
                return await tg_send_message(chat_id, "Warning cleared.", reply_to=msg_id)
            # .mute
            if cmd == ".mute":
                if not reply: return await tg_send_message(chat_id, "Reply to the user to mute.\nUsage: .mute 10m")
                secs = parse_duration_to_seconds(parse_args(text) or "10m")
                until = now_ts() + secs if secs else now_ts() + 600
                await tg_restrict(chat_id, reply["from"]["id"], until_ts=until, mute=True)
                return await tg_send_message(chat_id, f"🔇 Muted for {secs or 600} seconds.", reply_to=msg_id)
            # .unmute
            if cmd == ".unmute":
                if not reply: return await tg_send_message(chat_id, "Reply to the user to unmute.")
                await tg_restrict(chat_id, reply["from"]["id"], until_ts=0, mute=False)
                return await tg_send_message(chat_id, "🔉 Unmuted.", reply_to=msg_id)
            # .ban
            if cmd == ".ban":
                if not reply: return await tg_send_message(chat_id, "Reply to the user to ban.")
                await tg_ban(chat_id, reply["from"]["id"])
                await db_post(f"/admin_actions/{chat_id}", {"action":"ban","by":from_user["id"],"target":reply["from"]["id"],"at":now_ts()})
                return await tg_send_message(chat_id, "🚫 Banned.", reply_to=msg_id)
            # .unban
            if cmd == ".unban":
                target = target_id_from_reply_or_text(reply, text)
                if not target: return await tg_send_message(chat_id, "Reply to a user or pass a numeric id.")
                await tg_unban(chat_id, target)
                await db_post(f"/admin_actions/{chat_id}", {"action":"unban","by":from_user["id"],"target":target,"at":now_ts()})
                return await tg_send_message(chat_id, "✅ Unbanned.", reply_to=msg_id)
            # .kick
            if cmd == ".kick":
                if not reply: return await tg_send_message(chat_id, "Reply to the user to kick.")
                uid = reply["from"]["id"]
                await tg_ban(chat_id, uid); await tg_unban(chat_id, uid)
                await db_post(f"/admin_actions/{chat_id}", {"action":"kick","by":from_user["id"],"target":uid,"at":now_ts()})
                return await tg_send_message(chat_id, "🥾 Kicked.", reply_to=msg_id)
            # .promote
            if cmd == ".promote":
                target = target_id_from_reply_or_text(reply, text)
                if not target: return await tg_send_message(chat_id, "Reply to user or pass numeric id to promote.")
                await tg_promote(chat_id, target, promote=True)
                return await tg_send_message(chat_id, "⬆️ Promoted to admin.", reply_to=msg_id)
            # .demote
            if cmd == ".demote":
                target = target_id_from_reply_or_text(reply, text)
                if not target: return await tg_send_message(chat_id, "Reply to user or pass numeric id to demote.")
                await tg_promote(chat_id, target, promote=False)
                return await tg_send_message(chat_id, "⬇️ Demoted.", reply_to=msg_id)

        # AI commands
        if cmd == ".story":
            theme = parse_args(text) or "a digital night under neon skies"
            return await tg_send_message(chat_id, await ai_story(theme))
        if cmd == ".poem":
            parts = text.split(maxsplit=2)
            style = parts[1] if len(parts) > 1 else "haiku"
            topic = parts[2] if len(parts) > 2 else "friendship"
            return await tg_send_message(chat_id, await ai_poem(style, topic))
        if cmd == ".code":
            parts = text.split(maxsplit=2)
            lang = parts[1] if len(parts) > 1 else "python"
            task = parts[2] if len(parts) > 2 else "print hello world"
            return await tg_send_message(chat_id, await ai_code(lang, task))
        if cmd == ".debug":
            body = parse_args(text) or "print('helo world')"
            return await tg_send_message(chat_id, await ai_debug(body))
        if cmd == ".roast":
            target = parse_args(text) or "my friend"
            return await tg_send_message(chat_id, await ai_roast(target))
        if cmd == ".analyze":
            topic = parse_args(text) or "trade-offs of monolith vs microservices"
            return await tg_send_message(chat_id, await ai_analyze_topic(topic))
        if cmd == ".rewrite":
            parts = text.split(maxsplit=2)
            style = parts[1] if len(parts) > 1 else "concise professional"
            body = parts[2] if len(parts) > 2 else "This is fine, but could be better."
            return await tg_send_message(chat_id, await ai_rewrite(style, body))
        if cmd == ".discuss":
            topic = parse_args(text) or "AI safety and creativity"
            return await tg_send_message(chat_id, await ai_discuss(topic))

    # Passive replies (always-on): mention or reply-to-bot or a question
    txt_lower = text.lower()
    me_mentioned = (BOT_NAME.lower() in txt_lower) or (BOT_USERNAME and f"@{BOT_USERNAME}" in txt_lower)
    replied_to_me = (reply and reply.get("from", {}).get("id") == BOT_ID)
    is_question = "?" in text

    chat_ok = await rl_allow(f"chat_{chat_id}", 6)  # 1 reply / 6s per chat
    user_ok = await rl_allow(f"user_{chat_id}_{from_user.get('id')}", 25)  # 1 passive / 25s per user

    if (me_mentioned or replied_to_me or (is_question and user_ok)) and chat_ok:
        user_html = mention_html(from_user.get("first_name","User"), from_user["id"])
        out = await ai_short_reply(user_html, text)
        return await tg_send_message(chat_id, out, reply_to=msg_id)
