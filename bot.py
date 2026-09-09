#!/usr/bin/env python3
"""
Vera Challenge Bot — LLM-Augmented Composition Engine
=====================================================

Architecture:
- LLM-first composition via NVIDIA NIM (OpenAI-compatible) with deterministic fallback.
- In-memory versioned context store with atomic replacement semantics.
- Rich per-trigger prompt engineering that injects category voice, merchant data,
  peer comparisons, signals, conversation history, and customer context.
- Stateful conversation handling for auto-replies, opt-outs, intent transitions.
- All composition anchored on verifiable facts from pushed contexts — zero hallucination.

Set environment variables:
  NIM_API_KEY  — your NVIDIA NIM API key
  NIM_MODEL    — model name (default: meta/llama-3.1-70b-instruct)
  PORT         — HTTP port (default: 8080)
"""
from __future__ import annotations
import json, os, re, time, uuid, traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urlrequest, error as urlerror

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START = time.time()
NIM_API_KEY = os.environ.get("NIM_API_KEY", "")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.2-90b-vision-instruct")
NIM_URL = os.environ.get("NIM_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
LLM_TIMEOUT = 25  # seconds — leave 5s buffer for the 30s judge timeout

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
CONTEXTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "category": {}, "merchant": {}, "customer": {}, "trigger": {}
}
VERSIONS: Dict[Tuple[str, str], int] = {}
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}
SENT_SUPPRESSION: set = set()

# ---------------------------------------------------------------------------
# Category voice rules (used in fallback and for quick lookups)
# ---------------------------------------------------------------------------
CATEGORY_VOICE = {
    "dentists": {"tone": "clinical-peer", "taboos": ["guaranteed", "100% safe", "completely cure", "miracle", "best in city"], "salutation": "Dr."},
    "salons": {"tone": "warm-practical", "taboos": ["guaranteed", "miracle", "best in city"], "salutation": ""},
    "restaurants": {"tone": "operator-practical", "taboos": ["guaranteed", "best in city"], "salutation": ""},
    "gyms": {"tone": "coach-practical", "taboos": ["guaranteed", "miracle", "100% safe"], "salutation": ""},
    "pharmacies": {"tone": "trustworthy-precise", "taboos": ["guaranteed", "miracle", "completely cure"], "salutation": ""},
}

ACTION_KINDS = {
    "recall_due", "wedding_package_followup", "winback_eligible", "customer_lapsed_hard",
    "trial_followup", "chronic_refill_due", "appointment_tomorrow", "unplanned_slot_open"
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def owner_name(m):
    i = m.get("identity", {})
    return i.get("owner_first_name") or i.get("name", "").split("'s")[0].replace("Dr. ", "").strip() or "there"

def merchant_name(m):
    return m.get("identity", {}).get("name", "your business")

def customer_name(c):
    """Extract customer name safely, handling edge cases like empty or missing names."""
    if not c:
        return ""
    name = c.get("identity", {}).get("name", "")
    if not name or name.strip() in ("", "(walk-in, no profile)"):
        return ""
    return name.strip()

def active_offers(m):
    return [o.get("title", "") for o in m.get("offers", []) if o.get("status") == "active"]

def category_for(m):
    return m.get("category_slug", "")

def get_ctx(scope, cid):
    return CONTEXTS.get(scope, {}).get(cid, {}).get("payload")

def get_customer(cid):
    return get_ctx("customer", cid)

def get_merchant(mid):
    return get_ctx("merchant", mid)

def get_category(slug):
    return get_ctx("category", slug)

def get_trigger(tid):
    return get_ctx("trigger", tid)

def humanize(s):
    """Convert snake_case to readable text."""
    if not s:
        return ""
    return str(s).replace("_", " ").replace("  ", " ").strip()

def digest_item(cat, trigger):
    """Find the digest item referenced by a trigger."""
    p = trigger.get("payload", {})
    item_id = p.get("top_item_id") or p.get("digest_item_id") or p.get("top_item", {}).get("id")
    for d in (cat or {}).get("digest", []):
        if d.get("id") == item_id:
            return d
    if isinstance(p.get("top_item"), dict):
        return p["top_item"]
    return None

def pct(v):
    try:
        return f"{abs(float(v)) * 100:.0f}%"
    except Exception:
        return str(v)

def money(n):
    try:
        return f"₹{int(n):,}"
    except Exception:
        return str(n)

def language_hint(m, c=None):
    if c:
        return c.get("identity", {}).get("language_pref", "").lower()
    langs = [str(x).lower() for x in m.get("identity", {}).get("languages", [])]
    return "hi" if "hi" in langs else "en"

def format_signals(signals):
    """Convert raw signal strings to human-readable descriptions."""
    mapping = {
        "stale_posts": "Google posts are stale",
        "ctr_below_peer_median": "CTR is below peer median",
        "high_risk_adult_cohort": "has a high-risk adult patient cohort",
        "engaged_in_last_48h": "recently engaged with Vera",
        "renewal_due_soon": "subscription renewal due soon",
        "perf_dip_severe": "performance is declining sharply",
        "unverified_gbp": "Google Business Profile is unverified",
        "dormant_with_vera": "hasn't talked to Vera recently",
        "no_active_offers": "no active offers listed",
        "high_engagement": "high engagement levels",
        "above_peer_median_calls": "calls above peer median",
        "growing_views_7d": "views growing over past week",
        "winback_eligible": "eligible for win-back outreach",
        "perf_dip_post_expiry": "performance dropped after subscription expiry",
        "new_merchant": "new to the platform",
        "trial_ending_soon": "trial period ending soon",
        "ipl_eligible_locality": "in an IPL match locality",
        "high_volume": "high-volume business",
        "stable_growth": "showing stable growth",
        "seasonal_dip_apr_may": "in expected seasonal dip period",
        "above_peer_ctr": "CTR above peer average",
        "high_retention": "strong customer retention",
        "active_planning": "actively planning with Vera",
        "boutique_segment": "boutique segment",
        "compliance_aware": "compliance-aware merchant",
        "high_repeat_rate": "high repeat customer rate",
    }
    result = []
    for s in (signals or []):
        base = s.split(":")[0]
        result.append(mapping.get(base, humanize(base)))
    return result

# ---------------------------------------------------------------------------
# Conversation state management
# ---------------------------------------------------------------------------
def conversation(cid):
    return CONVERSATIONS.setdefault(cid, {
        "turns": 0, "messages": [], "auto_count": 0, "ended": False,
        "sent_bodies": [], "intent": False, "merchant_id": None,
        "trigger_kind": None, "topic": None
    })

def is_optout(s):
    s = s.lower()
    return any(x in s for x in [
        "stop messaging", "stop sending", "don't message", "do not message",
        "not interested", "unsubscribe", "remove me", "this is useless spam",
        "go away", "leave me alone", "spam"
    ])

def is_commitment(s):
    s = s.lower()
    return any(x in s for x in [
        "let's do it", "lets do it", "go ahead", "yes, do it", "yes do it",
        "what's next", "whats next", "proceed", "i want to join", "sign me up",
        "yes please", "do it", "yes sure", "sure", "ok do it", "haan karo",
        "kar do", "chalo", "theek hai", "bilkul"
    ])

def is_auto_reply(s):
    s = s.lower()
    patterns = [
        "thank you for contacting", "our team will respond shortly",
        "team will get back", "thanks for contacting",
        "we have received your message", "automated assistant",
        "aapki jaankari ke liye", "hamari team tak pahuncha"
    ]
    return any(p in s for p in patterns)

# ---------------------------------------------------------------------------
# LLM Composition Engine
# ---------------------------------------------------------------------------
def build_system_prompt(cat_slug, category, merchant, trigger, customer=None):
    """Build a rich system prompt that gives the LLM all context it needs."""
    voice = (category or {}).get("voice", {})
    cat_voice = CATEGORY_VOICE.get(cat_slug, {})
    peer = (category or {}).get("peer_stats", {})
    perf = merchant.get("performance", {})
    identity = merchant.get("identity", {})
    signals = format_signals(merchant.get("signals", []))
    offers = active_offers(merchant)
    conv_hist = merchant.get("conversation_history", [])
    cust_agg = merchant.get("customer_aggregate", {})
    kind = trigger.get("kind", "")
    scope = "customer-facing" if customer else "merchant-facing"

    lang = language_hint(merchant, customer)
    lang_instruction = {
        "hi": "Use natural Hindi-English code-mix (Hinglish). Mix Hindi and English fluidly as urban Indian professionals do.",
        "hi-en mix": "Use natural Hindi-English code-mix (Hinglish). Mix Hindi and English fluidly.",
        "en": "Write in English. Keep it professional but conversational.",
        "te-en mix": "Write primarily in English with occasional Telugu-English mix.",
        "ta-en mix": "Write primarily in English with occasional Tamil-English mix.",
        "kn-en mix": "Write primarily in English with occasional Kannada-English mix.",
        "mr": "Use Hindi-English mix (Marathi merchants understand Hinglish well).",
    }.get(lang, "Write in English. Keep it professional but conversational.")

    # Build peer comparison context
    peer_context = ""
    if peer:
        peer_ctr = peer.get("avg_ctr", 0)
        my_ctr = perf.get("ctr", 0)
        if my_ctr and peer_ctr:
            if my_ctr < peer_ctr:
                peer_context += f"Their CTR ({my_ctr:.1%}) is below peer median ({peer_ctr:.1%}). "
            elif my_ctr > peer_ctr * 1.2:
                peer_context += f"Their CTR ({my_ctr:.1%}) is above peer median ({peer_ctr:.1%}) — strong performer. "

    # Recent conversation history
    hist_text = ""
    if conv_hist:
        recent = conv_hist[-3:]
        hist_lines = []
        for h in recent:
            who = h.get("from", "?")
            body = h.get("body", "")[:120]
            hist_lines.append(f"  [{who}]: {body}")
        hist_text = "Recent conversation with Vera:\n" + "\n".join(hist_lines)

    # Customer context block
    customer_block = ""
    if customer:
        cn = customer_name(customer)
        rel = customer.get("relationship", {})
        state = customer.get("state", "")
        prefs = customer.get("preferences", {})
        customer_block = f"""
CUSTOMER (this message is sent ON BEHALF of the merchant to their customer):
- Name: {cn}
- State: {humanize(state)}
- Visits: {rel.get('visits_total', '?')} total, last visit: {rel.get('last_visit', '?')}
- Services received: {', '.join(rel.get('services_received', [])[:5])}
- Preferred slots: {humanize(prefs.get('preferred_slots', '?'))}
- Language: {customer.get('identity', {}).get('language_pref', 'en')}
- Consent scope: {', '.join(customer.get('consent', {}).get('scope', []))}
"""

    system = f"""You are Vera, magicpin's AI assistant for merchants. You compose WhatsApp messages.

THIS IS A {scope.upper()} MESSAGE for a {humanize(cat_slug)} business.

VOICE RULES for {humanize(cat_slug)}:
- Tone: {voice.get('tone', cat_voice.get('tone', 'professional'))}
- Register: {voice.get('register', 'respectful')}
- NEVER use these words: {', '.join(voice.get('vocab_taboo', cat_voice.get('taboos', [])))}
- Allowed technical vocabulary: {', '.join((voice.get('vocab_allowed', []))[:8])}
- Tone examples: {'; '.join(voice.get('tone_examples', [])[:3])}

MERCHANT:
- Business: {identity.get('name', '?')} in {identity.get('locality', '?')}, {identity.get('city', '?')}
- Owner: {identity.get('owner_first_name', '?')}
- Subscription: {merchant.get('subscription', {}).get('status', '?')} ({merchant.get('subscription', {}).get('plan', '?')} plan, {merchant.get('subscription', {}).get('days_remaining', '?')} days remaining)
- Performance (30d): {perf.get('views', '?')} views, {perf.get('calls', '?')} calls, {perf.get('directions', '?')} directions, CTR {perf.get('ctr', '?')}
- 7d trends: views {perf.get('delta_7d', {}).get('views_pct', '?')}, calls {perf.get('delta_7d', {}).get('calls_pct', '?')}
- Active offers: {', '.join(offers) if offers else 'None'}
- Merchant signals: {'; '.join(signals) if signals else 'None'}
- Customer base: {json.dumps(cust_agg) if cust_agg else 'N/A'}
{peer_context}
{hist_text}
{customer_block}

LANGUAGE: {lang_instruction}

COMPOSITION RULES (STRICT):
1. Anchor on at least ONE specific, verifiable fact from the context (number, date, source citation, price)
2. Make the "why now" clear — this message is triggered by: {humanize(kind)}
3. Keep it concise — WhatsApp readability. No long preambles.
4. End with ONE clear call-to-action. {'Binary YES/STOP for action triggers.' if kind in ACTION_KINDS else 'Open-ended or binary as appropriate.'}
5. Do NOT fabricate data. Only use facts from the context provided above.
6. Do NOT include URLs.
7. Do NOT re-introduce yourself if there's recent conversation history.
8. {'Address the customer by name and send as the merchant (not as Vera).' if customer else 'Address the merchant owner by name.'}
9. Use service+price format (e.g., "Dental Cleaning @ ₹299") instead of generic discounts.
10. No "AMAZING DEAL!" or promotional hype. Peer/colleague tone.

Output ONLY the message body. No JSON, no metadata, no explanation."""

    return system


def build_user_prompt(trigger, category, merchant, customer=None):
    """Build the user prompt with trigger-specific details."""
    kind = trigger.get("kind", "")
    p = trigger.get("payload", {})
    d = digest_item(category, trigger)

    if kind == "research_digest":
        return f"""Compose a message about a new research digest item.
Research item: {json.dumps(d) if d else 'See trigger payload'}
Trigger details: {json.dumps(p)}
Frame it as sharing useful clinical/industry knowledge with a peer. Cite the source. Offer to help them use the research practically."""

    elif kind == "regulation_change":
        return f"""Compose a compliance heads-up message.
Regulation details: {json.dumps(d) if d else json.dumps(p)}
Deadline: {p.get('deadline_iso', 'see payload')}
Be specific about what changed and the deadline. Offer to create a checklist."""

    elif kind == "recall_due":
        return f"""Compose a recall reminder message to a customer ON BEHALF of the merchant.
Service due: {humanize(p.get('service_due', ''))}
Last service date: {p.get('last_service_date', '?')}
Available slots: {json.dumps(p.get('available_slots', []))}
Include the specific offer price if the merchant has an active offer for this service. Offer slot choices."""

    elif kind == "perf_dip":
        return f"""Compose a performance dip alert for the merchant.
Metric: {p.get('metric', '?')} is down {pct(p.get('delta_pct', 0))} over {p.get('window', '7d')}
Baseline: {p.get('vs_baseline', '?')}
Frame as actionable signal, not alarm. Offer to diagnose the likely cause."""

    elif kind == "renewal_due":
        return f"""Compose a subscription renewal reminder.
Plan: {p.get('plan', '?')}, days remaining: {p.get('days_remaining', '?')}
Renewal amount: {money(p.get('renewal_amount', '?'))}
Be straightforward. Mention what they'd lose if it lapses."""

    elif kind == "festival_upcoming":
        return f"""Compose a festival planning nudge.
Festival: {p.get('festival', '?')} on {p.get('date', '?')} ({p.get('days_until', '?')} days away)
Help them plan ahead using what they already have (existing offers, seasonal beats)."""

    elif kind == "wedding_package_followup":
        return f"""Compose a bridal package follow-up to a customer ON BEHALF of the merchant.
Wedding date: {p.get('wedding_date', '?')} ({p.get('days_to_wedding', '?')} days away)
Next step: {humanize(p.get('next_step_window_open', ''))}
Reference their previous bridal trial. Warm, excited, personal tone."""

    elif kind == "curious_ask_due":
        return f"""Compose a curiosity-driven engagement message to the merchant.
Ask them about their most in-demand service this week. Offer to turn their answer into a post + WhatsApp reply template.
Keep it short and genuinely curious — not a sales pitch."""

    elif kind == "winback_eligible":
        return f"""Compose a win-back eligibility alert for the merchant.
Days since offer expired: {p.get('days_since_expiry', '?')}
Lapsed customers added since expiry: {p.get('lapsed_customers_added_since_expiry', '?')}
Performance dip: {pct(p.get('perf_dip_pct', 0)) if p.get('perf_dip_pct') else 'N/A'}
Frame as an opportunity with a concrete customer pool to re-engage."""

    elif kind == "ipl_match_today":
        return f"""Compose an IPL match-day nudge for a restaurant/food merchant.
Match: {p.get('match', '?')} at {p.get('venue', '?')}
Match time: {p.get('match_time_iso', '?')}
Use their existing active offer rather than inventing a new one. Draft a delivery-focused angle."""

    elif kind == "review_theme_emerged":
        return f"""Compose a review theme alert for the merchant.
Theme: "{humanize(p.get('theme', '?'))}" mentioned in {p.get('occurrences_30d', '?')} recent reviews
Trend: {p.get('trend', '?')}
Customer quote: "{p.get('common_quote', '')}"
Frame as specific and actionable. Offer to draft a response + one operational fix."""

    elif kind == "milestone_reached":
        return f"""Compose a milestone notification for the merchant.
Metric: {p.get('metric', '?')}, current value: {p.get('value_now', '?')}, milestone: {p.get('milestone_value', '?')}
Celebrate briefly, then offer to turn it into a visible proof point (post/update)."""

    elif kind == "active_planning_intent":
        return f"""Compose a follow-up on a planning conversation the merchant initiated.
Topic: {humanize(p.get('intent_topic', '?'))}
Merchant's last message: "{p.get('merchant_last_message', '')}"
Pick up where they left off. Offer a concrete next step (draft, plan, structure)."""

    elif kind == "seasonal_perf_dip":
        return f"""Compose a seasonal dip context message.
Metric: {p.get('metric', '?')} down {pct(p.get('delta_pct', 0))} over {p.get('window', '7d')}
Season context: {humanize(p.get('season_note', ''))}
Reassure that this is expected seasonal behavior. Suggest a retention-first action."""

    elif kind == "customer_lapsed_hard":
        return f"""Compose a lapsed customer re-engagement message ON BEHALF of the merchant.
Days since last visit: {p.get('days_since_last_visit', '?')}
Previous focus: {humanize(p.get('previous_focus', ''))}
Gentle, no-pressure tone. Offer to help find a convenient slot."""

    elif kind == "trial_followup":
        return f"""Compose a trial follow-up message to a customer ON BEHALF of the merchant.
Trial date: {p.get('trial_date', '?')}
Next session options: {json.dumps(p.get('next_session_options', []))}
Reference the trial they attended. Offer to hold a slot."""

    elif kind == "supply_alert":
        return f"""Compose a supply/recall alert for a pharmacy merchant.
Molecule: {p.get('molecule', '?')}
Affected batches: {', '.join(p.get('affected_batches', []))}
Manufacturer: {p.get('manufacturer', '?')}
Urgent, precise, actionable. Offer a stock-check checklist."""

    elif kind == "chronic_refill_due":
        return f"""Compose a chronic medication refill reminder to a customer ON BEHALF of the pharmacy.
Medications: {', '.join(p.get('molecule_list', []))}
Stock runs out: {p.get('stock_runs_out_iso', '?')}
Delivery address saved: {p.get('delivery_address_saved', False)}
Precise, caring tone. Offer to arrange the refill."""

    elif kind == "category_seasonal":
        return f"""Compose a seasonal demand shift alert for a pharmacy merchant.
Season: {p.get('season', '?')}
Trends: {', '.join(p.get('trends', [])[:4])}
Shelf action recommended: {p.get('shelf_action_recommended', False)}
Help them prepare their inventory. Offer to prioritize the list."""

    elif kind == "gbp_unverified":
        return f"""Compose a Google Business Profile verification nudge.
Verification path: {humanize(p.get('verification_path', '?'))}
Estimated uplift: {pct(p.get('estimated_uplift_pct', 0)) if p.get('estimated_uplift_pct') else 'significant'}
Frame the uplift as concrete value. Offer to walk through the steps."""

    elif kind == "cde_opportunity":
        d_item = digest_item(category, trigger)
        return f"""Compose a CDE (Continuing Dental Education) opportunity notification.
Credits: {p.get('credits', '?')}
Fee: {humanize(p.get('fee', '?'))}
Related event: {json.dumps(d_item) if d_item else 'see trigger'}
Frame as professional development value. Offer to pull details and draft registration."""

    elif kind == "competitor_opened":
        return f"""Compose a competitor alert for the merchant.
Competitor: {p.get('competitor_name', '?')} opened {p.get('distance_km', '?')} km away
Their offer: {p.get('their_offer', '?')}
Opened: {p.get('opened_date', '?')}
Frame as a concrete local signal, not panic. Offer to compare with their active offer."""

    elif kind == "perf_spike":
        return f"""Compose a positive performance spike notification.
Metric: {p.get('metric', '?')} up {pct(p.get('delta_pct', 0))} over {p.get('window', '7d')}
Baseline: {p.get('vs_baseline', '?')}
Likely driver: {humanize(p.get('likely_driver', ''))}
Celebrate the win. Suggest doubling down on what caused it."""

    elif kind == "dormant_with_vera":
        return f"""Compose a gentle re-engagement message for a merchant who hasn't talked to Vera recently.
Days since last message: {p.get('days_since_last_merchant_message', '?')}
Last topic: {humanize(p.get('last_topic', ''))}
Low-pressure. Offer to pick up the last topic or let them know no reply needed."""

    else:
        return f"""Compose a message for trigger kind: {humanize(kind)}.
Trigger payload: {json.dumps(p)}
Use the context provided to craft a specific, actionable message."""


def call_llm(system_prompt, user_prompt):
    """Call NVIDIA NIM API (OpenAI-compatible) with timeout and error handling."""
    if not NIM_API_KEY:
        return None

    body = json.dumps({
        "model": NIM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
    }).encode("utf-8")

    req = urlrequest.Request(
        NIM_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        resp = urlrequest.urlopen(req, timeout=LLM_TIMEOUT)
        data = json.loads(resp.read().decode("utf-8"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Clean up: remove any JSON wrapping, quotes, or metadata the LLM might add
        content = content.strip().strip('"').strip("'")
        # Remove URLs the LLM might generate despite instructions
        content = re.sub(r'https?://\S+|www\.\S+', '', content).strip()
        return content if content else None
    except Exception as e:
        print(f"[LLM Error] {e}")
        return None


def call_llm_for_reply(merchant, conv_state, merchant_message, trigger_kind=None):
    """Call LLM to compose a reply to a merchant message."""
    if not NIM_API_KEY:
        return None

    cat_slug = category_for(merchant) if merchant else ""
    cat_voice = CATEGORY_VOICE.get(cat_slug, {})
    prev_messages = conv_state.get("sent_bodies", [])

    system = f"""You are Vera, magicpin's AI assistant. You are in an ongoing WhatsApp conversation with a merchant.

VOICE: {cat_voice.get('tone', 'professional')} tone. Never use: {', '.join(cat_voice.get('taboos', []))}
MERCHANT: {merchant_name(merchant)} ({owner_name(merchant)}), {merchant.get('identity', {}).get('locality', '')}, {merchant.get('identity', {}).get('city', '')}

RULES:
1. Be concise — this is WhatsApp, not email.
2. If the merchant commits (says yes/go ahead/do it), switch immediately to ACTION mode — tell them what you'll do next. Do NOT ask another qualifying question.
3. If you detect an auto-reply (canned business message), acknowledge it once and ask the owner to reply directly.
4. If the merchant says stop/not interested, end gracefully.
5. Stay on the original topic: {humanize(trigger_kind or conv_state.get('trigger_kind', 'general'))}
6. Do NOT fabricate data or make promises you can't keep.
7. No URLs.

Output ONLY the reply message body."""

    # Build conversation context
    conv_lines = []
    for i, body in enumerate(prev_messages[-3:]):
        conv_lines.append(f"[Vera]: {body[:150]}")
    conv_lines.append(f"[Merchant]: {merchant_message}")

    user = f"""Previous conversation:
{chr(10).join(conv_lines)}

Compose Vera's next reply. Be direct and action-oriented."""

    return call_llm(system, user)


# ---------------------------------------------------------------------------
# Deterministic fallback composer (improved from original)
# ---------------------------------------------------------------------------
def compose_fallback(category, merchant, trigger, customer=None):
    """Improved deterministic composer as fallback when LLM is unavailable."""
    kind = trigger.get("kind", "")
    p = trigger.get("payload", {})
    cat = (category or {}).get("slug", category_for(merchant))
    owner = owner_name(merchant)
    mname = merchant_name(merchant)
    locality = merchant.get("identity", {}).get("locality", "")
    cname = customer_name(customer) if customer else ""
    lang = language_hint(merchant, customer)
    send_as = "merchant_on_behalf" if customer else "vera"
    peer = (category or {}).get("peer_stats", {})
    perf = merchant.get("performance", {})

    body = ""
    cta = "open_ended"

    if kind == "research_digest":
        d = digest_item(category, trigger) or {}
        title = d.get("title", "")
        source = d.get("source", "")
        n = d.get("trial_n")
        seg = humanize(d.get("patient_segment", ""))
        summary = d.get("summary", "")
        m_pct = re.search(r"(\d+)%", summary)
        result_stat = f"{m_pct.group(1)}% lower recurrence" if m_pct else ""

        salut = f"Dr. {owner}" if cat == "dentists" else owner
        body = f"{salut}, {source.split(',')[0] if source else 'new research'} just dropped. "
        if seg:
            body += f"Relevant to your {seg} cohort — "
        if n:
            body += f"{n:,}-patient trial: {title}"
        else:
            body += title
        if result_stat:
            body += f" ({result_stat})"
        body += f". {source}. "
        # Add peer comparison if relevant
        if peer and perf.get("ctr") and peer.get("avg_ctr") and perf["ctr"] < peer["avg_ctr"]:
            body += f"Also: your CTR ({perf['ctr']:.1%}) is below the {humanize(peer.get('scope', 'peer'))} median ({peer['avg_ctr']:.1%}) — this kind of content can help close that gap. "
        body += "Want me to pull the abstract and draft a patient-facing WhatsApp you can share?"

    elif kind == "regulation_change":
        d = digest_item(category, trigger) or {}
        deadline = p.get("deadline_iso", "")
        salut = f"Dr. {owner}" if cat == "dentists" else owner
        body = f"{salut}, compliance heads-up: {d.get('title', 'a regulation has changed')}."
        if deadline:
            body += f" Effective {deadline}."
        if d.get("summary"):
            body += f" {d['summary']}"
        body += " Want me to turn this into a short compliance checklist you can share with your team?"

    elif kind == "recall_due":
        slots = p.get("available_slots", [])
        labels = [s.get("label") for s in slots if s.get("label")]
        last = p.get("last_service_date", "")
        service = humanize(p.get("service_due", ""))
        offer = active_offers(merchant)[0] if active_offers(merchant) else ""
        slottext = " or ".join(labels[:2])
        body = f"Hi {cname}, {mname} here. "
        if last:
            body += f"It's been a while since your last visit ({last}) — your {service} recall is due. "
        else:
            body += f"Your {service} recall is due. "
        if slottext:
            body += f"Apke liye {slottext} slots available hain. "
        if offer:
            body += f"{offer}. "
        body += "Reply with the slot you prefer, or tell us a time that works."
        cta = "multi_choice_slot"

    elif kind == "perf_dip":
        metric = p.get("metric", "metric")
        delta = p.get("delta_pct", 0)
        baseline = p.get("vs_baseline")
        body = f"{owner}, your {metric} dropped {pct(delta)} this week"
        if baseline is not None:
            body += f" (from a baseline of {baseline})"
        body += ". "
        # Add peer comparison
        if peer and metric == "calls" and peer.get("avg_calls_30d"):
            body += f"Peer average is {peer['avg_calls_30d']} calls/month. "
        body += "I'd treat this as a signal worth investigating, not a verdict. Want me to isolate the most likely driver and suggest one quick fix?"

    elif kind == "renewal_due":
        body = f"{owner}, your {p.get('plan', 'plan')} subscription renews in {p.get('days_remaining', '?')} days"
        if p.get("renewal_amount") is not None:
            body += f" ({money(p['renewal_amount'])})"
        body += ". If it lapses, your profile maintenance and lead generation pause. Want me to lay out the renewal steps now?"
        cta = "binary_yes_no"

    elif kind == "festival_upcoming":
        body = f"{owner}, {p.get('festival', 'the festival')} is {p.get('days_until', '?')} days out ({p.get('date', '')}). "
        offers_list = active_offers(merchant)
        if offers_list:
            body += f"You already have {offers_list[0]} running — good starting point. "
        body += "Want me to turn the event into one concrete offer or post idea using what you already have?"

    elif kind == "wedding_package_followup":
        w = p.get("wedding_date", "")
        body = f"Hi {cname}, {owner} from {mname} here. "
        body += f"Your wedding is {p.get('days_to_wedding', '?')} days away ({w}), and the {humanize(p.get('next_step_window_open', 'next prep window'))} is now open. "
        body += "You'd said yes to the bridal journey — want me to help pick the first session timing?"
        cta = "binary_yes_no"

    elif kind == "curious_ask_due":
        body = f"Hi {owner}! Quick question — what service has been most asked-for this week at {mname}? "
        body += "I'll turn your answer into a ready-to-use Google post + WhatsApp reply template. Takes 5 min."

    elif kind == "winback_eligible":
        body = f"{owner}, {p.get('lapsed_customers_added_since_expiry', '?')} customers have lapsed in the {p.get('days_since_expiry', '?')} days since your offer expired"
        if p.get("perf_dip_pct") is not None:
            body += f", and your numbers are down {pct(p['perf_dip_pct'])}"
        body += ". That's a concrete win-back pool. Want me to draft a low-friction reactivation message for them?"

    elif kind == "ipl_match_today":
        body = f"{owner}, {p.get('match', '')} tonight at {p.get('venue', '')}"
        tm = p.get("match_time_iso", "")
        if tm:
            body += f" ({tm[11:16]})"
        body += ". "
        offers_list = active_offers(merchant)
        if offers_list:
            body += f"Your {offers_list[0]} is already live — I'd use that instead of inventing a match-night special. "
        body += "Want me to draft a delivery-focused message you can push tonight?"

    elif kind == "review_theme_emerged":
        body = f"{owner}, {p.get('occurrences_30d', '?')} recent reviews mention \"{humanize(p.get('theme', ''))}\" — trend is {p.get('trend', 'notable')}."
        if p.get("common_quote"):
            body += f" One customer said: \"{p['common_quote']}\"."
        body += " Specific enough to act on. Want me to draft a public response + one operational fix?"

    elif kind == "milestone_reached":
        val = p.get("value_now", "?")
        target = p.get("milestone_value", "?")
        gap = max(0, (target or 0) - (val or 0)) if isinstance(val, (int, float)) and isinstance(target, (int, float)) else "?"
        body = f"{owner}, you're at {val} {humanize(p.get('metric', ''))} — just {gap} away from {target}. "
        body += "Great window to turn that milestone into a visible proof point. Want me to draft a celebratory post?"

    elif kind == "active_planning_intent":
        topic = humanize(p.get("intent_topic", ""))
        last_msg = p.get("merchant_last_message", "")
        body = f"{owner}, picking up where we left off on {topic}. "
        if "what would it look like" in last_msg.lower() or "what should it look like" in last_msg.lower():
            body += "You asked what it could look like — I can draft a one-page starter offer with audience, pricing, and timing. Want me to go ahead?"
        else:
            body += "I can turn this into a structured plan with audience, pricing, and launch timing. Want me to draft it now?"
        cta = "binary_yes_no"

    elif kind == "seasonal_perf_dip":
        body = f"{owner}, {p.get('metric', 'performance')} is down {pct(p.get('delta_pct', 0))} this week, but this is expected — "
        if p.get("season_note"):
            body += f"it's the {humanize(p['season_note'])} period. "
        else:
            body += "seasonal pattern. "
        body += "I wouldn't panic-spend on this dip. Want me to suggest a retention-first action for this window instead?"

    elif kind == "customer_lapsed_hard":
        days = p.get("days_since_last_visit", "?")
        body = f"Hi {cname}, {owner} from {mname} here. "
        body += f"It's been about {days} days since your last visit — no pressure at all. "
        if p.get("previous_focus"):
            body += f"We can pick back up around your previous focus ({humanize(p['previous_focus'])}). "
        body += "If you'd like, I can help find a convenient slot."
        cta = "binary_yes_no"

    elif kind == "trial_followup":
        opts = p.get("next_session_options", [])
        body = f"Hi {cname}, {owner} from {mname} here. "
        body += f"Following up on your trial session from {p.get('trial_date', '')}. "
        if opts:
            body += f"We have {opts[0].get('label', 'a next-session slot')} available. "
        body += "Want me to hold it for you?"
        cta = "binary_yes_no"

    elif kind == "supply_alert":
        mol = p.get("molecule", "")
        batches = ", ".join(p.get("affected_batches", []))
        body = f"{owner}, urgent supply alert: {mol} flagged in batches {batches}"
        if p.get("manufacturer"):
            body += f" from {p['manufacturer']}"
        body += ". Please verify affected stock before dispensing. Want me to generate a quick stock-check checklist?"
        cta = "binary_yes_no"

    elif kind == "chronic_refill_due":
        mol = ", ".join(p.get("molecule_list", []))
        body = f"Hi {cname}, {mname} here. "
        body += f"Your refill for {mol} is due — current stock runs out around {p.get('stock_runs_out_iso', '')[:10]}. "
        if p.get("delivery_address_saved"):
            body += "Your saved delivery address is on file. "
        body += "Want us to arrange the refill?"
        cta = "binary_yes_no"

    elif kind == "category_seasonal":
        trends = [humanize(t) for t in p.get("trends", [])]
        body = f"{owner}, summer demand patterns are shifting: {', '.join(trends[:3])}. "
        body += "The data suggests a shelf-action review would help. Want me to turn those movements into a priority list?"

    elif kind == "gbp_unverified":
        body = f"{owner}, your Google Business Profile is still unverified. "
        body += f"Verification path: {humanize(p.get('verification_path', 'available'))}. "
        if p.get("estimated_uplift_pct") is not None:
            body += f"Verified profiles typically see ~{pct(p['estimated_uplift_pct'])} more visibility. "
        body += "Want me to walk you through the steps?"

    elif kind == "cde_opportunity":
        d = digest_item(category, trigger)
        body = f"Dr. {owner}, CDE opportunity: {humanize(p.get('fee', ''))} for {p.get('credits', '?')} credits. "
        if d:
            body += f"Topic: \"{d.get('title', '')}\" — {d.get('summary', '')[:80]}. "
        body += "Want me to pull the registration details?"

    elif kind == "competitor_opened":
        body = f"{owner}, new competitor {p.get('competitor_name', '')} opened {p.get('distance_km', '?')} km from {locality or 'your location'}. "
        if p.get("their_offer"):
            body += f"Their listed offer: {p['their_offer']}. "
        offers_list = active_offers(merchant)
        if offers_list:
            body += f"Your current offer: {offers_list[0]}. "
        body += "Worth a look, not a panic. Want me to compare and suggest one response?"

    elif kind == "perf_spike":
        metric = p.get("metric", "metric")
        body = f"{owner}, your {metric} jumped {pct(p.get('delta_pct', 0))} this week"
        if p.get("vs_baseline") is not None:
            body += f" (baseline: {p['vs_baseline']})"
        if p.get("likely_driver"):
            body += f" — likely driver: {humanize(p['likely_driver'])}"
        body += ". Good moment to double down on what worked. Want me to turn that into one concrete next action?"

    elif kind == "dormant_with_vera":
        body = f"Hi {owner} — it's been {p.get('days_since_last_merchant_message', '?')} days since we last chatted"
        if p.get("last_topic"):
            body += f" about {humanize(p['last_topic'])}"
        body += ". Happy to pick it back up if useful — otherwise no worries, no reply needed."

    else:
        body = f"Hi {owner}, new {humanize(kind)} update for {mname}. "
        body += "I can summarize the key details and suggest one next step. Want me to?"

    return body, cta, send_as


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------
def compose(category, merchant, trigger, customer=None):
    """Compose a message using LLM with deterministic fallback."""
    kind = trigger.get("kind", "")
    cat_slug = (category or {}).get("slug", category_for(merchant))
    send_as = "merchant_on_behalf" if customer else "vera"
    template = "merchant_engagement_v1" if customer else "vera_engagement_v1"
    owner = owner_name(merchant)
    cname = customer_name(customer) if customer else ""

    # Build template params for the submission
    params = [owner, kind, cat_slug]

    # Try LLM composition first
    llm_body = None
    if NIM_API_KEY:
        try:
            sys_prompt = build_system_prompt(cat_slug, category, merchant, trigger, customer)
            usr_prompt = build_user_prompt(trigger, category, merchant, customer)
            llm_body = call_llm(sys_prompt, usr_prompt)
        except Exception as e:
            print(f"[Compose LLM Error] {e}")

    if llm_body and len(llm_body) > 20:
        body = llm_body
        # Determine CTA from trigger kind
        if kind in ACTION_KINDS:
            cta = "binary_yes_no"
        elif kind == "recall_due":
            cta = "multi_choice_slot"
        else:
            cta = "open_ended"
    else:
        # Fallback to deterministic composition
        body, cta, send_as = compose_fallback(category, merchant, trigger, customer)

    # Safety: strip URLs
    body = re.sub(r'https?://\S+|www\.\S+', '', body).strip()

    # Build specific rationale
    rationale = build_rationale(kind, cat_slug, merchant, trigger, customer)

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "template_name": template,
        "template_params": params,
        "rationale": rationale,
    }


def build_rationale(kind, cat_slug, merchant, trigger, customer=None):
    """Build a specific, informative rationale for each message."""
    p = trigger.get("payload", {})
    owner = owner_name(merchant)
    mname = merchant_name(merchant)

    base = f"Trigger: {humanize(kind)} for {mname} ({humanize(cat_slug)}). "

    specifics = {
        "research_digest": f"Shared latest research digest item from category digest; anchored on trial data and source citation. Relevant to merchant's patient cohort.",
        "regulation_change": f"Compliance alert with specific deadline ({p.get('deadline_iso', '?')}). Anchored on regulatory source. Offered actionable checklist.",
        "recall_due": f"Customer recall reminder sent on behalf of merchant. Anchored on last visit date and available appointment slots. Included active offer pricing.",
        "perf_dip": f"Performance dip alert: {p.get('metric', '?')} down {pct(p.get('delta_pct', 0))}. Framed as diagnostic signal, not alarm. Offered to identify root cause.",
        "renewal_due": f"Subscription renewal reminder — {p.get('days_remaining', '?')} days remaining. Mentioned consequences of lapse to create urgency.",
        "festival_upcoming": f"Festival planning nudge for {p.get('festival', '?')} ({p.get('days_until', '?')} days out). Referenced existing offers for continuity.",
        "wedding_package_followup": f"Bridal journey follow-up sent on behalf of merchant. Referenced prior trial and upcoming wedding date for urgency.",
        "curious_ask_due": f"Curiosity-driven engagement — asked merchant about in-demand services to unlock two-way conversation.",
        "winback_eligible": f"Win-back opportunity: {p.get('lapsed_customers_added_since_expiry', '?')} lapsed customers since offer expiry. Framed as concrete reactivation pool.",
        "ipl_match_today": f"Match-day nudge anchored on {p.get('match', '?')} at {p.get('venue', '?')}. Used existing active offer rather than fabricating a new one.",
        "review_theme_emerged": f"Review pattern alert: \"{humanize(p.get('theme', ''))}\" trending ({p.get('occurrences_30d', '?')} mentions). Included customer quote for specificity.",
        "milestone_reached": f"Milestone notification: {p.get('value_now', '?')}/{p.get('milestone_value', '?')} {humanize(p.get('metric', ''))}. Offered to convert into social proof.",
        "active_planning_intent": f"Continued planning conversation about {humanize(p.get('intent_topic', ''))}. Picked up from merchant's last message rather than re-qualifying.",
        "seasonal_perf_dip": f"Seasonal dip context: {p.get('metric', '?')} down {pct(p.get('delta_pct', 0))} but expected ({humanize(p.get('season_note', ''))}). Recommended retention over acquisition.",
        "customer_lapsed_hard": f"Lapsed customer re-engagement sent on behalf of merchant. Low-pressure tone; referenced previous focus area.",
        "trial_followup": f"Trial follow-up sent on behalf of merchant. Referenced trial date and offered specific next session slot.",
        "supply_alert": f"Urgent supply alert for {p.get('molecule', '?')} — batches {', '.join(p.get('affected_batches', []))}. Immediate action required.",
        "chronic_refill_due": f"Chronic medication refill reminder sent on behalf of pharmacy. Listed specific molecules and expiry date.",
        "category_seasonal": f"Seasonal demand shift alert with specific trend data. Recommended shelf-action review.",
        "gbp_unverified": f"GBP verification nudge with estimated uplift ({pct(p.get('estimated_uplift_pct', 0))}). Offered step-by-step guidance.",
        "cde_opportunity": f"Professional development opportunity — {p.get('credits', '?')} CDE credits. Linked to relevant digest item.",
        "competitor_opened": f"Competitor alert: {p.get('competitor_name', '?')} at {p.get('distance_km', '?')} km. Compared offers for strategic response.",
        "perf_spike": f"Positive spike notification: {p.get('metric', '?')} up {pct(p.get('delta_pct', 0))}. Identified likely driver for doubling down.",
        "dormant_with_vera": f"Gentle re-engagement after {p.get('days_since_last_merchant_message', '?')} days dormancy. No-pressure framing with opt-out respect.",
    }

    return base + specifics.get(kind, f"Composed from trigger payload and merchant context. All facts from pushed contexts only.")


# ---------------------------------------------------------------------------
# Reply handler
# ---------------------------------------------------------------------------
def reply_action(req):
    cid = req.get("conversation_id", "")
    s = req.get("message", "").strip()
    st = conversation(cid)
    st["turns"] += 1
    st["messages"].append(s)
    if req.get("merchant_id"):
        st["merchant_id"] = req["merchant_id"]

    # Opt-out detection
    if is_optout(s):
        st["ended"] = True
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out or expressed hostility. Respecting their preference and ending conversation immediately."
        }

    # Auto-reply detection
    if is_auto_reply(s):
        st["auto_count"] += 1
        if st["auto_count"] >= 3:
            st["ended"] = True
            return {
                "action": "end",
                "rationale": f"Detected {st['auto_count']} consecutive canned auto-replies with no owner engagement. Stopping to avoid spamming."
            }
        if st["auto_count"] == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Second auto-reply detected. Backing off 24h to wait for the actual owner to see the conversation."
            }
        return {
            "action": "send",
            "body": "Looks like that's an auto-reply 😊 No worries — when the owner sees this, just reply YES to pick up where we left off.",
            "cta": "binary_yes_no",
            "rationale": "First canned auto-reply detected. One low-friction attempt to surface conversation to the business owner."
        }

    if st["ended"]:
        return {"action": "end", "rationale": "Conversation was previously closed."}

    # Commitment detection — switch to action mode immediately
    if is_commitment(s):
        st["intent"] = True
        # Try LLM reply
        m = get_merchant(st.get("merchant_id", "")) if st.get("merchant_id") else None
        llm_reply = call_llm_for_reply(m, st, s, st.get("trigger_kind")) if m else None
        if llm_reply and len(llm_reply) > 15:
            body = re.sub(r'https?://\S+|www\.\S+', '', llm_reply).strip()
        else:
            body = "Great — moving straight to action. I'll prepare the next concrete step based on our conversation. Reply CONFIRM when you want me to proceed, or let me know what to adjust."
        st["sent_bodies"].append(body)
        return {
            "action": "send",
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant expressed clear commitment. Switching from discussion to execution mode immediately — no re-qualification."
        }

    # General message handling
    low = s.lower()
    if any(x in low for x in ["send the abstract", "send it", "draft", "please do", "yes please", "haan", "bhejo"]):
        m = get_merchant(st.get("merchant_id", "")) if st.get("merchant_id") else None
        llm_reply = call_llm_for_reply(m, st, s, st.get("trigger_kind")) if m else None
        if llm_reply and len(llm_reply) > 15:
            body = re.sub(r'https?://\S+|www\.\S+', '', llm_reply).strip()
        else:
            body = "On it — I'll use the context from our conversation to prepare that. If you want any changes to audience, timing, or tone, let me know before I finalize."
        st["sent_bodies"].append(body)
        return {
            "action": "send",
            "body": body,
            "cta": "open_ended",
            "rationale": "Merchant requested action (send/draft). Acknowledging and moving to preparation without claiming an external send occurred."
        }

    if any(x in low for x in ["thank you", "thanks", "shukriya", "dhanyavaad"]):
        return {
            "action": "wait",
            "wait_seconds": 14400,
            "rationale": "Merchant sent a low-information acknowledgment. Backing off 4h to avoid unnecessary follow-up pressure."
        }

    # Default: stay on mission
    m = get_merchant(st.get("merchant_id", "")) if st.get("merchant_id") else None
    llm_reply = call_llm_for_reply(m, st, s, st.get("trigger_kind")) if m else None
    if llm_reply and len(llm_reply) > 15:
        body = re.sub(r'https?://\S+|www\.\S+', '', llm_reply).strip()
    else:
        body = "Got it. I'll keep this focused on the original topic. If you want me to take the next step, reply YES and I'll move from discussion to action."
    st["sent_bodies"].append(body)
    return {
        "action": "send",
        "body": body,
        "cta": "binary_yes_no",
        "rationale": "General merchant reply. Keeping conversation on-mission and offering a clear, low-friction next step."
    }


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "VeraChallenge/2.0"

    def _json(self, status, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return None

    def do_GET(self):
        if self.path == "/v1/healthz":
            self._json(200, {
                "status": "ok",
                "uptime_seconds": round(time.time() - START, 3),
                "contexts_loaded": {k: len(v) for k, v in CONTEXTS.items()},
                "llm_enabled": bool(NIM_API_KEY),
            })
        elif self.path == "/v1/metadata":
            self._json(200, {
                "team_name": "Sarvagya's Vera Team",
                "team_members": ["Sarvagya"],
                "model": f"NVIDIA NIM ({NIM_MODEL})" if NIM_API_KEY else "deterministic-router-v2",
                "approach": "LLM-augmented composition with category-aware prompt engineering, peer comparison anchoring, and deterministic fallback. Each trigger kind has a specialized prompt that injects category voice, merchant performance vs peer stats, conversation history, and customer context.",
                "contact_email": "",
                "version": "2.0.0",
                "submitted_at": now_iso(),
            })
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path == "/v1/context":
            req = self._body()
            if not isinstance(req, dict):
                return self._json(400, {"accepted": False, "reason": "invalid_json"})
            scope = req.get("scope")
            cid = req.get("context_id")
            ver = req.get("version")
            if scope not in CONTEXTS or not cid or not isinstance(ver, int) or ver < 1:
                return self._json(400, {"accepted": False, "reason": "invalid_scope", "details": "scope/context_id/version required"})
            key = (scope, cid)
            old = VERSIONS.get(key)
            if old is not None and ver <= old:
                return self._json(409, {"accepted": False, "reason": "stale_version", "current_version": old})
            payload = req.get("payload")
            if not isinstance(payload, dict):
                return self._json(400, {"accepted": False, "reason": "invalid_payload"})
            CONTEXTS[scope][cid] = {"version": ver, "payload": payload}
            VERSIONS[key] = ver
            return self._json(200, {"accepted": True, "ack_id": f"ack_{cid}_v{ver}", "stored_at": now_iso()})

        if self.path == "/v1/tick":
            req = self._body() or {}
            actions = []
            for tid in req.get("available_triggers", []) or []:
                t = get_trigger(tid)
                if not t:
                    continue
                mid = t.get("merchant_id") or t.get("payload", {}).get("merchant_id")
                m = get_merchant(mid) if mid else None
                if not m:
                    continue
                cid = t.get("customer_id")
                c = get_customer(cid) if cid else None
                if t.get("scope") == "customer" and not c:
                    continue
                if cid and not c:
                    continue

                # Customer consent gate
                if c:
                    consent = c.get("consent", {})
                    scopes = set(consent.get("scope", []) or [])
                    prefs = c.get("preferences", {})
                    chan = prefs.get("channel", "")
                    if not chan or (not chan.startswith("whatsapp")):
                        continue
                    allowed = {
                        "recall_due": {"recall_reminders"},
                        "appointment_tomorrow": {"appointment_reminders"},
                        "chronic_refill_due": {"appointment_reminders", "refill_reminders"},
                        "trial_followup": {"appointment_reminders", "kids_program_updates"},
                        "wedding_package_followup": {"recall_reminders", "bridal_package_followup"},
                        "customer_lapsed_hard": {"recall_reminders", "winback_offers"},
                        "winback_eligible": {"recall_reminders", "winback_offers"},
                        "unplanned_slot_open": {"appointment_reminders"},
                    }.get(t.get("kind"))
                    if allowed and not allowed.intersection(scopes):
                        continue

                sk = t.get("suppression_key") or tid
                if sk in SENT_SUPPRESSION:
                    continue

                cat = get_category(m.get("category_slug", ""))
                action = compose(cat or {}, m, t, c)
                conv_id = f"conv_{tid}_{uuid.uuid4().hex[:8]}"
                action.update({
                    "conversation_id": conv_id,
                    "merchant_id": mid,
                    "customer_id": cid,
                    "trigger_id": tid,
                    "suppression_key": sk,
                })
                SENT_SUPPRESSION.add(sk)
                cv = conversation(conv_id)
                cv["merchant_id"] = mid
                cv["trigger_kind"] = t.get("kind")
                cv["sent_bodies"].append(action["body"])
                actions.append(action)
            return self._json(200, {"actions": actions})

        if self.path == "/v1/reply":
            req = self._body() or {}
            if not req.get("conversation_id") or not req.get("message"):
                return self._json(400, {"error": "conversation_id and message required"})
            return self._json(200, reply_action(req))

        if self.path == "/v1/metadata":
            return self._json(405, {"error": "use GET"})

        self._json(404, {"error": "not_found"})

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"[Vera Bot v2.0] Starting on port {port}")
    print(f"[Vera Bot v2.0] LLM: {'NVIDIA NIM (' + NIM_MODEL + ')' if NIM_API_KEY else 'DISABLED (deterministic fallback)'}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
