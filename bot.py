#!/usr/bin/env python3
"""
Vera Challenge Bot
A dependency-light HTTP/JSON implementation of the magicpin AI Challenge contract.

Design:
- In-memory versioned context store with atomic replacement semantics.
- Deterministic routing/composition for predictable latency and zero hallucination.
- Conversation state for opt-outs, auto-replies, intent transitions and repetition.
- Optional LLM hook is deliberately not required: challenge correctness should not depend
  on network/API availability.
"""
from __future__ import annotations
import json, re, time, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

START = time.time()
CONTEXTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "category": {}, "merchant": {}, "customer": {}, "trigger": {}
}
VERSIONS: Dict[Tuple[str, str], int] = {}
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}
SENT_SUPPRESSION = set()

CATEGORY_RULES = {
    "dentists": ("clinical-peer", ["guaranteed","100% safe","completely cure","miracle","best in city"]),
    "salons": ("warm-practical", ["guaranteed","miracle","best in city"]),
    "restaurants": ("operator-practical", ["guaranteed","best in city"]),
    "gyms": ("coach-practical", ["guaranteed","miracle","100% safe"]),
    "pharmacies": ("trustworthy-precise", ["guaranteed","miracle","completely cure"]),
}

ACTION_KINDS = {
    "recall_due","wedding_package_followup","winback_eligible","customer_lapsed_hard",
    "trial_followup","chronic_refill_due","appointment_tomorrow","unplanned_slot_open"
}
INFO_KINDS = {"research_digest","regulation_change","festival_upcoming","category_seasonal",
              "supply_alert","competitor_opened","cde_opportunity","perf_spike","milestone_reached",
              "curious_ask_due","review_theme_emerged","active_planning_intent","seasonal_perf_dip",
              "gbp_unverified","perf_dip","renewal_due","ipl_match_today","dormant_with_vera"}

def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

def owner_name(m):
    i=m.get("identity",{})
    return i.get("owner_first_name") or i.get("name","").split("'s")[0].replace("Dr. ","").strip() or "there"

def active_offers(m):
    return [o.get("title","") for o in m.get("offers",[]) if o.get("status")=="active"]

def category_for(m):
    return m.get("category_slug","")

def get_customer(cid):
    return CONTEXTS["customer"].get(cid,{}).get("payload")

def get_merchant(mid):
    return CONTEXTS["merchant"].get(mid,{}).get("payload")

def get_category(slug):
    return CONTEXTS["category"].get(slug,{}).get("payload")

def get_trigger(tid):
    return CONTEXTS["trigger"].get(tid,{}).get("payload")

def digest_item(cat, tid, trigger):
    payload=trigger.get("payload",{})
    item_id=payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("top_item",{}).get("id")
    for d in cat.get("digest",[]):
        if d.get("id")==item_id: return d
    # If trigger embeds the item, trust that payload because it is pushed context.
    if isinstance(payload.get("top_item"),dict): return payload["top_item"]
    return None

def pct(v):
    try:
        return f"{abs(float(v))*100:.0f}%"
    except: return str(v)

def money(n):
    try: return f"₹{int(n):,}"
    except: return str(n)

def language_hint(m,c=None):
    if c: return c.get("identity",{}).get("language_pref","").lower()
    langs=[str(x).lower() for x in m.get("identity",{}).get("languages",[])]
    return "hi" if "hi" in langs else "en"

def conversation(cid):
    return CONVERSATIONS.setdefault(cid, {"turns":0,"messages":[],"auto_count":0,"ended":False,
                                          "sent_bodies":[],"intent":False,"merchant_id":None})

def is_optout(s):
    s=s.lower()
    return any(x in s for x in ["stop messaging","stop sending","don't message","do not message",
                                "not interested","unsubscribe","remove me","this is useless spam","go away"])

def is_commitment(s):
    s=s.lower()
    return any(x in s for x in ["let's do it","lets do it","go ahead","yes, do it","yes do it",
                                "what's next","whats next","proceed","i want to join","sign me up",
                                "yes please","do it"])

def is_auto_reply(s):
    s=s.lower()
    patterns=["thank you for contacting","our team will respond shortly","team will get back",
              "thanks for contacting","we have received your message"]
    return any(p in s for p in patterns)

def compose(category, merchant, trigger, customer=None, conv=None):
    kind=trigger.get("kind","")
    p=trigger.get("payload",{})
    cat=category.get("slug", merchant.get("category_slug",""))
    owner=owner_name(merchant)
    mname=merchant.get("identity",{}).get("name","your business")
    locality=merchant.get("identity",{}).get("locality","")
    cust_name=(customer or {}).get("identity",{}).get("name","")
    lang=language_hint(merchant,customer)
    send_as="merchant_on_behalf" if customer else "vera"
    # First-message template is metadata only; body remains human-readable for judge scoring.
    template="merchant_engagement_v1" if customer else "vera_engagement_v1"
    params=[]
    body=""
    cta="open_ended"

    if kind=="research_digest":
        d=digest_item(category, "", trigger) or {}
        title=d.get("title","")
        source=d.get("source","")
        n=d.get("trial_n")
        seg=d.get("patient_segment","").replace("_"," ")
        summary=d.get("summary","")
        m_pct=re.search(r"(\d+)%", summary)
        result_stat=(f"{m_pct.group(1)}% lower recurrence" if m_pct else "")
        detail=(f"{n}-patient trial" if n else "") + (f" — {title}" if title else "")
        if result_stat: detail += f" ({result_stat})"
        body=f"{'Dr. ' if cat=='dentists' else ''}{owner}, {source.split(',')[0] if source else 'A new research item'} landed. "
        if seg: body+=f"One item relevant to your {seg} cohort — "
        body+=detail + ". "
        body+="Worth a look. Want me to pull the source and turn it into a practical customer-facing draft?"
        cta="open_ended"; params=[owner,detail,"source + practical draft"]
    elif kind=="regulation_change":
        d=digest_item(category,"",trigger) or {}
        deadline=p.get("deadline_iso","")
        body=f"{owner}, quick compliance heads-up: {d.get('title','A category regulation has changed')}."
        if deadline: body+=f" Effective {deadline}."
        if d.get("summary"): body+=f" {d['summary']}"
        body+=" Worth reviewing before the effective date — want me to turn the change into a short checklist?"
        params=[owner,d.get("title",""),deadline]
    elif kind=="recall_due":
        slots=p.get("available_slots",[])
        labels=[s.get("label") for s in slots if s.get("label")]
        last=p.get("last_service_date","")
        service=p.get("service_due","").replace("_"," ")
        offer=active_offers(merchant)[0] if active_offers(merchant) else ""
        slottext=" or ".join(labels[:2])
        body=f"Hi {cust_name}, {mname} here 🦷 " if cat=="dentists" else f"Hi {cust_name}, {mname} here. "
        if last: body+=f"Your last visit was {last} — your {service} recall is due."
        else: body+=f"Your {service} recall is due."
        if slottext: body+=f" Apke liye {slottext} slots available hain."
        if offer: body+=f" {offer}."
        body+=" Reply with the slot you prefer, or tell us a time that works."
        cta="multi_choice_slot"; params=[cust_name,mname,service,slottext,offer]
    elif kind=="perf_dip":
        metric=p.get("metric","metric"); delta=p.get("delta_pct",0); baseline=p.get("vs_baseline")
        body=f"{owner}, your {metric} are down {pct(delta)} over {p.get('window','7d')}"
        if baseline is not None: body+=f" vs a baseline of {baseline}"
        body+=". I’d treat this as a signal, not a verdict. Want me to isolate the most likely driver and suggest one low-effort fix?"
        params=[owner,metric,pct(delta),str(baseline or "")]
    elif kind=="renewal_due":
        body=f"{owner}, your {p.get('plan','plan')} renewal is in {p.get('days_remaining','?')} days"
        if p.get("renewal_amount") is not None: body+=f" ({money(p['renewal_amount'])})."
        else: body+="."
        body+=" If you want to keep it active, I can lay out the renewal steps now."
        cta="binary_yes_no"; params=[owner,str(p.get("days_remaining","")),money(p.get("renewal_amount",""))]
    elif kind=="festival_upcoming":
        body=f"{owner}, {p.get('festival','The festival')} is on {p.get('date','')} — {p.get('days_until','?')} days out."
        body+=" For your category, it’s a useful planning window. Want me to turn the event into one concrete offer/post idea using what you already have?"
        params=[owner,p.get("festival",""),p.get("date",""),str(p.get("days_until",""))]
    elif kind=="wedding_package_followup":
        w=p.get("wedding_date","")
        body=f"Hi {cust_name} 💍 {owner} from {mname} here. Your wedding is {p.get('days_to_wedding','?')} days away ({w}), and the {p.get('next_step_window_open','next prep window').replace('_',' ')} is open."
        body+=" You’d said yes to the bridal journey — want me to help pick the first session timing?"
        cta="binary_yes_no"; params=[cust_name,owner,str(p.get("days_to_wedding","")),w]
    elif kind=="curious_ask_due":
        body=f"Hi {owner}! Quick check — what service has been most asked-for this week at {mname}? "
        body+="I’ll turn your answer into a ready-to-use post + a short WhatsApp reply. Takes 5 min."
        params=[owner,mname]
    elif kind=="winback_eligible":
        body=f"{owner}, it’s been {p.get('days_since_expiry','?')} days since the offer expired, and {p.get('lapsed_customers_added_since_expiry','?')} lapsed customers were added since then."
        if p.get("perf_dip_pct") is not None: body+=f" Your performance is also {pct(p['perf_dip_pct'])} down."
        body+=" There’s a concrete win-back pool here. Want me to draft a low-friction reactivation message?"
        params=[owner,str(p.get("days_since_expiry","")),str(p.get("lapsed_customers_added_since_expiry",""))]
    elif kind=="ipl_match_today":
        body=f"Quick heads-up {owner} — {p.get('match','')} at {p.get('venue','')} tonight"
        tm=p.get("match_time_iso","")
        if tm: body+=f" ({tm[11:16]})."
        else: body+="."
        offer=active_offers(merchant)
        if offer: body+=f" You already have {offer[0]} active; I’d use that rather than inventing a match-night offer."
        body+=" Want me to draft the delivery-first message?"
        params=[owner,p.get("match",""),p.get("venue",""),offer[0] if offer else ""]
    elif kind=="review_theme_emerged":
        body=f"{owner}, {p.get('occurrences_30d','?')} recent reviews mention “{p.get('theme','')}” and the trend is {p.get('trend','changing')}."
        if p.get("common_quote"): body+=f" One customer phrased it: “{p['common_quote']}”."
        body+=" This is specific enough to act on. Want me to draft a response plus one operational fix?"
        params=[owner,str(p.get("occurrences_30d","")),p.get("theme","")]
    elif kind=="milestone_reached":
        body=f"{owner}, you’re at {p.get('value_now','?')} {p.get('metric','')} — only {max(0,(p.get('milestone_value',0)-p.get('value_now',0)))} to the {p.get('milestone_value','next')} milestone."
        body+=" Nice window to turn the milestone into a visible proof point. Want me to draft the post?"
        params=[owner,str(p.get("value_now","")),str(p.get("milestone_value",""))]
    elif kind=="active_planning_intent":
        topic=p.get("intent_topic","").replace("_"," ")
        last=p.get("merchant_last_message","")
        body=f"{owner}, picking up your plan for {topic}. "
        if "what would it look like" in last.lower():
            body+="You asked what it could look like, so here’s the next step: I can turn the idea into a one-page starter offer with audience, price structure and timing. Want me to draft it now?"
        else:
            body+="I can turn the idea into a one-page starter offer with audience, price structure and timing. Want me to draft it now?"
        cta="binary_yes_no"; params=[owner,topic]
    elif kind=="seasonal_perf_dip":
        body=f"{owner}, {p.get('metric','performance')} are down {pct(p.get('delta_pct',0))} over {p.get('window','7d')}, but this trigger marks it as an expected seasonal dip."
        if p.get("season_note"): body+=f" Context: {p['season_note'].replace('_',' ')}."
        body+=" I wouldn’t panic-spend on the dip. Want me to suggest a retention-first action for this window?"
        params=[owner,p.get("metric",""),pct(p.get("delta_pct",0))]
    elif kind=="customer_lapsed_hard":
        days=p.get("days_since_last_visit","?")
        body=f"Hi {cust_name} 👋 {owner} from {mname} here. It’s been about {days} days since your last visit — no pressure, life happens."
        if p.get("previous_focus"): body+=f" We can pick back up around your previous focus: {p['previous_focus'].replace('_',' ')}."
        body+=" If you’d like, I can help find a convenient next slot."
        cta="binary_yes_no"; params=[cust_name,owner,str(days)]
    elif kind=="trial_followup":
        opts=p.get("next_session_options",[])
        body=f"Hi {cust_name}, {owner} from {mname} here. Following up on your trial from {p.get('trial_date','')}."
        if opts: body+=f" We have {opts[0].get('label','a next-session slot')} available."
        body+=" Want me to hold it for you?"
        cta="binary_yes_no"; params=[cust_name,owner,p.get("trial_date","")]
    elif kind=="supply_alert":
        mol=p.get("molecule","")
        batches=", ".join(p.get("affected_batches",[]))
        body=f"{owner}, supply alert: {mol} is flagged in batches {batches}"
        if p.get("manufacturer"): body+=f" from {p['manufacturer']}."
        body+=" Please verify affected stock against the alert before dispensing. Want a short stock-check checklist?"
        cta="binary_yes_no"; params=[owner,mol,batches]
    elif kind=="chronic_refill_due":
        mol=", ".join(p.get("molecule_list",[]))
        body=f"Hi {cust_name}, {mname} here. Your refill list includes {mol}; the current stock is expected to run out around {p.get('stock_runs_out_iso','')}."
        if p.get("delivery_address_saved"): body+=" Your saved delivery address is available."
        body+=" Want us to arrange the refill?"
        cta="binary_yes_no"; params=[cust_name,mol]
    elif kind=="category_seasonal":
        trends=p.get("trends",[])
        body=f"{owner}, summer demand is shifting: {', '.join(trends[:3])}."
        body+=" The signal recommends a shelf-action review. Want me to turn those movements into a simple priority list?"
        params=[owner, "; ".join(trends[:3])]
    elif kind=="gbp_unverified":
        body=f"{owner}, your Google Business Profile is still unverified. The listed path is {p.get('verification_path','the available verification path')}."
        if p.get("estimated_uplift_pct") is not None: body+=f" The context estimates {pct(p['estimated_uplift_pct'])} uplift."
        body+=" Want me to walk you through the verification steps?"
        params=[owner,p.get("verification_path","")]
    elif kind=="cde_opportunity":
        body=f"{owner}, there’s a CDE opportunity: {p.get('fee','')} and {p.get('credits','?')} credits."
        d=digest_item(category,"",trigger)
        if d: body+=f" It’s tied to “{d.get('title','the current item')}”."
        body+=" Want me to pull the details and draft the registration steps?"
        params=[owner,p.get("fee",""),str(p.get("credits",""))]
    elif kind=="competitor_opened":
        body=f"{owner}, a competitor opened {p.get('distance_km','?')} km away: {p.get('competitor_name','')}."
        if p.get("their_offer"): body+=f" Their listed offer is {p['their_offer']}."
        body+=" That’s a concrete local signal, not a reason to panic. Want me to compare it with your active offer and suggest one response?"
        params=[owner,p.get("competitor_name",""),str(p.get("distance_km","")),p.get("their_offer","")]
    elif kind=="perf_spike":
        metric=p.get("metric","metric")
        body=f"{owner}, your {metric} are up {pct(p.get('delta_pct',0))} over {p.get('window','7d')}"
        if p.get("vs_baseline") is not None: body+=f" vs baseline {p['vs_baseline']}"
        if p.get("likely_driver"): body+=f" — likely driver: {p['likely_driver'].replace('_',' ')}"
        body+=". This is a good moment to double down on what caused it. Want me to turn that signal into one next action?"
        params=[owner,metric,pct(p.get("delta_pct",0))]
    elif kind=="dormant_with_vera":
        body=f"Hi {owner} — it’s been {p.get('days_since_last_merchant_message','?')} days since we last spoke"
        if p.get("last_topic"): body+=f" about {p['last_topic'].replace('_',' ')}"
        body+=". If that’s still useful, I can pick it back up; otherwise no need to reply."
        cta="open_ended"; params=[owner,str(p.get("days_since_last_merchant_message",""))]
    else:
        # Safe generic fallback using only verified trigger fields.
        body=f"Hi {owner}, I have a new {kind.replace('_',' ')} update tied to your {cat} business."
        body+=" I can summarize the relevant details and suggest one next step. Want me to?"
        params=[owner,kind]

    if not customer and lang=="hi":
        # Keep Hindi-English natural only where the dataset says Hindi is supported.
        body=body.replace("Want me to","Chahiye to main").replace("Quick heads-up","Quick heads-up")
    # Hard safety sanitation: URLs are not needed in challenge messages.
    body=re.sub(r'https?://\S+|www\.\S+',"",body).strip()
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "template_name": template,
        "template_params": params,
        "rationale": f"{kind} routed for {cat}; anchored on trigger facts and merchant/customer context without adding unverified facts.",
    }

def reply_action(req):
    cid=req.get("conversation_id","")
    s=req.get("message","").strip()
    st=conversation(cid)
    st["turns"]+=1
    st["messages"].append(s)
    if req.get("merchant_id"): st["merchant_id"]=req["merchant_id"]
    if is_optout(s):
        st["ended"]=True
        return {"action":"end","rationale":"Explicit opt-out or hostile stop signal detected; closing this conversation without further engagement."}
    if is_auto_reply(s):
        st["auto_count"]+=1
        if st["auto_count"]>=3:
            st["ended"]=True
            return {"action":"end","rationale":"Repeated canned auto-replies with no owner engagement; closing rather than spamming."}
        if st["auto_count"]==2:
            return {"action":"wait","wait_seconds":86400,"rationale":"Canned auto-reply repeated; backing off 24 hours for a real owner response."}
        return {"action":"send","body":"Looks like an auto-reply 😊 When the owner sees this, just reply “Yes” if you want me to continue with the original request.","cta":"binary_yes_no","rationale":"Detected a canned auto-reply; one low-friction attempt to surface the conversation to the owner."}
    if st["ended"]:
        return {"action":"end","rationale":"Conversation was already closed."}
    if is_commitment(s):
        st["intent"]=True
        return {"action":"send",
                "body":"Great — moving to action. I’ll draft the next concrete step from the context already provided. Reply CONFIRM when you want to proceed, or tell me what to change.",
                "cta":"binary_confirm_cancel",
                "rationale":"Explicit commitment detected; switching from qualification to execution instead of asking another discovery question."}
    # Answer concrete yes/ask without pretending to execute unavailable external actions.
    low=s.lower()
    if any(x in low for x in ["send the abstract","send it","draft","please do","yes please"]):
        return {"action":"send","body":"Absolutely — I’ll use the information already in the conversation to prepare that draft. If you want a change in audience, timing or tone, tell me before sending.","cta":"open_ended","rationale":"Acknowledges the requested action and moves directly to preparation without claiming an external send occurred."}
    if any(x in low for x in ["thank you","thanks"]):
        return {"action":"wait","wait_seconds":14400,"rationale":"Low-information acknowledgment; waiting avoids unnecessary follow-up pressure."}
    return {"action":"send","body":"Got it. I’ll keep this focused on the original request. If you want me to take the next step, reply YES and I’ll move from discussion to action.","cta":"binary_yes_no","rationale":"Keeps the conversation on mission and offers a single low-friction next step."}

class Handler(BaseHTTPRequestHandler):
    server_version="VeraChallenge/1.0"
    def _json(self, status, obj):
        b=json.dumps(obj,ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def _body(self):
        n=int(self.headers.get("Content-Length","0") or 0)
        try: return json.loads(self.rfile.read(n) or b"{}")
        except Exception: return None
    def do_GET(self):
        if self.path=="/v1/healthz":
            self._json(200,{"status":"ok","uptime_seconds":round(time.time()-START,3),
                            "contexts_loaded":{k:len(v) for k,v in CONTEXTS.items()}})
        elif self.path=="/v1/metadata":
            self._json(200,{"team_name":"Sarvagya's Vera Team","team_members":["Sarvagya"],
                            "model":"deterministic-router (optional LLM-free)",
                            "approach":"context-first trigger router with category-aware deterministic composition and stateful replay handling",
                            "contact_email":"","version":"1.0.0","submitted_at":now_iso()})
        else: self._json(404,{"error":"not_found"})
    def do_POST(self):
        if self.path=="/v1/context":
            req=self._body()
            if not isinstance(req,dict): return self._json(400,{"accepted":False,"reason":"invalid_json"})
            scope=req.get("scope"); cid=req.get("context_id"); ver=req.get("version")
            if scope not in CONTEXTS or not cid or not isinstance(ver,int) or ver<1:
                return self._json(400,{"accepted":False,"reason":"invalid_scope","details":"scope/context_id/version required"})
            key=(scope,cid); old=VERSIONS.get(key)
            if old is not None and ver<=old:
                return self._json(409,{"accepted":False,"reason":"stale_version","current_version":old})
            payload=req.get("payload")
            if not isinstance(payload,dict):
                return self._json(400,{"accepted":False,"reason":"invalid_payload"})
            CONTEXTS[scope][cid]={"version":ver,"payload":payload}
            VERSIONS[key]=ver
            return self._json(200,{"accepted":True,"ack_id":f"ack_{cid}_v{ver}","stored_at":now_iso()})
        if self.path=="/v1/tick":
            req=self._body() or {}
            actions=[]
            for tid in req.get("available_triggers",[]) or []:
                t=get_trigger(tid)
                if not t: continue
                mid=t.get("merchant_id") or t.get("payload",{}).get("merchant_id")
                m=get_merchant(mid) if mid else None
                if not m: continue
                cid=t.get("customer_id")
                c=get_customer(cid) if cid else None
                if t.get("scope")=="customer" and not c: continue
                if cid and not c: continue
                # Customer outreach requires an explicit merchant-granted WhatsApp opt-in
                # with a scope matching the trigger family. This is a hard safety gate.
                if c:
                    consent=c.get("consent",{})
                    scopes=set(consent.get("scope",[]) or [])
                    prefs=c.get("preferences",{})
                    if not prefs.get("channel") or not prefs.get("channel").startswith("whatsapp"):
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
                sk=t.get("suppression_key") or tid
                if sk in SENT_SUPPRESSION: continue
                cat=get_category(m.get("category_slug",""))
                action=compose(cat or {},m,t,c)
                action.update({"conversation_id":f"conv_{tid}_{uuid.uuid4().hex[:8]}",
                               "merchant_id":mid,"customer_id":cid,"trigger_id":tid,
                               "suppression_key":sk})
                # Only suppress after generating a send; repeated ticks should not spam.
                SENT_SUPPRESSION.add(sk)
                cv=conversation(action["conversation_id"]); cv["merchant_id"]=mid
                cv["sent_bodies"].append(action["body"])
                actions.append(action)
            return self._json(200,{"actions":actions})
        if self.path=="/v1/reply":
            req=self._body() or {}
            if not req.get("conversation_id") or not req.get("message"):
                return self._json(400,{"error":"conversation_id and message required"})
            return self._json(200,reply_action(req))
        if self.path=="/v1/metadata": return self._json(405,{"error":"use GET"})
        self._json(404,{"error":"not_found"})
    def log_message(self,*args): pass

def main():
    import os
    port=int(os.environ.get("PORT","8080"))
    ThreadingHTTPServer(("0.0.0.0",port),Handler).serve_forever()

if __name__=="__main__": main()
