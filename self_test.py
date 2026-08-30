import json, pathlib, subprocess, sys, time, os, requests

ROOT=pathlib.Path(__file__).parent
BASE="http://127.0.0.1:18082"
proc=subprocess.Popen([sys.executable,str(ROOT/"bot.py")],cwd=ROOT,env={**os.environ,"PORT":"18082"})
try:
    for _ in range(30):
        try:
            if requests.get(BASE+"/v1/healthz",timeout=0.2).ok: break
        except: time.sleep(.2)
    assert requests.get(BASE+"/v1/healthz").json()["contexts_loaded"] == {"category":0,"merchant":0,"customer":0,"trigger":0}

    cats={}
    for f in (ROOT/"dataset/categories").glob("*.json"):
        d=json.load(open(f)); cats[d["slug"]]=d
        assert requests.post(BASE+"/v1/context",json={"scope":"category","context_id":d["slug"],"version":1,"payload":d}).status_code==200

    data=json.load(open(ROOT/"dataset/merchants_seed.json"))["merchants"]
    for d in data:
        assert requests.post(BASE+"/v1/context",json={"scope":"merchant","context_id":d["merchant_id"],"version":1,"payload":d}).status_code==200
    customers=json.load(open(ROOT/"dataset/customers_seed.json"))["customers"]
    for d in customers:
        assert requests.post(BASE+"/v1/context",json={"scope":"customer","context_id":d["customer_id"],"version":1,"payload":d}).status_code==200

    trigs=json.load(open(ROOT/"dataset/triggers_seed.json"))["triggers"]
    for d in trigs:
        assert requests.post(BASE+"/v1/context",json={"scope":"trigger","context_id":d["id"],"version":1,"payload":d}).status_code==200

    counts=requests.get(BASE+"/v1/healthz").json()["contexts_loaded"]
    assert counts=={"category":5,"merchant":10,"customer":15,"trigger":25}, counts

    actions=[]
    for i in range(0,len(trigs),5):
        r=requests.post(BASE+"/v1/tick",json={"now":"2026-04-26T10:35:00Z","available_triggers":[x["id"] for x in trigs[i:i+5]]})
        assert r.ok, r.text
        actions.extend(r.json()["actions"])
    assert len(actions)==25
    for a in actions:
        required=["conversation_id","merchant_id","customer_id","send_as","trigger_id","template_name","template_params","body","cta","suppression_key","rationale"]
        assert all(k in a for k in required), a
        assert not ("http://" in a["body"] or "https://" in a["body"])
    # idempotency
    first=trigs[0]
    r=requests.post(BASE+"/v1/context",json={"scope":"trigger","context_id":first["id"],"version":1,"payload":first})
    assert r.status_code==409
    print("PASS: contract, counts, 25 trigger compositions, URL safety, and version idempotency")
finally:
    proc.terminate()
    proc.wait(timeout=2)
