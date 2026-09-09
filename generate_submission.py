#!/usr/bin/env python3
"""
Generate improved submission.jsonl by running the bot against all 25 triggers.
Uses the bot's compose function directly (deterministic fallback since no LLM API in script).
"""
import json, pathlib, sys, io

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = pathlib.Path(__file__).parent

# Import compose and helpers from bot
sys.path.insert(0, str(ROOT))
from bot import compose, CONTEXTS, VERSIONS, get_category, get_merchant, get_customer, get_trigger, customer_name

def load_dataset():
    """Load all dataset files into the CONTEXTS store."""
    # Load categories
    for f in (ROOT / "dataset" / "categories").glob("*.json"):
        d = json.load(open(f))
        slug = d.get("slug", f.stem)
        CONTEXTS["category"][slug] = {"version": 1, "payload": d}
        VERSIONS[("category", slug)] = 1

    # Load merchants
    data = json.load(open(ROOT / "dataset" / "merchants_seed.json"))
    for d in data["merchants"]:
        mid = d["merchant_id"]
        CONTEXTS["merchant"][mid] = {"version": 1, "payload": d}
        VERSIONS[("merchant", mid)] = 1

    # Load customers
    data = json.load(open(ROOT / "dataset" / "customers_seed.json"))
    for d in data["customers"]:
        cid = d["customer_id"]
        CONTEXTS["customer"][cid] = {"version": 1, "payload": d}
        VERSIONS[("customer", cid)] = 1

    # Load triggers
    data = json.load(open(ROOT / "dataset" / "triggers_seed.json"))
    for d in data["triggers"]:
        tid = d["id"]
        CONTEXTS["trigger"][tid] = {"version": 1, "payload": d}
        VERSIONS[("trigger", tid)] = 1

    print(f"Loaded: {len(CONTEXTS['category'])} categories, {len(CONTEXTS['merchant'])} merchants, "
          f"{len(CONTEXTS['customer'])} customers, {len(CONTEXTS['trigger'])} triggers")


def generate_submission():
    """Generate all 30 test pair messages (25 base + 5 additional)."""
    load_dataset()
    
    triggers = json.load(open(ROOT / "dataset" / "triggers_seed.json"))["triggers"]
    results = []
    
    # First 25: one per trigger in the seed
    for i, t in enumerate(triggers):
        tid = t["id"]
        mid = t.get("merchant_id") or t.get("payload", {}).get("merchant_id")
        cid = t.get("customer_id")
        
        merchant = get_merchant(mid) if mid else None
        if not merchant:
            print(f"  SKIP T{i+1:02d}: no merchant {mid}")
            continue
            
        cat_slug = merchant.get("category_slug", "")
        category = get_category(cat_slug) or {}
        customer = get_customer(cid) if cid else None
        
        result = compose(category, merchant, t, customer)
        
        test_id = f"T{i+1:02d}"
        entry = {
            "test_id": test_id,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": t.get("suppression_key", tid),
            "rationale": result["rationale"],
        }
        results.append(entry)
        
        # Preview
        body_preview = result["body"][:80].replace("\n", " ")
        print(f"  {test_id} [{t['kind']:25s}] {body_preview}...")
    
    # Generate 5 additional cross-context test pairs (T26-T30)
    # These reuse triggers but with different customer/merchant pairings
    extra_pairs = [
        # T26: recall_due for Rohit (different customer same trigger shape as T03)
        {"trigger_idx": 2, "customer_override": "c_002_rohit_for_m001"},
        # T27: recall_due for Aanya (child patient)
        {"trigger_idx": 2, "customer_override": "c_003_aanya_for_m001"},
        # T28: wedding follow-up for Sneha (different bridal customer)
        {"trigger_idx": 6, "customer_override": "c_004_sneha_for_m003"},
        # T29: customer lapsed hard — different gym customer
        {"trigger_idx": 14, "customer_override": "c_009_arjun_for_m007"},
        # T30: trial followup — different yoga student
        {"trigger_idx": 16, "customer_override": "c_011_sumitra_for_m008"},
    ]
    
    for j, pair in enumerate(extra_pairs):
        test_id = f"T{26+j:02d}"
        t = triggers[pair["trigger_idx"]]
        mid = t.get("merchant_id") or t.get("payload", {}).get("merchant_id")
        cid_override = pair["customer_override"]
        
        merchant = get_merchant(mid) if mid else None
        if not merchant:
            continue
        
        cat_slug = merchant.get("category_slug", "")
        category = get_category(cat_slug) or {}
        customer = get_customer(cid_override)
        
        if not customer:
            print(f"  SKIP {test_id}: no customer {cid_override}")
            continue
        
        result = compose(category, merchant, t, customer)
        
        # Fix send_as for customer-facing messages
        send_as = "merchant_on_behalf" if customer else "vera"
        
        entry = {
            "test_id": test_id,
            "body": result["body"],
            "cta": result["cta"],
            "send_as": send_as,
            "suppression_key": f"{t.get('suppression_key', '')}:{cid_override}",
            "rationale": result["rationale"],
        }
        results.append(entry)
        
        cn = customer_name(customer)
        body_preview = result["body"][:80].replace("\n", " ")
        print(f"  {test_id} [{t['kind']:25s}] (customer: {cn}) {body_preview}...")
    
    # Write output
    output_path = ROOT / "submission.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\nWrote {len(results)} entries to {output_path}")
    return results


if __name__ == "__main__":
    generate_submission()
