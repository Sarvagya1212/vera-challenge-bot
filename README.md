# Vera Challenge Submission

## Approach
This submission implements the challenge contract as a dependency-free Python HTTP service. It uses a **context-first, deterministic router** rather than making every message depend on an external LLM. This keeps `/v1/context`, `/v1/tick`, and `/v1/reply` fast and reproducible and makes the no-fabrication constraint auditable.

The composer consumes the four pushed contexts:
- Category: voice, digest, peer signals and category-specific knowledge.
- Merchant: identity, performance, active offers, signals and customer aggregate.
- Trigger: the immediate "why now" event and suppression key.
- Customer: only when a customer-scoped trigger is supplied.

Trigger-specific composition routes cover the representative trigger families in the supplied dataset: research/compliance, performance, renewal, festival, reviews, milestones, planning, seasonal demand, GBP, competitors, CDE, IPL, supply/refill, recall/winback and trial/bridal follow-up.

## Safety and reliability
- Context versions are monotonic; same/older versions return `409 stale_version`; newer versions replace atomically.
- Customer messages require WhatsApp-compatible channel/consent scope when a relevant consent scope exists.
- Customer messages use `merchant_on_behalf`; merchant messages use `vera`.
- Suppression keys prevent duplicate proactive sends across ticks.
- URLs are stripped from generated bodies because they add no necessary value for the challenge and can trigger the hard URL penalty.
- No external data is fetched and no unsupported merchant/customer facts are invented.
- Reply state detects explicit opt-out, repeated canned auto-replies, and clear intent transitions.
- Conversation state is in memory because the challenge allows it; production would use durable Redis/DB state.

## Tradeoffs
The main tradeoff is choosing deterministic composition over an LLM in the hot path. This sacrifices some stylistic creativity, but gains predictable latency, lower operational risk, stronger provenance and protection against hallucinated offers/research facts. The architecture leaves the composition boundary isolated so a constrained LLM can be added later for wording while retaining deterministic retrieval, validation and safety gates.

## What additional context would help most
For a production version, the highest-value additions would be:
1. Verified appointment/inventory availability at send time.
2. Merchant-approved offer/catalog source of truth.
3. Customer consent timestamps and granular outreach scopes.
4. Conversation-level 24-hour session state and delivery status.
5. Fresh peer benchmarks by locality/category.
