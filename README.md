# Vera Challenge Submission

## Approach
This submission implements a hybrid **LLM-Augmented Composition Engine** with a highly capable deterministic fallback. 

The architecture is designed to handle both the speed/determinism constraints of the challenge and the dynamic requirements of an engaging conversation:
1. **Primary**: OpenAI-compatible LLM integration (configured for NVIDIA NIM). Uses rich prompt engineering to inject category voice constraints, merchant performance metrics (vs peers), active offers, and explicit conversational history.
2. **Fallback**: If the LLM is unavailable or times out, a highly-contextualized deterministic router takes over. It uses trigger-specific templates that still inject live metrics, seasonal context, and correct tone (improving upon the original 59-score submission).
3. **Conversational State**: In-memory state tracking to handle auto-replies gracefully, manage explicit opt-outs, and switch from conversational to action-oriented "intent" modes based on merchant commitments.

## Improvements Over Baseline
- **Context Injection**: Incorporates peer CTR comparisons, exact metric dips, and specific review quotes.
- **Audience Correctness**: Customer-facing messages correctly handle the `merchant_on_behalf` scope, explicitly formatting messages from the merchant to the customer. Fixed suppression keys and customer name formatting.
- **Language Nuance**: Prompt instructions vary dynamically to support Hindi-English code-mixing or other requested languages based on the merchant/customer identity.
- **Specific Rationales**: Each generated message now has a unique, detailed rationale describing exactly what data points from the `Category`, `Merchant`, `Trigger`, and `Customer` contexts were anchored upon.
- **Safety**: URL stripping and strict token formatting prevent LLM hallucination and ensure WhatsApp compliance.

## Environment configuration
To enable the LLM engine for testing, set:
- `NIM_API_KEY` (or standard `OPENAI_API_KEY` if adapting the base URL)
- `NIM_MODEL` (default: `meta/llama-3.1-70b-instruct`)

Without an API key, the service runs purely on its enhanced deterministic fallback mode.

## Testing
- `python self_test.py` validates endpoint idempotency and state logic.
- `python generate_submission.py` re-generates the `submission.jsonl` test set using the deterministic fallback if no LLM is present.
