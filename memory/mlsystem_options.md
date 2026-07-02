# ML Inference System Options

All options are evaluated against the same criteria: runs on K3s on a laptop,
produces a clear accuracy metric, outputs calibrated confidence scores, and has
a realistic feature/label pipeline that supports the planned chaos failure modes.

---

## Option A — Tabular Binary Classifier (XGBoost)

**Task:** Predict user conversion or fraud (synthetic e-commerce data, no downloads needed).
**Model:** XGBoost with `predict_proba()` for confidence scores.
**Features:** ~20 numeric + categorical features (age, cart value, click rate, device type, etc.).

### Pros
- Extremely lightweight — trains in seconds, inference in <1ms; K3s on a laptop is fine.
- Calibrated probabilities out of the box (XGBoost `predict_proba` → `prediction_confidence`).
- Easy to control training data distribution precisely — critical for manufacturing clean training-serving skew scenarios.
- Feature schema is fully synthetic and under your control, so every chaos scenario can be injected deterministically.
- Accuracy metric is clean: binary classification accuracy, easy to track.
- Closest match to the existing mock scenarios (which were designed around this pattern).

### Cons
- Not representative of how modern ML systems are actually deployed (most production systems are neural nets or LLMs).
- Concept drift requires artificial construction — you decide when the distribution shifts; it doesn't happen naturally.
- Less interesting latency profile — inference is so fast that latency chaos scenarios (OOM, CPU throttling) are the only levers.
- Low-fidelity for chaos scenarios like embedding drift or tokenizer failures, which don't apply.

### Best for
Getting the agent evaluation loop working quickly. Lowest setup risk.

---

## Option B — Text Sentiment Classifier (DistilBERT or TF-IDF + LR)

**Task:** Classify product review sentiment (positive/negative/neutral) from raw text.
**Model:** Two sub-options:
  - *Heavyweight:* DistilBERT fine-tuned (needs ~1GB RAM per pod, GPU preferred but not required).
  - *Lightweight:* TF-IDF vectorizer + Logistic Regression (CPU-only, fast, still text-native).
**Features:** Raw text → token embeddings (BERT) or bag-of-words vectors (TF-IDF).

### Pros
- Realistic modern NLP serving use case — representative of a real production system.
- Embedding drift is a natural, interesting chaos scenario: if the input text distribution shifts (e.g., new slang, different product category), the model degrades without any code change.
- Tokenizer failures are genuine chaos handles: feed out-of-vocabulary inputs, truncate sequences — produces real errors.
- Latency profile is interesting: BERT inference is 30–100ms, so latency spikes are measurable and meaningful.
- With the lightweight variant (TF-IDF + LR), you get text-native failure modes without the GPU/memory overhead.

### Cons
- The lightweight variant (TF-IDF + LR) loses the embedding-based failure modes that make NLP interesting.
- The heavyweight variant (DistilBERT) needs ~1–2GB RAM per inference pod — tight on a laptop K3s cluster.
- Feature pipeline is text preprocessing, not tabular features — `query_feature_distributions` (which computes PSI on numeric features) is awkward to apply. You'd need to track embedding statistics instead.
- Confidence scores from text classifiers are less well-calibrated than XGBoost by default (need Platt scaling or temperature scaling post-hoc).
- Synthetic text data is harder to generate convincingly than synthetic tabular data.

### Best for
If you want the system to feel like a real product, and you're willing to spend time on text data generation and embedding monitoring.

---

## Option C — Streaming Anomaly Detector (Isolation Forest + sliding window)

**Task:** Detect anomalous API requests or sensor readings from a continuous stream.
**Model:** Isolation Forest trained on a window of "normal" traffic; scores each incoming point.
**Features:** ~10 numeric features derived from request metadata (latency, payload size, request rate, error codes, etc.) over a rolling time window.

### Pros
- Temporal nature makes concept drift completely natural — "normal" behavior changes over time, and the model's anomaly scores drift without any artificial injection.
- Unsupervised → no label pipeline needed (removes one of the four services). Simpler system overall.
- Training-serving skew is the most realistic here: the training window captures one traffic pattern; serving encounters another.
- Streaming setup means the feature pipeline continuously computes rolling statistics — `query_feature_distributions` maps very naturally.

### Cons
- "Accuracy" is not well-defined for unsupervised anomaly detection — you need injected ground-truth anomalies to measure it.
- `prediction_confidence` doesn't have a natural equivalent — anomaly scores are not probabilities.
- No natural bad-deployment scenario tied to model version mismatches.

### Best for
If temporal and concept drift scenarios are your priority, and you're comfortable with a custom accuracy definition.

---

## Option D — Two-Tower Recommender (Candidate Retrieval + Ranker)

**Task:** Recommend products to users — first-stage ANN retrieval from an embedding index, second-stage LightGBM ranker.
**Model:** Two-tower neural net for retrieval + LightGBM ranker for scoring.
**Features:** User features + item features + interaction features; embeddings stored in FAISS.

### Pros
- Most representative of real production ML systems at scale.
- Two-stage architecture produces rich, distinct failure modes: retrieval failures look very different from ranking failures.
- A/B test contamination is a natural chaos scenario.
- Catalog drift (new items added/removed) produces natural concept drift.
- Interesting latency profile: retrieval vs. ranking are separately measurable.

### Cons
- Most complex to build of the original four — two models, two training pipelines, a vector DB, and a feature store.
- Harder to define a single scalar accuracy metric (NDCG or hit rate vs. simple binary accuracy).
- `prediction_confidence` requires extra work — rankers produce scores, not probabilities.
- K3s resource requirements are higher.

### Best for
If you want the agent tested against a genuinely production-grade system and have time to invest.

---

## Option E — LLM Inference Service (vLLM + small open-source model)

**Task:** Serve a small open-source LLM (e.g., Qwen-0.5B or TinyLlama-1.1B) via vLLM, handling
chat completion requests with token-level streaming. Quality is measured by an LLM-as-judge
scoring responses against reference answers.

**Model:** vLLM serving a 0.5B–1.1B parameter model. Small enough to run on CPU-only K3s
(slowly) or a single consumer GPU.

**Metrics exposed:**
- `tokens_per_second` → `throughput`
- `time_to_first_token_ms` → `latency_p50` / `latency_p99`
- `llm_judge_score` (0–1, rated by a second model call) → `accuracy` + `prediction_confidence`
- `request_error_rate` → `error_rate`

### Pros
- **This is literally what Anthropic and OpenAI run.** The failure modes (KV cache exhaustion,
  quantization regression, prompt template drift, context window overflow, worker crash loops)
  are the exact problems their SRE teams deal with daily. An agent that can diagnose these is
  immediately useful to them.
- LLM-as-judge gives you a continuous quality score that degrades naturally when the model or
  its inputs change — no need to construct artificial accuracy drops.
- vLLM exposes a Prometheus `/metrics` endpoint out of the box; almost no instrumentation work.
- Deployment version chaos is the most natural here: swapping model weights, changing
  quantization (fp16 → int4), updating the prompt template — all produce measurable regressions.
- The meta-narrative is compelling: **an AI agent diagnosing failures in an AI serving system.**
  This is a story that lands immediately with AI lab engineers and researchers.
- Most publishable framing: "automated root cause analysis for LLM serving infrastructure."

### Cons
- Even a 0.5B model needs ~1GB RAM; vLLM itself needs ~2GB overhead. Tight on a laptop K3s
  cluster, but workable with resource limits.
- Without a GPU, throughput will be low (~5–10 tokens/sec on CPU). This limits how realistic
  the latency signals are, though for chaos evaluation purposes the relative changes still work.
- LLM-as-judge scoring introduces a second API call per request (or a second local model),
  adding latency and cost.
- The "feature pipeline" concept doesn't map cleanly — there are no tabular features.
  `query_feature_distributions` would need to track prompt statistics (length distribution,
  token overlap, topic drift) instead of numeric feature PSI.
- Quantization and model weight chaos requires care to not brick the K3s node.

### Best for
Maximum relevance to AI labs. Use this if the goal is to impress or publish toward an
audience of AI infrastructure engineers and researchers.

---

## Option F — RAG Pipeline (Embedding Model + Vector DB + LLM Generator)

**Task:** Answer questions over a document corpus using retrieval-augmented generation.
Ingestion pipeline chunks + embeds documents into Qdrant; at query time, retrieves top-K
chunks and feeds them to a small LLM for answer generation.

**Components:**
- Ingestion service: chunking → embedding model (e.g., `all-MiniLM-L6-v2`) → Qdrant
- Retrieval service: query embedding → ANN search → ranked chunks
- Generation service: retrieved context + query → LLM → answer
- Evaluation service: LLM-as-judge scores answer vs. ground truth → `accuracy` metric

**Metrics exposed:**
- `retrieval_precision_at_k` → maps to `accuracy`
- `answer_relevance_score` (LLM-judge) → maps to `prediction_confidence`
- `retrieval_latency_ms` + `generation_latency_ms` → `latency_p50` / `latency_p99`
- `retrieval_error_rate` + `generation_error_rate` → `error_rate`

### Pros
- RAG is the dominant architecture for enterprise AI deployments — both Anthropic and OpenAI
  actively sell into this pattern (Claude for Enterprise, ChatGPT Enterprise both use it).
  An agent that diagnoses RAG failures is directly applicable.
- Multi-component architecture produces the richest and most distinct failure modes of any option:
  - Index staleness (documents updated but embeddings not refreshed)
  - Embedding model version mismatch (retriever trained on different embedding space than index)
  - Retrieval degradation without generation errors (agent must distinguish the two)
  - Context poisoning (retrieved chunks that actively mislead the generator)
  - Document pipeline corruption (chunking strategy change → retrieval quality collapses)
- The feature pipeline maps naturally: document chunk statistics, embedding drift (mean cosine
  similarity to centroid), retrieval score distributions — all are trackable as Prometheus metrics.
- `query_feature_distributions` can track embedding distribution drift via PSI on embedding
  dimension statistics — a genuinely interesting signal.
- Three independently-deployable services (ingestion, retrieval, generation) give you
  fine-grained deployment history — each can be versioned and rolled back independently.

### Cons
- Most operationally complex of all options: three model-serving components plus a vector DB
  (Qdrant), all running in K3s. Resource usage: ~2–3GB RAM total minimum.
- Two latency sources (retrieval + generation) that must be tracked separately, then combined —
  the tool schema only has two latency metrics (`latency_p50`, `latency_p99`), so you need to
  decide whether to expose end-to-end latency or per-component latency.
- Requires a real document corpus (even a synthetic one needs to be coherent enough for
  retrieval to work meaningfully).
- LLM-as-judge scoring adds API call cost and latency, same as Option E.
- Failure modes involving the vector DB (index corruption, Qdrant crashes) require chaos
  tooling that targets the DB pod, not just the model pods.

### Best for
The most architecturally rich option for AI lab audiences. If you want to demonstrate that
the agent can reason across a multi-component AI pipeline — not just a single model — this
is the strongest choice.

---

## Option G — Multi-Agent Orchestration System

**Task:** An orchestrator agent routes incoming tasks (code review, summarization, data
extraction) to specialized sub-agents. Each sub-agent is a separate model-serving pod.
The orchestrator tracks task completion rate and routes based on sub-agent health.

**Components:**
- Orchestrator service: classifies task type → routes to appropriate sub-agent
- Sub-agent pods: 2–3 specialized agents (e.g., Summarizer, Extractor, Coder)
- Evaluation service: scores task outputs via LLM-as-judge → `accuracy`

**Metrics exposed:**
- `task_completion_rate` → `accuracy`
- `routing_confidence_score` → `prediction_confidence`
- `end_to_end_latency_ms` + `per_agent_latency_ms` → `latency_*`
- `agent_error_rate` + `routing_error_rate` → `error_rate`

### Pros
- The meta-narrative is the strongest of all options: **an AI agent diagnosing failures
  in a system of AI agents.** This is directly on Anthropic's research agenda (multi-agent
  reliability) and highly novel — no evaluation benchmark we're aware of does this.
- Failure modes that are unique to this option and impossible to study otherwise:
  - Routing model degradation (orchestrator sends tasks to wrong sub-agent)
  - Sub-agent cascade failure (one pod fails → orchestrator floods another → both degrade)
  - Agent coordination failure (sub-agents produce inconsistent outputs when composed)
  - Prompt injection in one sub-agent affects downstream agents
- Extremely publishable: "agentic failure diagnosis for multi-agent systems" is a paper-worthy
  framing that doesn't exist yet.
- Each sub-agent pod can be versioned independently — rich deployment history.

### Cons
- Requires running 3–4 LLM-serving pods simultaneously, which is heavy for a laptop K3s node
  without GPU. Practically requires either small models (Qwen-0.5B each) or a machine with
  more RAM (16GB+).
- The most complex evaluation story: what counts as "correct" when multiple agents compose?
  LLM-as-judge helps but adds cost and uncertainty.
- Failure mode signals are noisier — it's genuinely hard to tell if `task_completion_rate`
  dropped because of the orchestrator, a sub-agent, or the evaluation judge.
- Building a convincing orchestrator requires prompt engineering effort, not just model
  wiring — you'd be building two systems at once.
- Higher risk that the target system's own complexity obscures the chaos signal you're
  trying to inject.

### Best for
Maximum novelty and research impact. Best if you're targeting a paper submission or
want to tell a story that no one else is telling yet.

---

## Summary Table

| Criterion | A (XGBoost) | B (DistilBERT) | C (Anomaly) | D (Recommender) | E (LLM Serving) | F (RAG) | G (Multi-Agent) |
|---|---|---|---|---|---|---|---|
| Setup complexity | Low | Medium | Medium | High | Medium | High | Very High |
| K3s resource usage | Very low | Medium | Low | High | Medium | High | Very High |
| Accuracy metric clarity | High | High | Low | Medium | Medium | Medium | Low |
| Confidence score quality | High | Medium | Low | Medium | Medium | Medium | Low |
| Feature pipeline realism | Medium | Low | High | High | Low | High | Low |
| Label pipeline realism | High | High | Low | High | Medium | Medium | Low |
| Natural concept drift | No | Partial | Yes | Yes | Yes | Yes | Partial |
| Chaos scenario richness | Medium | Medium | Medium | High | High | Very High | Very High |
| Time to first working system | 1–2 days | 2–3 days | 2–3 days | 4–6 days | 3–4 days | 5–7 days | 7–10 days |
| AI lab attractiveness | Low | Low | Low | Medium | Very High | Very High | Highest |

---

## Which Option AI Labs Find Most Attractive

AI labs (Anthropic, OpenAI, DeepMind) are running LLM inference at massive scale and
actively building multi-agent systems. Their SRE and reliability teams face the exact
failure modes that Options E, F, and G are designed around. Options A–D describe generic
ML system failures; Options E–G describe *AI system* failures.

**Ranked by AI lab attractiveness:**

1. **Option G (Multi-Agent)** — Highest. The only option with a story no one else is
   telling. "An AI agent that diagnoses failures in multi-agent systems" is directly on
   Anthropic's research roadmap and OpenAI's agents platform agenda. This is a paper, not
   just a tool. The risk is that it's genuinely hard to build well.

2. **Option F (RAG)** — Very High. RAG is the dominant enterprise AI architecture right
   now. Both labs actively help customers build and debug RAG systems. An agent that can
   diagnose "why did RAG quality degrade?" is a product, not just a research artifact.
   The multi-component failure mode space is the richest of any option.

3. **Option E (LLM Serving)** — Very High. Directly maps to what both companies operate
   internally. The meta-narrative (AI agent diagnosing AI serving failures) is immediately
   legible to anyone who has been on LLM infra on-call. Most practically useful.

4. **Option D (Recommender)** — Medium. Relevant to OpenAI's enterprise customers and to
   labs with consumer products, but not to the core infrastructure problems these labs face.

5. **Options A–C** — Low. These are generic ML system patterns. Not wrong, but not
   differentiated. No AI lab engineer would feel a personal connection to these failure modes.

**Practical recommendation for AI lab positioning:**

Build **Option E (LLM Serving)** as the base system. It's the best balance of AI lab
relevance, build complexity, and signal clarity. Once the evaluation harness is validated,
extend to **Option F (RAG)** by adding retrieval components in front of the LLM — you
reuse most of the serving infrastructure. Option G is the long-term research target if
you're heading toward a publication.
