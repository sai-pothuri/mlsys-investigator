# MLSys Investigator — Infrastructure Development Plan

## Overview

This plan builds the infrastructure for a real ML inference system that the chaos harness
will inject failures into and the diagnostic agent will investigate. The actual ML model is
a **configurable stub** — a FastAPI service that returns synthetic predictions driven by
env vars rather than a real model. This means:

- All infrastructure (Docker, K3s, Helm, Prometheus, Loki) can be built and validated
  before committing to a model choice
- Chaos injection is trivially simple: patch a ConfigMap, no code changes needed
- Swapping in a real model later requires only changing the inference service's `app.py`
  and its Dockerfile — every other component stays identical

**Explicitly out of scope for this plan:**
- Real model training (XGBoost, LLM, RAG — all deferred until infra is validated)
- Chaos injection logic
- Rewiring `src/tools.py` to query Prometheus/Loki

---

## Directory Structure

```
ml-system/
├── ml/
│   └── feature_schema.json           # Canonical 20-feature list (shared contract)
├── services/
│   ├── inference-service/
│   │   ├── app.py                    # Configurable stub — no real model
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── feature-pipeline/
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── label-pipeline/
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── load-generator/
│       ├── app.py
│       ├── requirements.txt
│       └── Dockerfile
├── docker-compose.yml
└── infra/
    ├── setup.sh                      # K3s + Helm bootstrap script
    ├── namespaces.yaml
    └── helm/
        ├── ml-system/                # Umbrella chart for all 4 services
        │   ├── Chart.yaml
        │   ├── values.yaml
        │   └── templates/
        │       ├── configmap.yaml
        │       ├── inference-deployment.yaml
        │       ├── feature-pipeline-deployment.yaml
        │       ├── label-pipeline-deployment.yaml
        │       ├── load-generator-deployment.yaml
        │       ├── services.yaml
        │       └── service-monitors.yaml
        └── monitoring/
            ├── prometheus-values.yaml
            └── loki-values.yaml
```

---

## Step 1: Feature Schema

**File:** `ml/feature_schema.json`

The canonical list of 20 features. This is the single source of truth for:
- What `query_feature_distributions` validates feature names against (used by agent's tool)
- What the feature pipeline generates and tracks statistics for
- What the inference service accepts in the request body

```json
{
  "version": "1.0.0",
  "features": {
    "numeric": [
      "user_age", "session_duration_s", "pages_viewed", "cart_value",
      "days_since_last_purchase", "click_rate", "scroll_depth",
      "device_battery_pct", "network_latency_ms", "api_response_time_ms",
      "feature_age_days", "request_hour"
    ],
    "categorical": [
      "device_type", "browser", "country_code", "product_category",
      "user_tier", "referral_source", "payment_method", "platform"
    ]
  },
  "baseline_distributions": {
    "user_age": {"mean": 34.2, "std": 11.5, "min": 18, "max": 85},
    "session_duration_s": {"mean": 312.0, "std": 180.0, "min": 5, "max": 3600},
    "feature_age_days": {"mean": 182.5, "std": 60.0, "min": 0, "max": 365}
  }
}
```

The `baseline_distributions` section is used by the feature pipeline to generate realistic
synthetic feature vectors and to compute PSI against for drift detection.

**Note on `feature_age_days`:** This specific feature is the chaos handle for the
`FEATURE_DRIFT` scenario from `src/scenarios.py` — injecting a schema change causes it to
spike to ~847 (well outside the 0–365 baseline range).

---

## Step 2: Placeholder Inference Service

**File:** `services/inference-service/app.py`

FastAPI app that simulates an ML inference service without loading a real model. All
behaviour is driven by env vars, which become Kubernetes ConfigMap entries.

### Env vars (ConfigMap keys)

| Env var | Default | Effect |
|---|---|---|
| `ACCURACY_RATE` | `0.91` | Fraction of predictions the label-pipeline will mark "correct" |
| `CONFIDENCE_MEAN` | `0.87` | Mean of Gaussian sampling for confidence scores |
| `CONFIDENCE_STD` | `0.05` | Std dev for confidence score sampling |
| `LATENCY_MEAN_MS` | `45.0` | Mean artificial latency injected per request |
| `LATENCY_STD_MS` | `8.0` | Std dev for latency sampling |
| `ERROR_RATE` | `0.002` | Fraction of requests that raise a 500 error |
| `MODEL_VERSION` | `v1.0.0` | Reported in logs; used for deployment history |

### Endpoints

- `POST /predict` — accepts `{"request_id": str, "features": {name: value, ...}}`,
  validates feature names against `feature_schema.json`, sleeps for sampled latency,
  raises error at `ERROR_RATE` frequency, returns `{"prediction": int, "confidence": float, "request_id": str, "model_version": str}`
- `GET /health` — liveness probe, returns `{"status": "ok"}`
- `GET /metrics` — Prometheus exposition format

### Prometheus metrics

```
inference_latency_seconds{endpoint="/predict"}   # Histogram
inference_confidence_score                        # Histogram, buckets=[0.1,0.2,...,1.0]
inference_predictions_total{prediction_class="0|1"}  # Counter
inference_errors_total{error_type="validation|internal"}  # Counter
```

These map to the agent's `MetricName` enum:
- `latency_p50` / `latency_p99` ← `histogram_quantile(0.5|0.99, inference_latency_seconds)`
- `prediction_confidence` ← mean of `inference_confidence_score`
- `throughput` ← `rate(inference_predictions_total[5m])`
- `error_rate` ← `rate(inference_errors_total[5m]) / rate(inference_predictions_total[5m])`

### Logging

All log lines are structured JSON to stdout:
```json
{"timestamp": "2026-07-02T14:32:01Z", "severity": "info", "service": "inference_service", "message": "prediction served", "request_id": "...", "model_version": "v1.0.0", "confidence": 0.84}
```

Error log example:
```json
{"timestamp": "...", "severity": "error", "service": "inference_service", "message": "internal error during inference", "request_id": "...", "error_type": "internal"}
```

The `service` field must match the `ServiceName` enum value exactly: `inference_service`.

---

## Step 3: Feature Pipeline

**File:** `services/feature-pipeline/app.py`

Daemon that wakes every 30 seconds (configurable via `BATCH_INTERVAL_S`).

### Logic per tick
1. Generate a batch of N synthetic feature vectors (N = `BATCH_SIZE`, default 100) using
   the distributions in `feature_schema.json`
2. Validate each feature name against the schema; log WARNING for any unknown feature
3. Write the batch to a shared volume as `feature_batch_latest.json` for the inference
   service to optionally consume (used by label-pipeline to associate features with predictions)
4. Compute per-feature statistics (mean, std) over the batch
5. Update Prometheus gauges

### Prometheus metrics

```
feature_mean{feature="<name>"}    # Gauge, one per feature
feature_stddev{feature="<name>"}  # Gauge, one per feature
feature_batch_size                # Gauge
feature_validation_errors_total   # Counter
```

### Logging

```json
{"severity": "info", "service": "feature_pipeline", "message": "batch generated", "batch_size": 100, "validation_errors": 0}
{"severity": "warning", "service": "feature_pipeline", "message": "schema validation error", "unknown_feature": "event_date", "expected": "insertion_timestamp"}
```

The WARNING log above is the chaos handle for the `FEATURE_DRIFT` scenario — injecting
an upstream schema change causes validation errors that surface here.

---

## Step 4: Label Pipeline

**File:** `services/label-pipeline/app.py`

Daemon that wakes every 15 seconds (configurable via `EVAL_INTERVAL_S`).

### Logic per tick
1. Read the last N prediction records from a shared prediction log written by the
   inference service (JSON-Lines file on a shared volume)
2. For each prediction, generate a synthetic ground-truth label:
   - With probability `ACCURACY_RATE` (from env var, default 0.91), label = prediction
   - Otherwise, label ≠ prediction (i.e., the model was wrong)
3. Join labels to predictions by `request_id`; log WARNING for any unmatched IDs
4. Compute rolling accuracy over the joined set
5. Update Prometheus gauges

### Prometheus metrics

```
label_accuracy_rate          # Gauge — fraction of correct predictions
label_join_success_rate      # Gauge — fraction of predictions successfully joined to a label
label_eval_batch_size        # Gauge
```

`label_join_success_rate` is the chaos handle for `LABEL_PIPELINE_CORRUPTION` — a config
change to the join key drops this metric while `label_accuracy_rate` appears to degrade
(but the model is actually unchanged).

### Logging

```json
{"severity": "info", "service": "label_pipeline", "message": "evaluation complete", "accuracy": 0.912, "join_success_rate": 1.0, "batch_size": 85}
{"severity": "warning", "service": "label_pipeline", "message": "join miss", "request_id": "...", "reason": "key not found"}
```

---

## Step 5: Load Generator

**File:** `services/load-generator/app.py`

Simple request loop that keeps all metrics non-zero at baseline.

### Logic
- Loop forever: POST synthetic prediction requests to `http://inference-service:8080/predict`
- Feature values are randomly sampled from `feature_schema.json` baseline distributions
- Sleep `1 / REQUEST_RATE` seconds between requests (default `REQUEST_RATE=10`, so 10 req/s)
- Log each response at INFO; log errors at ERROR

### Env vars

| Env var | Default |
|---|---|
| `INFERENCE_SERVICE_URL` | `http://inference-service:8080` |
| `REQUEST_RATE` | `10` |

### Logging

```json
{"severity": "info", "service": "load_generator", "message": "prediction request sent", "request_id": "...", "response_ms": 48}
{"severity": "error", "service": "load_generator", "message": "prediction request failed", "request_id": "...", "status_code": 500}
```

Note: `load_generator` is not in the `ServiceName` enum (which only covers the four
diagnostic targets). Its logs are informational only and won't be queried by the agent.

---

## Step 6: Dockerfiles

All four services share the same Dockerfile pattern:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY ../../ml/feature_schema.json /app/feature_schema.json
CMD ["python", "app.py"]
```

No model artifact is copied. The `feature_schema.json` is the only shared file.

### `requirements.txt` per service

**inference-service:** `fastapi`, `uvicorn[standard]`, `prometheus-client`, `pydantic`

**feature-pipeline:** `prometheus-client`, `pydantic`

**label-pipeline:** `prometheus-client`, `pydantic`

**load-generator:** `httpx`, `pydantic`

---

## Step 7: Docker Compose (Local Dev)

**File:** `docker-compose.yml`

Runs all 4 services + Prometheus + Loki locally without K3s. Used for rapid iteration.

```yaml
services:
  inference-service:
    build: services/inference-service
    ports: ["8080:8080"]
    env_file: services/inference-service/.env.defaults
    volumes:
      - prediction-log:/app/logs
    networks: [ml-net]

  feature-pipeline:
    build: services/feature-pipeline
    volumes:
      - feature-batches:/app/data
    networks: [ml-net]

  label-pipeline:
    build: services/label-pipeline
    volumes:
      - prediction-log:/app/logs
    networks: [ml-net]

  load-generator:
    build: services/load-generator
    environment:
      INFERENCE_SERVICE_URL: http://inference-service:8080
    networks: [ml-net]
    depends_on: [inference-service]

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./infra/prometheus-local.yml:/etc/prometheus/prometheus.yml
    networks: [ml-net]

  loki:
    image: grafana/loki:latest
    ports: ["3100:3100"]
    networks: [ml-net]

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - ./infra/promtail-local.yml:/etc/promtail/config.yml
    networks: [ml-net]
    depends_on: [loki]

volumes:
  prediction-log:
  feature-batches:

networks:
  ml-net:
```

Additional files needed for local compose:
- `infra/prometheus-local.yml` — scrapes all 4 service `/metrics` endpoints
- `infra/promtail-local.yml` — reads Docker container logs, labels by `service` field from JSON

---

## Step 8: K3s Setup

**File:** `infra/setup.sh`

Run once on a fresh machine to bootstrap the full cluster.

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Install K3s (single-node, disable Traefik — we don't need it)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik" sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# 2. Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 3. Add repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# 4. Create namespaces
kubectl apply -f infra/namespaces.yaml

# 5. Deploy monitoring stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  -f infra/helm/monitoring/prometheus-values.yaml \
  --wait

helm install loki grafana/loki-stack \
  -n monitoring \
  -f infra/helm/monitoring/loki-values.yaml \
  --wait

# 6. Build images and load into K3s image store
SERVICES=(inference-service feature-pipeline label-pipeline load-generator)
for svc in "${SERVICES[@]}"; do
  docker build -t ml-system/${svc}:latest ml-system/services/${svc}/
  docker save ml-system/${svc}:latest | sudo k3s ctr images import -
done

# 7. Deploy ml-system
helm install ml-system infra/helm/ml-system/ \
  -n ml-system \
  --wait

echo "Done. Run 'kubectl get pods -A' to verify."
```

**File:** `infra/namespaces.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-system
---
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
```

---

## Step 9: Helm Charts

### `infra/helm/monitoring/prometheus-values.yaml`

Key overrides for `kube-prometheus-stack`:

```yaml
prometheus:
  prometheusSpec:
    serviceMonitorSelectorNilUsesHelmValues: false  # pick up ServiceMonitors from ml-system ns
    retention: 30d
    resources:
      requests:
        memory: 512Mi
        cpu: 200m

grafana:
  adminPassword: "mlsys-investigator"
  persistence:
    enabled: false
```

### `infra/helm/monitoring/loki-values.yaml`

Key overrides for `loki-stack`:

```yaml
loki:
  persistence:
    enabled: false
  resources:
    requests:
      memory: 256Mi

promtail:
  enabled: true
  config:
    snippets:
      pipelineStages:
        - json:
            expressions:
              severity: severity
              service: service
              message: message
        - labels:
            severity:
            service:
```

This pipeline stage parses the JSON logs from all pods, extracts `severity` and `service`
as Loki stream labels. The agent's `query_logs` tool will filter on these labels.

### `infra/helm/ml-system/Chart.yaml`

```yaml
apiVersion: v2
name: ml-system
version: 0.1.0
description: ML inference system for chaos testing
```

### `infra/helm/ml-system/values.yaml`

```yaml
imageTag: latest
imagePullPolicy: Never   # use locally-imported images in K3s

inferenceService:
  replicas: 1
  resources:
    requests: {cpu: 100m, memory: 128Mi}
    limits: {cpu: 500m, memory: 256Mi}
  config:
    ACCURACY_RATE: "0.91"
    CONFIDENCE_MEAN: "0.87"
    CONFIDENCE_STD: "0.05"
    LATENCY_MEAN_MS: "45.0"
    LATENCY_STD_MS: "8.0"
    ERROR_RATE: "0.002"
    MODEL_VERSION: "v1.0.0"

featurePipeline:
  replicas: 1
  resources:
    requests: {cpu: 50m, memory: 64Mi}
  config:
    BATCH_INTERVAL_S: "30"
    BATCH_SIZE: "100"

labelPipeline:
  replicas: 1
  resources:
    requests: {cpu: 50m, memory: 64Mi}
  config:
    EVAL_INTERVAL_S: "15"

loadGenerator:
  replicas: 1
  resources:
    requests: {cpu: 50m, memory: 64Mi}
  config:
    REQUEST_RATE: "10"
    INFERENCE_SERVICE_URL: "http://inference-service:8080"
```

### `infra/helm/ml-system/templates/configmap.yaml`

One ConfigMap per service, populated from `values.yaml`. Chaos injection patches these
ConfigMaps and triggers a rolling restart — no code changes required.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-service-config
  namespace: ml-system
data:
  {{- range $k, $v := .Values.inferenceService.config }}
  {{ $k }}: {{ $v | quote }}
  {{- end }}
```

### `infra/helm/ml-system/templates/inference-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
  namespace: ml-system
  annotations:
    deployment.kubernetes.io/revision: "1"   # incremented on each rollout
spec:
  replicas: {{ .Values.inferenceService.replicas }}
  selector:
    matchLabels: {app: inference-service}
  template:
    metadata:
      labels: {app: inference-service}
      annotations:
        # Forces pod restart when ConfigMap changes
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      containers:
        - name: inference-service
          image: ml-system/inference-service:{{ .Values.imageTag }}
          imagePullPolicy: {{ .Values.imagePullPolicy }}
          ports:
            - name: http
              containerPort: 8080
            - name: metrics
              containerPort: 8080
          envFrom:
            - configMapRef:
                name: inference-service-config
          resources: {{- toYaml .Values.inferenceService.resources | nindent 12 }}
          livenessProbe:
            httpGet: {path: /health, port: 8080}
            initialDelaySeconds: 5
```

Same pattern for `feature-pipeline-deployment.yaml`, `label-pipeline-deployment.yaml`,
`load-generator-deployment.yaml` — swap names and config references.

### `infra/helm/ml-system/templates/services.yaml`

One ClusterIP Service per ml-system pod with a named `metrics` port:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: inference-service
  namespace: ml-system
spec:
  selector: {app: inference-service}
  ports:
    - name: http
      port: 8080
      targetPort: 8080
    - name: metrics
      port: 8080
      targetPort: 8080
```

### `infra/helm/ml-system/templates/service-monitors.yaml`

One `ServiceMonitor` per service. Tells Prometheus (in the `monitoring` namespace) to
scrape the `metrics` port:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: inference-service
  namespace: ml-system
  labels:
    release: prometheus    # must match kube-prometheus-stack's serviceMonitorSelector
spec:
  selector:
    matchLabels: {app: inference-service}
  endpoints:
    - port: metrics
      interval: 15s
```

---

## Step 10: Deployment History

`query_deployment_history` will eventually need a backend. Design decision for now:

- **All model version changes go through `kubectl rollout`** (Helm upgrade or
  `kubectl set image`) — this ensures Kubernetes preserves rollout history
- Rollout history is queryable with:
  ```bash
  kubectl rollout history deployment/inference-service -n ml-system
  ```
- A thin adapter (`infra/deployment-history-exporter.py`) that watches the K8s API
  for Deployment revision changes and writes `DeploymentEvent` records to a JSON log
  is **deferred to the tool-rewiring phase** (after this infra is validated)

For now: document that the pattern is `kubectl rollout` for all changes, and the history
is preserved automatically.

---

## Verification

After running `infra/setup.sh`:

### 1. All pods running
```bash
kubectl get pods -n ml-system
kubectl get pods -n monitoring
```
Expect: `inference-service`, `feature-pipeline`, `label-pipeline`, `load-generator` all
`Running 1/1`. Plus Prometheus, Grafana, AlertManager, Loki, Promtail in `monitoring`.

### 2. Inference service responding
```bash
kubectl port-forward svc/inference-service 8080:8080 -n ml-system &
curl -s -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"request_id": "smoke-test-1", "features": {"user_age": 28, "session_duration_s": 420, "pages_viewed": 5, "cart_value": 89.99, "days_since_last_purchase": 14, "click_rate": 0.23, "scroll_depth": 0.71, "device_battery_pct": 82, "network_latency_ms": 38, "api_response_time_ms": 52, "feature_age_days": 180, "request_hour": 14, "device_type": 1, "browser": 2, "country_code": 0, "product_category": 3, "user_tier": 1, "referral_source": 0, "payment_method": 1, "platform": 0}}'
```
Expect: `{"prediction": 1, "confidence": 0.84, "request_id": "smoke-test-1", "model_version": "v1.0.0"}`

### 3. Metrics flowing into Prometheus
```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring &
```
Open `http://localhost:9090` and run these queries:
- `rate(inference_predictions_total[5m])` — expect ~10/s
- `histogram_quantile(0.99, inference_latency_seconds_bucket)` — expect ~0.07s
- `label_accuracy_rate` — expect ~0.91
- `feature_mean{feature="feature_age_days"}` — expect ~182

### 4. Logs in Loki
```bash
kubectl port-forward svc/prometheus-grafana 3000:80 -n monitoring &
```
Open `http://localhost:3000` (admin / mlsys-investigator), go to Explore → Loki datasource,
run:
```
{service="inference_service"} | json | severity="info"
```
Expect: prediction log lines flowing in real-time.

### 5. Chaos injection smoke test
Patch the ConfigMap to simulate a bad deployment (high error rate):
```bash
kubectl patch configmap inference-service-config -n ml-system \
  --type merge \
  -p '{"data": {"ERROR_RATE": "0.15", "MODEL_VERSION": "v1.1.0-broken"}}'

kubectl rollout restart deployment/inference-service -n ml-system
kubectl rollout status deployment/inference-service -n ml-system
```
Then re-run the Prometheus query: `rate(inference_errors_total[5m])` — expect it to spike.
This confirms the chaos injection path works end-to-end before building the real harness.

---

## Sequencing

```
Step 1 (feature_schema.json)
    └── unblocks Steps 2, 3, 4, 5 (all services) — run in parallel

Steps 2–5 → Step 6 (Dockerfiles) → Step 7 (docker-compose, local validation)
                                  → Step 8 (K3s setup) → Step 9 (Helm deploy)

Step 9 → Step 10 (deployment history — deferred, just document the pattern now)
       → Verification
```

Local validation with Docker Compose (Step 7) should happen before K3s deployment (Step 8)
to catch service bugs cheaply.
