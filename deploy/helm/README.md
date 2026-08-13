# RigorousRAG Helm deployment profile

This chart is a conservative Kubernetes baseline for the current RigorousRAG storage model.
It intentionally deploys **exactly one application replica**. The application currently owns
local SQLite registries and local Chroma/classic-storage state, so active-active horizontal
scaling would create incorrect coordination semantics. The chart fails rendering when
`replicaCount > 1` or `distributedMode.enabled=true` rather than pretending those modes are
safe.

## Persistent state

All mutable application paths are redirected beneath `/data` and share one `ReadWriteOnce`
PersistentVolumeClaim by default:

- Chroma/vector data
- sparse index SQLite state
- index-generation registry
- job registry
- document registry
- uploads
- classic search storage
- usage telemetry JSONL
- evidence-graph, evidence-graph-job, and evidence-graph-compaction registries

The Deployment uses `Recreate` strategy so a rolling update does not intentionally run two
writers against that volume. Disabling persistence is supported for disposable development
clusters only; it uses `emptyDir` and loses all state with the Pod.

## Health and security

Startup and readiness probes execute the repository-owned `python -m tools.healthcheck`
contract. Liveness uses the HTTP port as a process-level signal. The Pod uses the image's
non-root user, RuntimeDefault seccomp, no privilege escalation, and drops all Linux
capabilities. The default NetworkPolicy permits port 8000 only from Pods in the same
namespace; adapt ingress policy deliberately when exposing the service through an ingress
controller or gateway in another namespace.

Provider credentials and other secrets must not be stored in `values.yaml`. Create a Secret
outside this chart and set `existingSecret` to its name; it is injected with `envFrom`.

## Example

```bash
helm upgrade --install rigorousrag ./deploy/helm \
  --set image.repository=registry.example/rigorousrag \
  --set image.tag=4.4.0 \
  --set existingSecret=rigorousrag-provider-secrets
```

## Deliberately unsupported production modes

The chart does not claim distributed correctness. Before more than one replica can be
allowed, RigorousRAG still needs validated external durable registries/stores, distributed
leases or queue ownership, cross-worker idempotency/exactly-once effect handling, distributed
rate limiting/admission control, and migration/cutover coordination that survives worker
failure. Those are tracked as separate production-hardening work rather than hidden behind a
replica-count switch.
