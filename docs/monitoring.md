# Monitoring & Observability Guide

Complete guide for monitoring AI Forge in production.

## Overview

AI Forge includes a comprehensive monitoring stack:

- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and visualization
- **Alertmanager** - Alert routing and notifications

---

## Quick Start

### Enable Monitoring

```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
# Alertmanager: http://localhost:9093
```

---

## Metrics

### API Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request latency |
| `http_requests_in_progress` | Gauge | Active requests |

### Training Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `ai_forge_training_jobs_total` | Counter | Total training jobs |
| `ai_forge_training_jobs_failed_total` | Counter | Failed jobs |
| `ai_forge_training_job_duration_seconds` | Histogram | Job duration |
| `ai_forge_training_loss` | Gauge | Current training loss |

### Resource Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `process_resident_memory_bytes` | Gauge | Memory usage |
| `process_cpu_seconds_total` | Counter | CPU usage |
| `python_gc_objects_collected_total` | Counter | GC objects |

---

## Logging

### Configuration

```python
# In service.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/aiforge/app.log'),
    ]
)
```

### Log Levels

| Level | When to Use |
|-------|-------------|
| DEBUG | Development troubleshooting |
| INFO | Normal operations |
| WARNING | Unusual but handled situations |
| ERROR | Errors that affect functionality |
| CRITICAL | System-wide failures |

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "training_started",
    job_id="job_123",
    model="llama3.2",
    epochs=3,
)
```

---

## Alerts

### Configured Alerts

| Alert | Severity | Condition |
|-------|----------|-----------|
| AIForgeDown | Critical | Service unreachable for 1m |
| OllamaDown | Critical | Ollama unreachable for 1m |
| HighLatency | Warning | p95 latency > 1s for 5m |
| HighErrorRate | Warning | Error rate > 5% for 5m |
| HighMemoryUsage | Warning | Memory > 12GB for 5m |
| TrainingJobFailed | Warning | Any job fails |
| TrainingJobStuck | Warning | Job > 2 hours |

### Alert Routing

```yaml
# alerts.yml
routes:
  - match:
      severity: critical
    receiver: pagerduty
  - match:
      severity: warning
    receiver: slack
```

---

## Dashboards

### API Performance Dashboard

Key panels:
1. Request rate (requests/second)
2. Latency percentiles (p50, p95, p99)
3. Error rate
4. Active connections

### Training Dashboard

Key panels:
1. Active training jobs
2. Job duration distribution
3. Training loss over time
4. Failed job count

### Resource Dashboard

Key panels:
1. Memory usage
2. CPU usage
3. Disk I/O
4. Network traffic

---

## Best Practices

### What to Log

```python
# DO log
logger.info("Request processed", user_id=user, duration_ms=100)
logger.error("Training failed", job_id=job, error=str(e))

# DON'T log
logger.debug(f"Full request: {request}")  # Too verbose
logger.info(f"API key: {api_key}")  # Sensitive data
```

### Metric Naming

```
# Good
ai_forge_training_jobs_total
ai_forge_request_duration_seconds

# Bad
training_jobs
latency
```

### Alert Best Practices

1. Use meaningful thresholds based on SLOs
2. Avoid alert fatigue (only alert on actionable items)
3. Include runbook links in alert descriptions
4. Route appropriately by severity

---

## SLOs (Service Level Objectives)

| Objective | Target | Measurement |
|-----------|--------|-------------|
| Availability | 99.9% | Uptime per month |
| Latency (p95) | < 500ms | API response time |
| Error Rate | < 1% | 5xx responses / total |
| Training Success | > 95% | Successful jobs / total |

---

## Runbooks

### API Down

1. Check container status: `docker ps`
2. View logs: `docker logs ai-forge`
3. Check resources: `docker stats`
4. Restart if needed: `docker-compose restart ai-forge`

### High Memory

1. Check current usage: `docker stats ai-forge`
2. Look for memory leaks in logs
3. Reduce batch size if training
4. Restart service to clear memory

### Training Failure

1. Check job logs: `cat output/{job_id}/training.log`
2. Verify data quality: `python -m judge.validate_data`
3. Check disk space: `df -h`
4. Retry with smaller batch size

---

## Integration

### Slack

```yaml
# alertmanager.yml
receivers:
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/xxx'
        channel: '#alerts'
        title: '{{ .Status }} - {{ .CommonAnnotations.summary }}'
```

### PagerDuty

```yaml
receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'your-key'
        severity: '{{ .CommonLabels.severity }}'
```

### Email

```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'team@example.com'
        from: 'alerts@example.com'
```

---

## Next Steps

- [Deployment Guide](deployment.md) - Deploy the stack
- [Production Checklist](production_checklist.md) - Go-live checks
- [Troubleshooting](troubleshooting.md) - Common issues
