# Production Checklist

Complete checklist for deploying AI Forge to production.

---

## Pre-Deployment

### Code Quality

- [ ] All unit tests pass (`pytest tests/unit`)
- [ ] All integration tests pass (`pytest tests/integration`)
- [ ] No critical linting errors (`ruff check .`)
- [ ] Type checking passes (`mypy ai_forge`)
- [ ] Code coverage > 80%

### Security Review

- [ ] No hardcoded secrets in code
- [ ] API keys stored in environment variables
- [ ] Dependencies scanned for vulnerabilities (`pip-audit`)
- [ ] CORS configured properly
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints

### Performance

- [ ] Latency benchmarks acceptable (< 500ms p95)
- [ ] Memory usage within limits (< 12GB peak)
- [ ] Training time acceptable for target model
- [ ] No memory leaks in long-running tests

### Documentation

- [ ] README updated with latest features
- [ ] API documentation complete
- [ ] Deployment guide reviewed
- [ ] Troubleshooting guide updated
- [ ] Changelog updated

### Infrastructure

- [ ] Backup procedures tested
- [ ] Disaster recovery plan documented
- [ ] Monitoring configured
- [ ] Alerting rules set up
- [ ] Log aggregation configured

---

## Deployment Day

### Pre-Deployment (T-1 hour)

- [ ] Notify stakeholders of deployment
- [ ] Create backup of current system
- [ ] Verify backup restoration works
- [ ] Prepare rollback procedure
- [ ] Check system resources available

### Deployment (T-0)

#### Step 1: Backup Current State

```bash
# Backup current models
./scripts/backup.sh

# Verify backup
ls -la /backups/ai-forge/
```

- [ ] Backup completed successfully
- [ ] Backup verified

#### Step 2: Stop Services

```bash
# Graceful shutdown
docker-compose down
# OR
pkill uvicorn
```

- [ ] Services stopped gracefully
- [ ] No active jobs in progress

#### Step 3: Deploy New Version

```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -e ".[all]"

# Apply migrations (if any)
./scripts/migrate.sh
```

- [ ] Code updated
- [ ] Dependencies installed
- [ ] Migrations applied

#### Step 4: Start Services

```bash
# Start Ollama
ollama serve &

# Start AI Forge
docker-compose up -d
# OR
uvicorn conductor.service:app --host 0.0.0.0 --port 8000 &
```

- [ ] Ollama running
- [ ] AI Forge service running

#### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test endpoint
curl -X POST http://localhost:8000/v1/retrain \
  -d '{"project_path": ".", "force": false}'

# Check logs
tail -f /var/log/aiforge/stdout.log
```

- [ ] Health check passes
- [ ] API responds correctly
- [ ] No errors in logs
- [ ] Ollama connection working

#### Step 6: Smoke Tests

```bash
# Run smoke tests
pytest tests/smoke -v

# Manual query test
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

- [ ] Smoke tests pass
- [ ] Manual tests successful

---

## Post-Deployment

### Immediate (0-1 hour)

- [ ] Monitor error logs for exceptions
- [ ] Check resource usage (CPU, memory)
- [ ] Verify no 5xx errors in logs
- [ ] Confirm alerting is working

### Short-Term (1-24 hours)

- [ ] Monitor query latency (< 500ms p95)
- [ ] Check for memory leaks (stable memory)
- [ ] Verify no OOM errors
- [ ] Review user feedback
- [ ] Monitor training job success rate

### Long-Term (1-7 days)

- [ ] Analyze performance trends
- [ ] Review error patterns
- [ ] Update documentation with learnings
- [ ] Schedule post-mortem if issues found
- [ ] Plan improvements for next release

---

## Rollback Procedure

### When to Rollback

- [ ] Critical functionality broken
- [ ] Error rate > 5%
- [ ] Latency > 2x normal
- [ ] OOM errors frequent
- [ ] Data corruption detected

### Rollback Steps

```bash
# 1. Stop current services
docker-compose down

# 2. Restore previous version
git checkout previous-release-tag

# 3. Restore dependencies
pip install -e ".[all]"

# 4. Restore data from backup
./scripts/restore.sh /backups/ai-forge/latest.tar.gz

# 5. Start services
docker-compose up -d

# 6. Verify rollback
curl http://localhost:8000/health
```

- [ ] Services stopped
- [ ] Previous version restored
- [ ] Backup restored
- [ ] Services restarted
- [ ] Health checks pass

---

## Monitoring Checklist

### Metrics to Watch

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| API Latency (p95) | < 500ms | 500-1000ms | > 1000ms |
| Error Rate | < 1% | 1-5% | > 5% |
| Memory Usage | < 70% | 70-90% | > 90% |
| CPU Usage | < 60% | 60-80% | > 80% |
| Disk Usage | < 70% | 70-85% | > 85% |

### Alerts Configured

- [ ] API down alert (immediate)
- [ ] High error rate alert (> 5%)
- [ ] High latency alert (p95 > 1s)
- [ ] High memory alert (> 90%)
- [ ] Disk space alert (> 85%)
- [ ] Training failure alert

---

## Security Checklist

### Network

- [ ] HTTPS enabled
- [ ] TLS 1.2+ only
- [ ] Firewall configured
- [ ] Only required ports open (8000, 11434)

### Application

- [ ] Debug mode disabled
- [ ] Verbose logging disabled
- [ ] API keys rotated
- [ ] CORS restricted to allowed origins

### Access

- [ ] SSH keys only (no passwords)
- [ ] Admin access audited
- [ ] Service accounts minimal privilege
- [ ] Secrets in secure storage

---

## Quick Reference

### Essential Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Health check
curl http://localhost:8000/health

# Restart service
docker-compose restart ai-forge

# Check resources
docker stats
```

### Emergency Contacts

| Role | Name | Contact |
|------|------|---------|
| On-Call Engineer | TBD | TBD |
| Platform Team | TBD | TBD |
| Security Team | TBD | TBD |

### Important URLs

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Health Check | http://localhost:8000/health |
| Docs | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |
| Metrics | http://localhost:9090 |

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| QA | | | |
| DevOps | | | |
| Manager | | | |
