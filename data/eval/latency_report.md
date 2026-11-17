# Latency Benchmark Report

## local-cpu-hybrid-mock-llm (10 questions, top_k=5)

| pass | mean | p50 | p95 | p99 | errors |
|---|---|---|---|---|---|
| sequential | 1.392s | 1.137s | 3.611s | 3.611s | 0 |
| concurrent x4 | 4.081s | 3.631s | 5.523s | 5.523s | 0 |
