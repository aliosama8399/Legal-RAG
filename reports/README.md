# Reports

Generated artifacts live here. Nothing in this directory is committed by hand —
every file is produced by a command, so the numbers always match the code that
produced them.

## Load test

`locust_report.html` and `locust_report.csv` come from a 50-user run against the
running stack:

```bash
docker compose up -d bento-api
docker compose --profile loadtest run --rm loadtest
```

Then open `locust_report.html` in a browser. The run is 3 minutes with a ramp of
5 users/second, and `locustfile.py` fails any `/ask/stream` request that returns
zero tokens, so a passing report also proves streaming works under load.

To vary it:

```bash
docker compose --profile loadtest run --rm loadtest \
  python -m locust -f locustfile.py --host http://bento-api:3000 \
  --headless -u 50 -r 5 -t 5m --html reports/locust_report.html
```

## Evaluation

`../data/eval/evaluation_report.md` is the RAG evaluation report — retrieval
sweep, chunk sweep and Ragas scores, including how many questions were scored.

## Latency

`../data/eval/latency_report.md` accumulates one table per configuration, so the
before/after quantization comparison in the top-level README quotes measured
p50/p95 rather than estimates:

```bash
python -m scripts.latency_bench --label fp16
# switch .env to the 4-bit settings, restart vllm, then:
python -m scripts.latency_bench --label nf4-4bit
```
