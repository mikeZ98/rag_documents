# Corpus

Drop any `.pdf` files here and run:

```bash
uv run python -m src.ingest --recreate
```

`sample_policy_handbook.pdf` ships with the repository: a five-page synthetic
"Acme Cloud Services" master services agreement and employee handbook. The
questions in `eval/eval_ragas.py` are written against it, so the evaluation
harness works out of the box.

Real corpora are git-ignored (`documents/*.pdf`, except `sample_*.pdf`) — keep
customer documents out of version control.
