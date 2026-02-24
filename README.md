# Development

Run tests: `pytest tests`
Run benchmarks: `pytest benchmarks`
Fancy benchmarks: `pytest -q -k test_bench_exact_quadrature --benchmark-only --benchmark-group-by=param:d --benchmark-sort=mean --benchmark-name=short --benchmark-columns=mean`
