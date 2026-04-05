# perf: optimize time-range query performance

## Summary
- Implement partition pruning for date columns
- Add index hints for common query patterns
- Cache frequently accessed aggregations

## Benchmarks
- 7-day query: 450ms -> 42ms (10.7x)
- 30-
