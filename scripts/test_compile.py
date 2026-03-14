"""Compile experiment placeholder.

The production Hopper inference path intentionally avoids torch compilation and
regional compilation by default because first-request latency is the priority.
If compile experiments return, keep them isolated from the production app and
benchmark them separately against the deployed fp16 snapshot path.
"""

if __name__ == "__main__":
    print(
        "Compile experiments are disabled for the default Hopper inference path. "
        "Use scripts/benchmark_inference.py to benchmark the deployed production setup."
    )
