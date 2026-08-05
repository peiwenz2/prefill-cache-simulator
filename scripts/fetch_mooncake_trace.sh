#!/usr/bin/env bash
set -euo pipefail

trace_url='https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl'
expected_sha='b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711'
target_path="${1:-mooncake_trace.jsonl}"
temporary_path="${target_path}.download"

curl --fail --location --silent --show-error "$trace_url" --output "$temporary_path"
actual_sha=$(shasum -a 256 "$temporary_path" | awk '{print $1}')
if [[ "$actual_sha" != "$expected_sha" ]]; then
  printf 'trace SHA-256 mismatch: expected %s, got %s\n' "$expected_sha" "$actual_sha" >&2
  exit 1
fi
mv "$temporary_path" "$target_path"
printf 'verified %s\n' "$target_path"
