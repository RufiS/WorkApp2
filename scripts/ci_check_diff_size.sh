#!/bin/bash
set -e

BASE_SHA="${{ github.event.pull_request.base.sha }}"
HEAD_SHA="${{ github.sha }}"

# Ensure we have the base commit
git fetch --depth=1 origin "$BASE_SHA"

FILES_CHANGED=$(git diff --name-only "$BASE_SHA" "$HEAD_SHA" | wc -l)
LINES_CHANGED=$(git diff --numstat "$BASE_SHA" "$HEAD_SHA" | awk '{sum+=$1+$2} END {print sum}')

echo "📊 PR size: $FILES_CHANGED files, $LINES_CHANGED lines"

if [ "$FILES_CHANGED" -gt 3 ]; then
  echo "❌ PR touches $FILES_CHANGED files (max: 3)"
  exit 1
fi

if [ "$LINES_CHANGED" -gt 500 ]; then
  echo "❌ PR changes $LINES_CHANGED lines (max: 500)"
  exit 1
fi

echo "✅ PR size within limits"
