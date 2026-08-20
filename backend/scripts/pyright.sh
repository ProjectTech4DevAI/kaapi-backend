#!/usr/bin/env bash

set -e
set -x

# Pyright runs via uvx (ephemeral) so it never touches uv.lock.
# Default target is app/; pass paths/files to narrow the check.
TARGET="${*:-app}"

uvx pyright $TARGET
