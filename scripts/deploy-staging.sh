#!/usr/bin/env sh
#
# Staging deploy on the EC2 host: pull AWS secrets, then bring up the stack.
# Secrets are fetched at deploy time (host-side) — never baked into the image.
#
# Required env:
#   SECRET_ID            Secret name or ARN, e.g. kaapi/staging
#   DOCKER_IMAGE_BACKEND Backend image (ECR repo)
#   TAG                  Image tag to run

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "${SCRIPT_DIR}")
cd "${PROJECT_DIR}"

sh "${SCRIPT_DIR}/fetch-secrets.sh"

docker compose -f docker-compose.staging.yml pull
docker compose -f docker-compose.staging.yml --profile migrate run --rm migrate
docker compose -f docker-compose.staging.yml up -d
