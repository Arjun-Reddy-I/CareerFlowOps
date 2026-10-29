#!/usr/bin/env bash
# Run on the authorized EC2 host from a checkout matching the image SHA.
set -euo pipefail
: "${API_IMAGE:?Export the immutable ECR API image URI with Git SHA tag}"
: "${WEB_IMAGE:?Export the immutable ECR web image URI with Git SHA tag}"
: "${AWS_REGION:?Set AWS_REGION}"
[[ -f .env ]] || { echo 'Create a protected .env on the EC2 host first.' >&2; exit 1; }
command -v aws >/dev/null
command -v docker >/dev/null
command -v curl >/dev/null
command -v python3 >/dev/null
command -v git >/dev/null
head_sha="$(git rev-parse HEAD)"
[[ "${API_IMAGE##*:}" == "$head_sha" && "${WEB_IMAGE##*:}" == "$head_sha" ]] || {
  echo 'Both image tags must equal the full Git SHA of this checkout.' >&2; exit 1;
}
registry="${API_IMAGE%%/*}"
[[ "$registry" == *.dkr.ecr.*.amazonaws.com ]] || { echo 'API_IMAGE must reference your ECR registry.' >&2; exit 1; }
[[ "${WEB_IMAGE%%/*}" == "$registry" ]] || { echo 'Both images must use the same ECR registry.' >&2; exit 1; }
: "${DVC_REMOTE_URL:?Export the production s3://bucket/prefix DVC remote URL}"
[[ "$DVC_REMOTE_URL" == s3://*/* ]] || { echo 'Use an S3 remote with a project prefix.' >&2; exit 1; }
dvc_bin="${DVC_BIN:-.venv-deploy/bin/dvc}"
[[ -x "$dvc_bin" ]] || { echo 'Install requirements-deploy.lock into .venv-deploy first.' >&2; exit 1; }
umask 022
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$registry"
python3 infra/scripts/configure.py
export API_IMAGE WEB_IMAGE
compose=(docker compose -f docker-compose.yml -f infra/aws/compose.production.yml)
"${compose[@]}" pull api web mlflow
# Local config contains no credentials. EC2 obtains S3/ECR access from its instance role.
"$dvc_bin" remote add --local --force production "$DVC_REMOTE_URL"
"$dvc_bin" pull -r production ml/artifacts/production
# Validate with the actual published image before restarting the running application.
"${compose[@]}" run --rm --no-deps api python -m pipelines.export_model --validate /app/ml/artifacts/production
expected_bundle="$(python3 -c 'from pathlib import Path; import hashlib; print(hashlib.sha256(Path("ml/artifacts/production/manifest.json").read_bytes()).hexdigest())')"
"${compose[@]}" up --no-build --wait --wait-timeout 240 postgres mlflow
"${compose[@]}" up --no-build --force-recreate --wait --wait-timeout 240 api web
readiness="$(curl --fail --silent --show-error --retry 10 --retry-delay 3 --retry-connrefused "${API_HEALTH_URL:-http://127.0.0.1:8000/ready}")"
python3 -c 'import json,sys; data=json.loads(sys.argv[1]); assert data["status"] == "ready" and data["bundle_id"] == sys.argv[2], "Unexpected production model"; print(json.dumps(data))' "$readiness" "$expected_bundle"
echo 'Deployment passed database, model integrity, inference and release identity checks.'
