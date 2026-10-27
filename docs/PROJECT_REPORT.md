# CareerFlowOps: deploying a job-ranking model with an MLOps pipeline on AWS

End-semester MLOps evaluation. Deployment date: 27 October 2026. Repository: `Arjun-Reddy-I/CareerFlowOps`. Live system: https://15-252-197-53.sslip.io

## 1. What was built

CareerFlowOps collects job postings from public employer job boards, ranks them against a candidate's skills and explains each ranking. The application side is a React portal and a FastAPI backend on PostgreSQL. This report covers the MLOps side: how the ranking model is versioned, tracked, packaged, delivered to a server and watched while it runs.

The work follows two supplied diagrams, which are kept in the repository root: `mlops_pipeline_overview.png` (data to prediction) and `ec2_deployment_flow.png` (image and model delivery to EC2).

## 2. Data and model

Postings come from the official public endpoints of Greenhouse, Lever and Ashby. Every job keeps its source name, source job ID, source URL and apply URL, and the raw payload is stored apart from the normalized record. Duplicates are detected without deleting provenance.

Three rankers are trained and compared: TF-IDF, a pretrained sentence-transformer (`all-MiniLM-L6-v2`, pinned to a fixed revision) and a hybrid of the two. A result carries five component scores (semantic, skill, eligibility, location, freshness), the matched and missing skills, a text explanation, the model version and a fallback reason. The fallback reason is empty in production, because the server refuses to answer from any model other than the verified one.

The evaluation set is small: 29 frozen real postings and six fictional candidate queries, with relevance labels written during development. They are not human-validated.

## 3. Pipeline (CO2)

A DVC pipeline of seven stages reproduces the model: ingest, preprocess, build_training_set, train, evaluate, register and export_model. The last stage writes a self-contained bundle with a manifest and checksums. DVC pushes the bundle to an S3 bucket (21 objects, 92,630,923 bytes). Git holds code and the small pointer files `dvc.yaml` and `dvc.lock`.

A push to `main` starts a GitHub Actions workflow with three groups of jobs:

1. Checks. The backend job runs Ruff, pytest and the Alembic migration check. The frontend job runs type checking, ESLint, Prettier, tests, a production build and `npm audit`. The monitoring job validates the Prometheus configuration and tests the alert rules.
2. Build. The API and web images are built and tagged with the full Git commit SHA. ECR repositories are set to immutable tags, so a published tag cannot be overwritten.
3. Deploy. The images are pushed to ECR and the host is updated through AWS Systems Manager.

GitHub reaches AWS without stored keys. The workflow asks for a short-lived OIDC token and exchanges it for temporary credentials. Two IAM roles exist, each trusted only for this repository's `main` branch: one that can push to the two ECR repositories, and one that can only send a shell command to the instance tagged `CareerFlowOps`.

On the host, the deploy script stops the API and web containers, pulls both images from ECR, pulls the model bundle from S3, validates inference inside the published API image, starts PostgreSQL and MLflow, starts the API and web, and compares the running bundle ID with the downloaded manifest. A failed check stops the deployment. The deployment run for the final code is [run 40569780650](https://github.com/Arjun-Reddy-I/CareerFlowOps/actions/runs/40569780650), started by a push, with all five jobs passing.

## 4. Cloud services (CO1)

| Service | Use |
| --- | --- |
| S3 | DVC remote holding the model bundle. Versioned, encrypted, public access blocked. |
| ECR | Two repositories, `careerflowops-api` and `careerflowops-web`. Scan on push, immutable tags. |
| EC2 | One `m7i-flex.large` instance (2 vCPU, 8 GiB), Ubuntu 24.04, 40 GiB encrypted disk, IMDSv2 required. |
| IAM and OIDC | Keyless CI, plus an instance role that can pull images and read the model prefix only. |
| Systems Manager | Deployment and administration without SSH. Port 22 is closed. |
| Elastic IP, security group | Stable address; only ports 80 and 443 are open. |

The host runs seven containers from Docker Compose: API, web, PostgreSQL, MLflow, Caddy, Prometheus and Grafana. Caddy terminates HTTPS with a Let's Encrypt certificate and redirects port 80. Everything else listens on the loopback interface only.

The planned instance type, `t3.medium`, was rejected because the account is on the AWS Free plan, which only launches eligible types. `m7i-flex.large` was chosen as an eligible type with 8 GiB of memory.

## 5. Tracking, monitoring and recovery (CO3)

MLflow holds one experiment, `careerflowops-ranking`, with 12 finished runs, and a registered model `careerflowops-ranker` with versions 1 to 4. The alias `production` points to version 4. The tracking store was created during training on the development machine and copied once to the EC2 server, so the history there is the original one. The saved metrics of the three current rankers are:

| Ranker | Validation MRR | Validation nDCG@5 | Test MRR | Test nDCG@5 | Test latency (ms) |
| --- | --- | --- | --- | --- | --- |
| tfidf | 1.0 | 1.0 | 1.0 | 1.0 | 26.94 |
| sentence_transformer | 1.0 | 1.0 | 1.0 | 1.0 | 30.88 |
| hybrid | 1.0 | 1.0 | 1.0 | 0.7426 | 29.00 |

The selection rule in `pipelines/evaluate.py` compares validation nDCG@5, then Precision@5, then the algorithm name. All three rankers tie on the first two, so the name decides and TF-IDF is chosen. Latency is not part of the rule. These numbers show that the pipeline works end to end. With six queries they cannot separate the rankers and say nothing reliable about accuracy on real users.

Monitoring uses Prometheus and Grafana. The API exports request counts and durations, inference time, ingestion runs and detected duplicates. Prometheus scrapes it, and the target `careerflowops-api` reports up. Four alert rules cover an unreachable API, repeated readiness failures, sustained server errors and failed or partial ingestion.

Recovery was tested by killing the server process inside the API container. Docker restarted the container without any manual step (restart count 1), and the readiness endpoint answered again after about four seconds, including through the public HTTPS address. Killing the container with `docker kill` does not work as a test, since Docker treats it as a manual stop and leaves the container down.

## 6. Results of the deployment checks

| Check | Result |
| --- | --- |
| Readiness endpoint | Status `ready`, model version `b3e854956e87-574536b6`, algorithm `tfidf`, bundle `ded4d86e980bff8216b4629df8d2c8106888d9fab4bd78c69c6c9f52058d73e2` |
| Images in ECR for the deployed commit | API `sha256:ad8f43fb328c79b5cd745d01f9ec37f7501fa93744ace964207286a100c0fcbb`, web `sha256:cacc3e35bf5a860ce8972442a2ffd36146cf41ea0d75498d542a57915e312542` |
| Live ingestion on the server | 1,052 postings fetched (714 Greenhouse, 306 Lever, 32 Ashby), 7 duplicates, no source errors, 1,045 jobs stored |
| Request without a token | HTTP 401 |
| Request with the candidate token | Ranked jobs, each with an explanation and an apply link; model version equal to the readiness endpoint; no fallback |
| Cross-origin request from another site | Refused; only the site's own origin is allowed |
| Certificate | Let's Encrypt, valid until 25 January 2027 |

The candidate used in the checks is a fictional profile with six skills and a one-line summary. No resume was uploaded.

## 7. Problems met and how they were resolved

1. The production API refused to start with a plain HTTP address, because the project rejects non-HTTPS origins in production. The check was kept. The site is served over HTTPS on a name derived from the IP address (`15-252-197-53.sslip.io`), which avoids buying a domain.
2. GitHub's login to AWS was rejected. The repository uses GitHub's newer subject format, which includes the owner and repository IDs, and the trust policy only listed the older format. Both formats are now accepted.
3. On the new fork, manual workflow runs worked but pushes did not start the workflow. Setting the repository's Actions permission explicitly through the GitHub API fixed it. The cause was not investigated further.
4. The runbook's restart test used `docker kill` and could not pass, for the reason given in section 5. The test and the documentation were corrected.
5. A new MLflow server starts empty, so the real run history was copied across from the development machine.

## 8. Limitations

- One shared candidate token and one admin token guard the whole site. There are no per-user accounts, so the deployment suits a single demonstration candidate.
- Alert rules exist but nothing receives them: no email or chat target is configured.
- There is no statistical drift monitoring and no host or disk metrics.
- The system runs on a single host. Database migrations are not rolled back automatically, so rollback of the application and model is a redeploy of an earlier commit.
- The instance was never stopped and started, so recovery after a reboot is expected from the restart policies but not tested.
- The MLflow history on the server is a one-time copy, and the running API does not read from it; it serves the bundle delivered through DVC.

## 9. Reproducing and removing the deployment

The commands, the observed values and the differences from the original runbook are in `docs/remaining/03_AWS_DEPLOYMENT.md`, section 11. The helper `infra/aws/demo_access.ps1` shows the host status, prints the generated tokens on the local console and opens the private Grafana, Prometheus and MLflow pages through Systems Manager. The same section lists the teardown steps: terminate the instance, release the Elastic IP, delete the ECR repositories, empty and delete the versioned bucket, and remove the IAM roles, the OIDC provider and the deployer's access key. The instance and its storage use account credits until then.
