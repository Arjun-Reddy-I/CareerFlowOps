# MLOps study guide: understanding CareerFlowOps from end to end

A plain-words guide to every MLOps idea used in this project. Each topic follows the same pattern: what the idea is, why it exists, how this project does it, and the exact file or command where you can see it.

## 1. The project in plain words

CareerFlowOps helps a student find jobs. It collects job postings from company career pages (Greenhouse, Lever and Ashby), reads a candidate's skills, and ranks the jobs by how well they fit. For every ranked job it shows why: which skills matched, which are missing, and a short explanation. The student then opens the official apply link. The system never applies for anyone.

The parts you can touch:

- A website (React) where the candidate sees jobs, a profile, saved jobs and an application tracker.
- A backend API (FastAPI) that stores jobs in PostgreSQL and serves rankings.
- A ranking model (TF-IDF, a sentence-transformer, or a mix of both) that scores each job against the candidate.
- An MLOps layer that trains, tracks, packages, ships and watches that model.

This guide is about the last item. The website and API are the product. MLOps is how the model inside them is produced and kept healthy.

## 2. What MLOps is, and why this project needs it

A machine learning model is not a normal program. It depends on data, on settings, on the code that trained it and on the libraries it ran with. Change any of these and you get a different model. That creates problems that ordinary software does not have:

| Problem | Plain version | How this project solves it |
| --- | --- | --- |
| It worked on my laptop | Nobody else can get the same model | DVC pipeline plus pinned library files, so anyone can rebuild it |
| Which data made this model? | The data changed and nobody noticed | DVC records a hash of every input and output in `dvc.lock` |
| Which experiment was best? | Results live in someone's notebook | MLflow stores every run with its settings and scores |
| How does the model reach the server? | Copying files by hand, differently each time | A checked, versioned model bundle pulled from S3 |
| Is it still working after release? | The server runs but gives bad answers | Health checks, metrics, alerts, and a server that refuses to use a broken model |
| Can I go back to last week's version? | Old model is gone | Immutable image tags and old model files kept in S3 |

**MLOps** means applying the habits of DevOps (version control, automation, testing, monitoring) to machine learning, so a model can be reproduced, delivered and watched reliably.

### The life of a model in this project

```
 raw job data
     |  DVC stage: ingest
     v
 normalized jobs  --- preprocess ---> training set --- train ---> 3 trained rankers
                                                                      |
                                           evaluate (scores) <--------+
                                                |
                                  register in MLflow (version, alias)
                                                |
                                   export_model -> verified bundle
                                                |
                          dvc push -> S3 (model storage)        git push -> GitHub
                                                                      |
                                           GitHub Actions: test, build images, push to ECR
                                                                      |
                                          deploy to EC2 through Systems Manager
                                                                      |
              EC2 pulls image (ECR) + model (S3), checks it, starts the service
                                                                      |
                        user request -> API -> ranked jobs with explanation
                                                                      |
                           Prometheus + Grafana watch it; Docker restarts crashes
```

### The MLOps parts, and where each lives

| MLOps part | Tool here | Where to look |
| --- | --- | --- |
| Data and model versioning | DVC | `dvc.yaml`, `dvc.lock`, `.dvc/config` |
| Reproducible pipeline | DVC stages | `dvc.yaml`, `pipelines/` |
| Settings as data | `params.yaml` | `params.yaml` |
| Experiment tracking | MLflow | `pipelines/train.py`, `pipelines/evaluate.py` |
| Model registry | MLflow registry | `pipelines/register.py`, `ml/registry/` |
| Model packaging | Export bundle with manifest | `pipelines/export_model.py`, `ml/inference/bundle.py` |
| Containers | Docker, Compose | `infra/docker/`, `docker-compose.yml` |
| Continuous integration | GitHub Actions | `.github/workflows/ci.yml` |
| Continuous delivery | GitHub Actions, ECR, Systems Manager | `.github/workflows/docker-publish.yml`, `infra/aws/` |
| Cloud hosting | AWS (S3, ECR, EC2, IAM) | `infra/aws/`, `docs/remaining/03_AWS_DEPLOYMENT.md` |
| Serving | FastAPI | `apps/api/` |
| Monitoring | Prometheus, Grafana | `infra/monitoring/` |
| Reliability | Docker restart policies, health checks | `docker-compose.yml` |
| Security | OIDC, tokens, HTTPS, CORS | `apps/api/app/config.py`, `infra/aws/policies.py` |

## 3. Three tools, three jobs: Git, DVC and MLflow

Beginners mix these up. Each stores a different kind of thing.

| Tool | Stores | Does not store |
| --- | --- | --- |
| **Git** | Code, small text files, the pointer files `dvc.yaml` and `dvc.lock` | Big data and model files |
| **DVC** | Big files (data, models) in a remote such as S3, plus the recipe to rebuild them | Experiment comparisons |
| **MLflow** | Each experiment run: settings, scores, files, and the model registry | Source code or raw data |

A simple way to remember it: Git is the history of the code, DVC is the history of the data and models, MLflow is the lab notebook of experiments.

Git cannot hold a 90 MB model well, so DVC keeps the file in S3 and Git keeps a tiny pointer to it (a hash). When you check out an old Git commit, the pointer tells DVC which exact file to fetch.

## 4. DVC in depth

### 4.1 What a DVC pipeline is

A DVC pipeline is a list of stages in `dvc.yaml`. Each stage says: run this command, it reads these inputs (`deps`), it produces these outputs (`outs`). DVC then works out what is out of date and reruns only that, much like `make`.

### 4.2 The seven stages in this project

| Stage | What it does | Main output |
| --- | --- | --- |
| `ingest` | Loads a frozen snapshot of real job postings from the three job boards | `data/raw/snapshot.json` |
| `preprocess` | Cleans and normalizes jobs into one common shape | `data/interim/jobs.json` |
| `build_training_set` | Builds the evaluation set: 29 jobs, six fictional candidate queries, relevance labels | `data/processed/dataset.json` |
| `train` | Fits the TF-IDF index and computes sentence-transformer embeddings | `tfidf.joblib`, `embeddings.npy`, `training.json` |
| `evaluate` | Scores all three rankers on validation and test queries, picks the best | `evaluation.json` (a DVC metric) |
| `register` | Records the chosen model in MLflow with a version and the `production` alias | `model.json`, `registry.json` |
| `export_model` | Packs the chosen model into one verified bundle for the server | `ml/artifacts/production/` |

Read the order as a chain: each stage's output is the next stage's input. If you change only the ranking weights, DVC sees that only `train` and the stages after it are affected, and skips the first three.

### 4.3 `params.yaml`: settings kept out of the code

```yaml
seed: 42
ranking: {semantic: 0.40, skill: 0.30, eligibility: 0.15, location: 0.08, freshness: 0.07}
embedding: {model: sentence-transformers/all-MiniLM-L6-v2, revision: 1110a243...}
```

- `seed` makes random steps repeatable.
- `ranking` weights decide how much each score counts in the final match score. They add up to 1.0.
- `embedding` pins the exact pretrained model and its revision, so a later upload to the model hub cannot silently change your results.

The `train` stage lists `params: [seed, ranking, embedding]`. Change a value there and DVC knows `train` must rerun.

### 4.4 `dvc.lock`: the receipt

After a run, `dvc.lock` holds a hash for every input and output of every stage. This is the proof of what produced what. If a hash differs, something changed. Git commits this file, so each Git commit points at one exact set of data and models.

### 4.5 Remotes: where big files live

A DVC remote is the shared storage. This project has two:

- `storage`: a local folder, used for offline work.
- `production`: an S3 bucket (`careerflowops-models-...`), used for the deployed system. It was added with `--local`, so the setting lives in an ignored file and is never committed.

### 4.6 The commands you should know

| Command | Meaning |
| --- | --- |
| `dvc repro` | Rerun the stages whose inputs changed |
| `dvc repro --force` | Rerun everything |
| `dvc status` | Show which stages are out of date |
| `dvc dag` | Draw the stage graph |
| `dvc push -r production` | Upload outputs to the S3 remote |
| `dvc pull -r production` | Download outputs that match the current `dvc.lock` |
| `dvc status -c -r production` | Check that the cache and the remote match |
| `dvc remote list` | Show configured remotes |

### 4.7 Why this matters

Without DVC, "the model" is a file someone emailed. With DVC, "the model" is "the output of commit X with data hash Y", and anyone can rebuild or fetch exactly that.

## 5. Experiment tracking with MLflow

### 5.1 Key words

- **Experiment**: a named group of runs. Here: `careerflowops-ranking`.
- **Run**: one training or evaluation attempt. It stores parameters (settings), metrics (scores), tags (labels such as the Git commit) and artifacts (files).
- **Model registry**: a catalogue of named models with numbered versions. Here: `careerflowops-ranker`, versions 1 to 4.
- **Alias**: a movable label on a version. The alias `production` points at version 4. Moving the alias promotes a different version without renaming anything.

### 5.2 What this project recorded

- One experiment with 12 finished runs, including the three current rankers: `tfidf`, `sentence_transformer` and `hybrid`.
- For each: validation and test MRR, nDCG@5, Precision@5, Recall@5 and inference latency.
- Tags: the dataset split policy, the label provenance, and the Git commit and branch used.

### 5.3 How MLflow is run here

MLflow runs as a service in Docker (`mlflow server`). It keeps its records in a small SQLite database file and its artifacts in a folder, both under `mlflow-data/`. Clients reach artifacts through the server (`mlflow-artifacts:/...` paths), so records do not contain machine-specific folder paths. That is why the whole folder could be copied to the EC2 server and still work.

### 5.4 DVC metrics and MLflow metrics

`evaluation.json` is a DVC metric file, so `dvc metrics show` can print it and compare commits. The same numbers are also logged to MLflow. DVC answers "what did this commit produce?". MLflow answers "how do all my runs compare?".

## 6. Evaluating the model

### 6.1 The metrics, in simple words

The model returns a ranked list. A good list puts the truly relevant jobs at the top.

| Metric | Plain meaning |
| --- | --- |
| **Precision@5** | Of the top 5 jobs shown, what fraction are relevant? |
| **Recall@5** | Of all relevant jobs, what fraction appear in the top 5? |
| **MRR** (mean reciprocal rank) | How high does the first relevant job appear? Rank 1 scores 1.0, rank 2 scores 0.5, and so on, averaged over queries |
| **nDCG@5** | Like precision, but a relevant job at rank 1 counts more than at rank 5. 1.0 is a perfect ordering |
| **Latency** | Milliseconds to produce a ranking |

### 6.2 Validation and test

Queries are split into a validation set (used to choose the model) and a test set (held back to check the choice). Choosing on the test set would be cheating, because the model would then be tuned to the very questions used to judge it.

### 6.3 How the best model is chosen

In `pipelines/evaluate.py` the choice is the highest validation nDCG@5, then the highest Precision@5, then the algorithm name. All three rankers score 1.0 on both metrics, so the name decides, and `tfidf` wins. Latency plays no part.

### 6.4 Honest limits (say these in a viva)

- The benchmark is six fictional queries on 29 real postings, with relevance labels written during development. They are not human-validated.
- Because everything ties, the numbers show that the pipeline works. They do not show how good the ranking is for real users.
- A larger, human-labelled benchmark is the next step.

## 7. Packaging the model: the bundle

### 7.1 Why not just copy `model.pkl`?

Copying a file to a server by hand leads to the wrong version, a corrupt copy, or a model that does not match the code. So the project ships a **bundle**: a folder with the model files and a `manifest.json` that lists every file with its checksum.

### 7.2 Bundle ID

The bundle ID is the SHA-256 hash of `manifest.json`. If any file changes, the manifest changes, and so does the ID. The running server reports it at `/ready`, so you can prove which model is live. For this project it is `ded4d86e...`.

### 7.3 Fail closed

In production the API loads only a verified bundle. If the bundle is missing, damaged or incompatible, the API returns 503 instead of quietly using some other model. Failing loudly is safer than answering wrongly. The result of every request also carries `model_version` and a `fallback_reason`, which stays empty in production.

### 7.4 Model version name

`b3e854956e87-574536b6` is built from two parts: the start of the dataset hash and the start of the Git commit used for training. One short string tells you which data and which code made the model.

## 8. Containers: Docker and Compose

### 8.1 Basics

- An **image** is a frozen package: code, libraries and settings.
- A **container** is a running copy of an image.
- **Docker Compose** starts several containers together from one file (`docker-compose.yml`).

Containers fix "works on my machine": the server runs the same image that CI built and tested.

### 8.2 This project's containers

| Container | Job |
| --- | --- |
| `api` | FastAPI backend; runs database migrations, then serves requests |
| `web` | nginx serving the built React site |
| `postgres` | The database |
| `mlflow` | The experiment tracking server (it reuses the API image) |
| `edge` | Caddy: HTTPS and routing to the API and web containers |
| `prometheus`, `grafana` | Metrics and dashboards |

### 8.3 Two details worth knowing

- The API address is baked into the website at build time (a Vite variable). Changing it means building a new image, not editing a server file.
- Services such as PostgreSQL, MLflow and Grafana listen on the server's loopback address only. They are not reachable from the internet.

## 9. CI/CD: automatic testing and delivery

### 9.1 Words

- **CI** (continuous integration): every change is built and tested automatically.
- **CD** (continuous delivery or deployment): a passing change is packaged and released automatically.
- **CT** (continuous training): retraining automatically when data changes. This project does not do that. Training is run on purpose with `dvc repro`, which is a reasonable choice for a small system.

### 9.2 The workflow

GitHub Actions runs `.github/workflows/docker-publish.yml` on every push to `main`:

1. **Checks** (`ci.yml`): backend (Ruff, pytest, database migration check), frontend (type check, ESLint, Prettier, tests, build, `npm audit`), monitoring (Prometheus config and alert-rule tests).
2. **Build**: build the API and web images and tag them with the full Git commit SHA.
3. **Publish**: push both images to ECR.
4. **Deploy**: tell the EC2 server to update, through AWS Systems Manager.

If any step fails, later steps do not run.

### 9.3 Why tag images with the commit SHA

A tag like `latest` moves, so you never know what is running. A SHA tag points to one commit forever. ECR is set to **immutable tags**, so a tag can never be overwritten. This gives traceability (running image to Git commit) and an easy rollback (deploy an older SHA).

### 9.4 Logging in to AWS without a password

The workflow uses **OIDC**. GitHub proves "this is a run of repository X on branch main" with a short-lived token, and AWS hands back temporary credentials. No AWS key is stored in GitHub. Two IAM roles exist: one that can only push to the two ECR repositories, and one that can only run a command on the tagged server.

### 9.5 What the deploy script does on the server

`infra/aws/deploy.sh`, in order, stopping at the first failure:

1. Check that the checked-out code matches the image tag.
2. Log in to ECR and pull the images.
3. Pull the model bundle from S3 with DVC.
4. Test inference inside the published API image.
5. Start PostgreSQL and MLflow, then the API (which runs migrations) and the web container.
6. Ask `/ready` for the bundle ID and compare it with the bundle that was downloaded.

Step 6 is the proof that the running service uses exactly the intended model.

## 10. The cloud pieces and why each is there

| AWS service | Job | Why this and not something else |
| --- | --- | --- |
| **S3** | Stores the DVC model files | Cheap, durable, versioned object storage |
| **ECR** | Stores Docker images | Private registry with scanning and immutable tags |
| **EC2** | Runs the containers | Simple single server; the system is small and stateful |
| **IAM** | Controls who may do what | Roles with only the permissions each job needs |
| **Systems Manager** | Runs deploy commands and gives access | No SSH key and no open port 22 |
| **Elastic IP** | Fixed public address | The address survives a stop and start |
| **Security group** | Firewall | Only ports 80 and 443 are open |

The instance has an IAM role, so it needs no stored AWS keys to pull images or read the model bucket. IMDSv2 is required, which blocks a common credential-theft trick.

Why not Kubernetes or SageMaker? The project is a small single-host system, and its rules deliberately avoid extra moving parts. For a larger system, ECS or EKS and a managed database would be the next step.

## 11. Serving: how a request becomes a prediction

1. The browser sends `GET /api/jobs/recommended` with a bearer token.
2. Caddy receives it over HTTPS and forwards it to the API.
3. The API checks the token, loads the candidate profile and the active jobs from PostgreSQL.
4. The loaded model bundle scores each job. Five component scores are combined with the `params.yaml` weights: semantic, skill, eligibility, location and freshness.
5. The API returns the top jobs, each with its scores, matched and missing skills, an explanation, the model version and the official apply link.

Two endpoints matter for operations:

- `/health`: is the process alive?
- `/ready`: is it ready to serve? It also reports the model version, the algorithm and the bundle ID.

**Explainable output** is part of the design. A hiring-related ranking must be inspectable, so no result is only a number.

## 12. Monitoring and alerting

### 12.1 Why monitor an ML service

A service can be up and still be wrong or slow. Monitoring shows it.

### 12.2 What this project records

The API exposes metrics at `/metrics`. Prometheus collects them at regular intervals, and Grafana draws them. The main metrics:

| Metric | Meaning |
| --- | --- |
| `careerflowops_http_requests_total` | Requests counted by route and status |
| `careerflowops_http_duration_seconds` | How long requests take |
| `careerflowops_inference_seconds` | How long the model takes to rank |
| `careerflowops_ingestion_runs_total` | Job-board refresh runs and their results |
| `careerflowops_duplicates_detected_total` | Duplicate postings found |

Four alert rules fire for: API unreachable, repeated readiness failures, sustained server errors, and failed or partial ingestion. The alert rules are tested in CI.

### 12.3 What is not built

- Nobody receives the alerts yet (no email or chat receiver).
- There is no drift detection and no host or disk monitoring.

### 12.4 Drift, in simple words

- **Data drift**: the inputs change. For example, resumes in a new city or new job titles.
- **Concept drift**: the meaning of "a good match" changes.

To detect drift you compare the live input and score distributions with the training ones, and alert on a threshold. A tool such as Evidently does this. It is a sensible next step, not part of this project today.

## 13. Reliability: restart, health checks and rollback

### 13.1 Restart policy

Every container has `restart: unless-stopped`. If a container exits by accident, Docker starts it again. After a server reboot the containers come back too.

Two things to know:

- Docker counts `docker kill` as a manual stop, so the container stays down. To test crash recovery you must kill the server process inside the container, which is what was done here. The API came back in about 4 seconds.
- A restart policy does not notice a process that is running but broken. That is what health checks and alerts are for.

### 13.2 Health checks

Compose checks `/ready` every few seconds, so the API and web containers wait for a healthy database and API before they start.

### 13.3 Rollback

Deploy the older Git commit. Its checkout restores the matching `dvc.lock`, so the older image and the older model come back together. The limit: database changes are not undone automatically.

## 14. Security inside an MLOps system

- **No secrets in Git.** `.env` is ignored. Server secrets are generated on the server.
- **No stored cloud keys in CI.** OIDC gives temporary credentials.
- **Least privilege.** Each role can do only its own job.
- **HTTPS only in production.** The API refuses weak tokens, SQLite, a missing model and non-HTTPS origins.
- **Separate tokens** for candidate and admin pages.
- **CORS** allows only the site's own address.
- **Private services.** Databases and dashboards listen on loopback only.
- **Compliance.** Only official public job endpoints are used. Applications are never submitted automatically.

## 15. Traceability: one run, five identities

When something looks wrong you need to trace back. Each identity answers one question:

| Identity | Example | Answers |
| --- | --- | --- |
| Git commit | `380800d...` | Which code? |
| Dataset hash | `b3e85495...` | Which data? |
| MLflow run ID | `9023340c...` | Which experiment run? |
| Bundle ID | `ded4d86e...` | Which exact model files are live? |
| Image digest | `sha256:ad8f43fb...` | Which exact container image is running? |

## 16. Walkthrough: what happens when you change a ranking weight

1. Edit `ranking.skill` in `params.yaml`.
2. `dvc status` reports that `train` and later stages are out of date.
3. `dvc repro` reruns them. New scores appear in `evaluation.json`, a new MLflow run is logged, and a new model version is registered.
4. `export_model` writes a new bundle with a new bundle ID.
5. `dvc push -r production` uploads the new files to S3.
6. You commit the changed `params.yaml` and `dvc.lock`, then push to GitHub.
7. GitHub Actions tests the code, builds new images tagged with the new SHA and pushes them to ECR.
8. The deploy job tells EC2 to update. EC2 pulls the images and the new bundle, tests it and restarts the API.
9. `/ready` now shows the new bundle ID. Prometheus keeps recording, so you can compare latency before and after.

This is the whole MLOps loop. The project's `make release` command automates the local half of it: the checks, `dvc repro`, the upload to S3 and the Git push.

## 17. Problems that happened, and what they teach

| Problem | Lesson |
| --- | --- |
| Production refused a plain HTTP address | Safety checks that fail closed are working as designed. Fix the setup, not the check. |
| GitHub could not log in to AWS | Trust policies must match the exact identity GitHub sends. Read the error and the token claims. |
| `docker kill` did not trigger a restart | Know what a feature really covers before you rely on it, and test it. |
| The Free plan rejected the planned instance type | Cloud accounts have limits. Check them early. |
| A new MLflow server was empty | Experiment history lives in a store, not in the code. Plan to move it. |
| Pushes did not start CI on a fork | New repositories can behave differently. Verify the pipeline with a real push. |

## 18. Quick glossary

- **Artifact**: a file produced by a run or a stage (model, report, data).
- **Bundle**: the packaged model folder with its manifest.
- **CI/CD**: automatic testing, then automatic delivery.
- **Container**: a running image.
- **DVC remote**: shared storage for big files.
- **Immutable tag**: an image tag that cannot be overwritten.
- **Inference**: using a trained model to make predictions.
- **Lineage**: the traceable chain from data and code to a deployed model.
- **Registry**: the catalogue of model versions.
- **Reproducibility**: getting the same result from the same inputs.
- **SHA**: the unique ID of a Git commit.

## 19. Self-test: questions you should be able to answer

1. What problem does MLOps solve that ordinary DevOps does not?
2. What is the difference between Git, DVC and MLflow? What does each store?
3. What does `dvc.lock` contain and why is it committed?
4. Name the seven DVC stages in order. Which one always reruns, and why?
5. Why is the embedding model's revision pinned in `params.yaml`?
6. What is the difference between validation and test data?
7. Explain MRR and nDCG@5. How was the best model chosen, and why is that choice weak here?
8. What is a bundle ID and how does the deploy script use it?
9. What does "fail closed" mean for the API?
10. Why tag images with the commit SHA and make the tags immutable?
11. How does GitHub deploy to AWS without a stored password?
12. List the steps of a deployment and say which one proves the right model is running.
13. Why is `docker kill` a poor crash test?
14. Which monitoring pieces exist, and which are missing?
15. What is data drift? How would you detect it here?
16. How do you roll back a bad release, and what can you not roll back?
17. Walk through what happens after you change a value in `params.yaml`.
18. What is continuous training, and why does this project not do it?

## 20. Command cheat sheet

```bash
# Data and model pipeline
dvc dag                        # show the stage graph
dvc status                     # what is out of date
dvc repro                      # rerun changed stages
dvc push -r production         # upload to S3
dvc pull -r production         # download from S3
dvc status -c -r production    # is the remote in sync

# Pipeline results and runs
gh run list --repo Arjun-Reddy-I/CareerFlowOps --limit 5
aws ecr describe-images --repository-name careerflowops-api --profile careerflow --region ap-south-1

# The running system
curl https://15-252-197-53.sslip.io/ready      # model version and bundle ID
docker compose ps                              # containers and health (on the server)
docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' <container>
```
