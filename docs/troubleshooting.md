# Troubleshooting

Documented solutions for issues encountered during development and deployment of this system.

---

### WSL2 Out-of-Memory During Training

**Symptom**: Training process is killed, or the WSL2 terminal terminates without warning.

**Root cause**: TensorFlow's `tf.data.Dataset.cache()` loads the entire dataset into RAM. WSL2 has a default memory cap configured by Windows.

**Resolution**: The `.cache()` call was removed from `preprocessing.py`. The pipeline uses `prefetch(AUTOTUNE)` without in-memory caching. GPU memory growth is set to incremental via `tensorflow.config.experimental.set_memory_growth()` in `build_model.py`.

---

### AWS CLI Not Found in WSL Virtual Environment

**Symptom**: `aws` commands fail with "command not found" despite successful `pip install awscli`.

**Root cause**: The virtual environment's `bin/` directory is not in the PATH, or a system-level AWS CLI conflicts with the venv installation.

**Resolution**: Use the full path to the venv binary: `~/venv_deploy/bin/aws ecs ...`. Alternatively, ensure the virtual environment is activated before running any AWS commands.

---

### Docker Daemon Connection Failure in WSL2

**Symptom**: `docker ps` or `docker build` fails with "Cannot connect to the Docker daemon" or pipe errors.

**Root cause**: Docker Desktop's WSL2 integration is not enabled for the active distribution, or Docker Desktop is not running.

**Resolution**: Open Docker Desktop, navigate to Settings > Resources > WSL Integration, and enable the toggle for your Ubuntu distribution. Restart Docker Desktop if the issue persists.

---

### ECS Task Fails with CannotPullContainerError

**Symptom**: ECS service shows tasks cycling in PENDING/STOPPED states. Logs show `CannotPullContainerError`.

**Root cause**: The Docker image was not fully pushed to ECR before the ECS service update was triggered, or the task definition references an incorrect image URI.

**Resolution**: The `deploy.yml` workflow now includes `aws ecs wait services-stable` after `update-service` to ensure the image is available and the task is healthy before marking the deployment as successful.

---

### Kaggle API Authentication Failure

**Symptom**: `kaggle.api.authenticate()` raises an authentication error even with credentials in `.env`.

**Root cause**: The Kaggle library reads from `~/.kaggle/kaggle.json` by default and may not pick up environment variables set after import.

**Resolution**: In `data_ingestion.py`, the `os.environ["KAGGLE_USERNAME"]` and `os.environ["KAGGLE_KEY"]` assignments are placed before the `from kaggle.api.kaggle_api_extended import KaggleApi` import statement to ensure the library binds correctly at import time.

---

### Model Inference Returns 503 on ECS

**Symptom**: The `/predict` endpoint returns `{"detail": "Model not loaded"}` with HTTP 503.

**Root cause**: The model file was not downloaded from S3 during container startup. This can occur if S3 permissions are missing or the bucket/key path is incorrect.

**Resolution**: Verify that the ECS task execution role has `s3:GetObject` permission on the configured bucket. Check that `AWS_S3_BUCKET` and `MODEL_S3_KEY` in the configuration match the actual S3 path. The `InferencePipeline` in `infer.py` logs the specific S3 error on failure.

---

### Structured Logging Not Producing JSON

**Symptom**: Logs appear as plain text instead of structured JSON in CloudWatch.

**Root cause**: `structlog` detects an interactive terminal (`sys.stderr.isatty()`) and uses `ConsoleRenderer` instead of `JSONRenderer`.

**Resolution**: In containerized environments (ECS), `stderr` is not a TTY, so JSON output is used automatically. For local debugging in JSON format, set `TERM=dumb` before running the application.
