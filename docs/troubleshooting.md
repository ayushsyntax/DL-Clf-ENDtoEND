# 🛠️ Real-World Troubleshooting (Hard-Earned Lessons)

This section isn't just theory—it's a collection of the actual "gotchas" we hit while building this end-to-end pipeline in WSL2 and deploying to AWS.

### 1. The "Where is my AWS CLI?" Mystery (WSL2 Pathing)
**Symptom:** You run `aws configure` in WSL and it works, but later, your scripts can't find the `aws` command.
**What happened:** We found that the system-installed AWS CLI and the one inside your specific `venv` can conflict.
**Fix:** Always use the full path to the virtual environment binary if a direct call fails. Example: `~/venv_deploy/bin/aws ecs ...`.

### 2. WSL2 Out-of-Memory (OOM) Death
**Symptom:** Training starts, then the entire WSL terminal just... disappears or prints "Killed".
**What happened:** TensorFlow's default `tf.data` behavior tried to cache everything in RAM. WSL2 by default has a memory cap set by Windows.
**Fix:** We explicitly disabled the `.cache()` step in `preprocessing.py` and set `configure_gpu_memory()` to use incremental growth instead of grabbing the whole 4GB of a GTX 1650 immediately.

### 3. Docker "Pipe/Daemon" Connection Errors
**Symptom:** `docker ps` or `docker build` fails with "The system cannot find the file specified" or "open //./pipe/dockerDesktopLinuxEngine".
**What happened:** This is usually because Docker Desktop's WSL2 integration isn't toggled ON for your specific distribution, or the Docker Desktop app itself isn't running in Windows.
**Real Solution:** Check Docker Desktop Settings → Resources → WSL Integration. Ensure your distro is checked. If it still fails, restart Docker Desktop.

### 4. The ECR "Image Not Found" Deployment Loop
**Symptom:** ECS deployment shows "PRIMARY" but the task keeps stopping with `CannotPullContainerError`.
**What happened:** This usually happens if the GitHub Action tries to deploy before the Docker image is fully pushed/tagged in ECR.
**Fix:** Our `deploy.yml` now uses `aws ecs wait services-stable` to ensure AWS actually sees and stabilizes the target image before we consider it a success.

### 5. Kaggle API Key Environment Drift
**Symptom:** `kaggle.api.authenticate()` fails even after setting the API key.
**What happened:** The Kaggle library looks for `~/.kaggle/kaggle.json`. When running in headless scripts, it sometimes misses standard environment variables.
**Fix:** In `data_ingestion.py`, we manually inject `os.environ['KAGGLE_USERNAME']` BEFORE importing the Kaggle API to ensure it binds correctly.

---
> [!TIP]
> If all else fails, check the structured logs! We use `structlog` (JSON output) in `src/common/logging.py` so you can pipe the output to a file and actually see what the inference engine is thinking.
