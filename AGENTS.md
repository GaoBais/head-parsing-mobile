# Agent Instructions

## Environment Roles

This repository is developed locally on Windows and trained later on a Linux CUDA server.

Local Windows environment:

- Use this machine for code development, static checks, lightweight CPU smoke tests, and packaging only.
- Do not assume an NVIDIA GPU is available on Windows.
- Do not run CUDA training or CUDA benchmark tasks locally.
- Python is installed on the host and should be run from the host environment, not from a sandbox-only bundled runtime.
- Current verified host Python:
  - `Python 3.13.7`
  - `C:\Users\hengx\AppData\Local\Programs\Python\Python313\python.exe`

Linux server environment:

- Use the server for full training, CUDA validation, distillation runs, and performance-critical experiments.
- Code will be uploaded to the server, for example with `scp`, after local development changes are ready.
- Server Python should use a dedicated virtual environment with a PyTorch/CUDA-compatible Python version, preferably Python 3.10 or 3.11.

## Python Execution Policy

- Prefer host Python on this Windows machine for local script execution.
- Do not treat the bundled Codex Python runtime as the project runtime.
- The bundled runtime may be used only for emergency syntax checks when host Python is unavailable, and results must be labeled as such.
- Before running project Python scripts locally, verify `python --version` and `where python` if there is any ambiguity.

## Local Validation Scope

Allowed locally:

- Syntax checks.
- CLI help checks.
- Dataset path and metadata inspection.
- Small CPU-only smoke tests.
- Unit tests that do not require CUDA.

Not expected locally:

- Full model training.
- CUDA AMP validation.
- GPU latency benchmark.
- TFLite/Core ML device benchmark.

## Progress Tracking

- Keep implementation progress in `docs/progress.md`.
- Update the task table when a task changes state.
- Add a short change-log entry for meaningful implementation changes.
