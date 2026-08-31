# Security Policy

## Sensitive data

Do not commit notification webhook URLs, application secrets, access tokens, local DP-GEN paths, model files, FP datasets, SQLite databases, logs, or generated evaluation artifacts.

Local configuration should use `configs/*.local.toml`. Credentials and runtime data should remain under `runtime/`, or outside the repository entirely. Both locations are excluded from version control by default.

## Reporting a vulnerability

Report security issues privately to the repository owner instead of opening a public issue. Include the affected version, reproduction steps, and potential impact without attaching live credentials or private scientific data.

If a credential may have been exposed, revoke or rotate it immediately before removing it from files or Git history.
