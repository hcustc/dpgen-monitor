# DP-GEN Monitor

Configurable monitoring, DeepMD evaluation, plotting, and notifications for complete DP-GEN workflows.

`dpgen-monitor` reads an existing DP-GEN run directory without modifying it. Runtime state, evaluation outputs, figures, and delivery history are written to a separate configured output directory.

## Features

- Tracks DP-GEN iterations and task stages from `record.dpgen` and `dpgen.log`.
- Extracts candidate, failed, and accurate exploration statistics.
- Runs previous-FP absorption evaluation after training is complete.
- Runs same-iteration FP blind-spot evaluation after `post_fp` is complete.
- Stores restart-safe state by iteration, evaluation phase, and model in SQLite.
- Avoids duplicate computation and duplicate notification delivery.
- Produces force-density, best-model parity, absorption-gain, learning-curve, and exploration-trend figures.
- Supports console, Feishu, and generic JSON webhook notifications.
- Logs state transitions while keeping unchanged polling cycles quiet; optional low-frequency heartbeat included.

## Requirements

- Python 3.11 or newer
- A working DeepMD command for model evaluation
- An existing DP-GEN run directory

Install from source:

```bash
python -m pip install .
```

## Configuration

Create a local configuration that Git will ignore:

```bash
cp configs/example.toml configs/local.toml
```

Edit `configs/local.toml` for the DP-GEN run directory, output directory, model format, task thresholds, and notification channels.

Notification credentials should be supplied through environment variables. When using the bundled Conda launcher, its default private environment file is:

```text
runtime/private/dpgen_monitor.env
```

Example variable names:

```bash
FEISHU_BOT_URL=...
FEISHU_APP_ID=...
FEISHU_APP_SECRET=...
```

Protect the file with `chmod 600`. A different file can be selected with `DPGEN_MONITOR_ENV_FILE`.

## Usage

With the installed CLI:

```bash
dpgen-monitor check configs/local.toml
dpgen-monitor sync configs/local.toml
dpgen-monitor once configs/local.toml
dpgen-monitor run configs/local.toml
dpgen-monitor status configs/local.toml
```

The bundled launcher runs the same CLI inside a Conda environment:

```bash
DPGEN_MONITOR_CONDA_ENV=feishu_bot bash dpgen_monitor.sh run configs/local.toml
```

Commands:

- `check`: validate configuration and inspect the DP-GEN directory without writing, evaluating, or notifying.
- `sync`: import observed iteration stages and statistics into SQLite without evaluating or notifying.
- `once`: perform one complete scan, evaluation, and notification cycle.
- `run`: monitor continuously until interrupted.
- `status`: print the persistent state summary.

## Evaluation phases

Each iteration can have two independent evaluation phases:

- `absorption`: evaluates the newly trained model against the previous iteration's FP data.
- `blind_spot`: evaluates the model against same-iteration FP data before those labels are absorbed by later training.

The SQLite primary key includes the phase, so one phase cannot overwrite the other. Existing valid six-column force outputs are reused after restart.

## Project layout

```text
dpgen-monitor/
├── dpgen_monitor/       # Python package
├── configs/             # Public example configuration; *.local.toml is ignored
├── scripts/             # Maintenance and legacy-state migration tools
├── tests/               # Unit tests
├── notebooks/           # Sanitized plotting reference
├── runtime/             # Private state, credentials, outputs, and archives (ignored)
├── dpgen_monitor.sh     # Optional Conda launcher
└── pyproject.toml
```

## Development

Run the test suite:

```bash
python -m unittest discover -v -s tests -p 'test_*.py'
```

Build the distributions:

```bash
uv build
```

## Security

Never commit webhook URLs, application secrets, access tokens, project-local configuration, model files, FP data, SQLite state, or generated outputs. See [SECURITY.md](SECURITY.md) for reporting and credential-rotation guidance.

## License

MIT License. See [LICENSE](LICENSE).
