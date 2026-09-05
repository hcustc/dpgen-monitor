# DP-GEN Monitor

Configurable monitoring, DeepMD evaluation, plotting, and notifications for complete DP-GEN workflows.

`dpgen-monitor` reads an existing DP-GEN run directory without modifying it. Runtime state, evaluation outputs, figures, and delivery history are written to a separate configured output directory.

## Features

- Tracks DP-GEN iterations and task stages from `record.dpgen` and `dpgen.log`.
- Extracts candidate, failed, and accurate exploration statistics.
- Runs previous-FP absorption evaluation after training is complete.
- Runs same-iteration FP blind-spot evaluation after `post_fp` is complete.
- Optionally replays unselected trajectory candidates with the newly trained
  model committee, without running new MD.
- Stores restart-safe state by iteration, evaluation phase, and model in SQLite.
- Detects iteration rollback/recreation and isolates regenerated results from stale state.
- Avoids duplicate computation and duplicate notification delivery.
- Produces force-density, best-model parity, absorption-gain, learning-curve, and exploration-trend figures.
- Supports console, Feishu, and generic JSON webhook notifications.
- Logs state transitions while keeping unchanged polling cycles quiet; optional low-frequency heartbeat included.

## Requirements

- Python 3.11 or newer
- A working DeepMD command for model evaluation
- An existing DP-GEN run directory
- Committee replay additionally requires a DP-GEN build or preprocessing step
  that emits an atomic candidate-selection JSONL report; the required schema is
  documented below. Standard runs without that report can use all other monitor
  features but cannot enable committee replay.

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
dpgen-monitor tui configs/local.toml
dpgen-monitor proposals configs/local.toml
dpgen-monitor approve configs/local.toml PROPOSAL_ID
dpgen-monitor reject configs/local.toml PROPOSAL_ID --note "reason"
dpgen-monitor apply configs/local.toml PROPOSAL_ID
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
- `tui`: open the interactive overview, delivery history, and safe settings editor.
- `proposals`: list durable `model_devi_jobs` proposals and their evidence.
- `approve` / `reject`: record a human decision without editing DP-GEN files.
- `apply`: append one approved, validated job; it never starts DP-GEN.

## Terminal UI

The TUI is operationally read-only: it never runs `dp test`, submits jobs, or
sends notifications. The settings page can update these numeric TOML fields:

- scan and heartbeat intervals;
- statistics task and start iteration;
- evaluation start iteration;
- absorption and blind-spot readiness tasks.

On terminals at least 124 columns wide, the overview automatically switches to
a two-pane layout with iteration history on the left and selected-iteration
details on the right. Narrower terminals use a compact single-pane layout.

Press `Enter` to edit a selected setting and `w` to save. Saving validates a
temporary configuration, writes `<config>.bak`, and atomically replaces the
TOML file. Notification credentials are neither displayed nor modified. A
separately running `dpgen-monitor run` process must be restarted to load the
new values.

## Evaluation phases

Each iteration can have two independent evaluation phases:

- `absorption`: evaluates the newly trained model against the previous iteration's FP data.
- `blind_spot`: evaluates the model against same-iteration FP data before those labels are absorbed by later training.

Committee replay is a separate, opt-in evaluation because it measures
committee disagreement on unlabeled trajectory frames rather than error against
DFT labels:

- `committee_replay`: after iter.N training, deterministically samples wholly
  unselected candidate frames from iter.N-1 (or other configured source
  offsets), builds a compact DeepMD/Numpy holdout from the saved
  `traj/<step>.lammpstrj` files, and runs all newly trained models together with
  `dp model-devi`.

The replay reports how many previously unselected candidate atoms moved into
the low-deviation (`accurate`) bucket, remained candidates, or worsened to
failed, plus all-atom candidate/failed ratios for the fixed holdout. Here
`accurate` means committee agreement under the configured thresholds, not
accuracy against DFT labels. Replay does not integrate equations of motion and
therefore does not replace a final on-policy MD validation.

Replay is disabled by default because it can use a GPU. Configure it explicitly:

```toml
[committee_replay]
enabled = true
command = ["conda", "run", "-n", "deepmd-3.2", "dp"]
executor_profile = "local_gpu"
dispatcher_check_interval = 30
cpu_per_node = 4
gpu_device = 0
max_total_frames = 2048
model_ids = ["000", "001", "002", "003"]
model_pattern = "graph.{model_id}.pt2"
source_offsets = [1, 2]
candidate_manifest = "02.fp/candidate_selection.000.jsonl"
exclude_selected = true
max_frames_per_task = 256
time_bins = 5
relative = 1.0
f_trust_lo = 0.15
f_trust_hi = 0.30
start_iteration = 29
ready_task = 2
```

`candidate_manifest` is an explicit compatibility boundary with DP-GEN. Each
non-empty JSONL row must contain `task`, `step`, `atom_index`,
`model_deviation`, and boolean `selected`. `atom_index` is zero-based and must
be a non-negative integer; frame-level rows with `atom_index = null` are not
supported because they cannot provide atom-resolved absorption evidence. The
report must include every selected frame, plus the unselected candidates
intended for replay, so the monitor cannot mistake an FP-labeled frame for
holdout data. For multi-system runs, point this setting at the report for the
system being audited and use a separate monitor configuration/output directory
for each additional system.

Replay execution goes through DPDispatcher 1.x using the sole registered
`local_gpu` profile. An immutable semantic request contains iteration IDs,
holdout paths, model paths, and frame counts, but no command. A controller
validates iteration offsets, exact committee model paths, output containment,
per-task limits, and `max_total_frames`; only then does it construct the fixed
`dp model-devi` task. The profile uses `Shell` with `LazyLocalContext`, one GPU,
and `CUDA_VISIBLE_DEVICES=gpu_device`.

There is intentionally no generic SSH command, remote target, transfer command,
or scheduler header in this interface. A future cluster profile must be added as
an administrator-owned, named DPDispatcher profile rather than supplied by an
Agent. DPDispatcher submission metadata and the validated request are retained
with the replay output for audit and recovery. Submission is non-blocking: the
local process keeps running under DPDispatcher, while later monitor scans recover
the same request and query its state. Only one replay is allowed to own the local
GPU at a time.

The holdout is balanced deterministically over model-deviation tasks and time
bins. Results are keyed by model iteration, source iteration, and both
iterations' recovery generations. Generated datasets, raw deviation output,
logs, and `summary.json` are stored below:

```text
evaluations/iter.000029/committee_replay/source.iter.000028/
```

The SQLite primary key includes the phase, so one phase cannot overwrite the
other. Existing valid six-column force outputs are reused after restart.

## Human-approved parameter proposals

This is a deterministic workflow, not an LLM Agent. It uses DP-GEN's natural
stage gate: when `model_devi_jobs` contains entries `0..N-1`, DP-GEN can train
iter.N and stop at `post_train` because no exploration entry exists for N.

Enable the workflow with an explicit active parameter file:

```toml
[parameter_proposals]
enabled = true
parameter_file = "/path/to/run/param.recovery.yaml"
strategy = "repeat_last"
start_iteration = 29
required_task = 2
max_nsteps = 25000000
```

A proposal is created only when all of these conditions hold:

- the target iteration equals the current length of `model_devi_jobs`;
- `record.dpgen` and the iteration snapshot are at exactly task 02;
- `committee_replay.source_offsets` is exactly `[1]`, and that immediately
  previous iteration replay is complete;
- the final job has the first-version supported shape: NVT with only
  `sys_idx`, `temps`, `trj_freq`, `nsteps`, `ensemble`, and `_idx`;
- `nsteps` does not exceed `max_nsteps`.

The `repeat_last` rule copies the last job's physical settings and changes only
`_idx` to the gated iteration. Replay summaries are attached as evidence for
human review; they do not silently rewrite the rule or its output.

Parameter decisions deliberately use only iter.N-1. Older sources may be useful
for a separate regression or reproducibility audit, but they neither contribute
to nor block the next `model_devi` proposal.

Review is deliberately split from mutation:

```bash
bash dpgen_monitor.sh proposals configs/local.toml
bash dpgen_monitor.sh approve configs/local.toml PROPOSAL_ID
bash dpgen_monitor.sh apply configs/local.toml PROPOSAL_ID
```

`approve` changes SQLite state only. `apply` rechecks the parameter-file SHA-256,
the exact post-train record, absence of `01.model_devi`, the append-only list
length, and the repeat-last whitelist. It writes a timestamped backup, performs
an atomic replacement, and verifies that no other YAML value changed. It does
not invoke `dpgen run`; resuming DP-GEN remains a separate human action.

## Rollback and recovery

If a DP-GEN recovery removes later iterations or reuses an iteration number,
the monitor detects the directory/stage regression and invalidates only the
superseded iteration state. Regenerated model outputs are written below a
generation-specific directory, so previous files remain available without
being mistaken for current results. Statistics and status summaries include
only iteration directories that are active in the current run.

Delivery history is retained across recoveries. Notifications are deduplicated
by their stable event identity and content digest: identical regenerated
results are not sent twice, while changed results remain eligible for delivery.
When a late scan discovers absorption results and exploration statistics at the
same time, absorption notifications are emitted first.

The `run` and `once` commands share an exclusive lock in the configured output
directory, preventing concurrent processes from evaluating or notifying the
same iteration. During `run`, `Ctrl-C` cancels the active evaluation, preserves
any valid output already produced, suppresses cancellation-as-failure alerts,
and stops before starting the next model.

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
