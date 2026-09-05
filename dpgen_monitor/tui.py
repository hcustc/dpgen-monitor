from __future__ import annotations

from dataclasses import dataclass
import curses
import locale
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Mapping
import unicodedata

from .config import MonitorConfig, load_config
from .dpgen import DpgenObserver, IterationSnapshot, StatisticsInspection
from .state import StateStore


@dataclass(frozen=True)
class SettingSpec:
    section: str
    key: str
    label: str
    minimum: int
    maximum: int
    help_text: str

    @property
    def identity(self) -> str:
        return f"{self.section}.{self.key}"


SETTINGS = (
    SettingSpec(
        "project", "check_interval", "扫描间隔", 1, 86400,
        "秒；检查 DP-GEN 目录的频率",
    ),
    SettingSpec(
        "project", "heartbeat_interval", "心跳间隔", 0, 604800,
        "秒；0 表示关闭心跳",
    ),
    SettingSpec("dpgen", "statistics_task", "统计就绪 task", 0, 99, "通常为 task 06"),
    SettingSpec(
        "dpgen",
        "statistics_start_iteration",
        "统计起始迭代",
        0,
        999999,
        "更早迭代不产生新通知",
    ),
    SettingSpec(
        "evaluation",
        "start_iteration",
        "评估起始迭代",
        0,
        999999,
        "更早迭代不运行 dp test",
    ),
    SettingSpec(
        "evaluation",
        "absorption_ready_task",
        "吸收评估 task",
        0,
        99,
        "训练完成后的触发阶段",
    ),
    SettingSpec(
        "evaluation",
        "blind_spot_ready_task",
        "盲区评估 task",
        0,
        99,
        "建议为 post_fp 完成后的 task 08",
    ),
)


def config_setting_values(config: MonitorConfig) -> dict[str, int]:
    return {
        "project.check_interval": config.project.check_interval,
        "project.heartbeat_interval": config.project.heartbeat_interval,
        "dpgen.statistics_task": config.dpgen.statistics_task,
        "dpgen.statistics_start_iteration": config.dpgen.statistics_start_iteration,
        "evaluation.start_iteration": config.evaluation.start_iteration,
        "evaluation.absorption_ready_task": config.evaluation.absorption_ready_task,
        "evaluation.blind_spot_ready_task": config.evaluation.blind_spot_ready_task,
    }


def validate_setting_values(values: Mapping[str, int]) -> None:
    for spec in SETTINGS:
        if spec.identity not in values:
            raise ValueError(f"缺少设置 {spec.identity}")
        value = values[spec.identity]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{spec.label} 必须是整数")
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(
                f"{spec.label} 必须在 {spec.minimum}–{spec.maximum} 之间"
            )


def _render_toml_settings(text: str, values: Mapping[str, int]) -> str:
    validate_setting_values(values)
    targets = {(spec.section, spec.key): spec for spec in SETTINGS}
    found: set[tuple[str, str]] = set()
    seen_sections: set[str] = set()
    section = ""
    rendered: list[str] = []
    preferred_newline = "\r\n" if "\r\n" in text else "\n"

    def append_missing(current_section: str) -> None:
        missing = [
            spec
            for spec in SETTINGS
            if spec.section == current_section
            and (spec.section, spec.key) not in found
        ]
        if not missing:
            return
        if rendered and not rendered[-1].endswith(("\n", "\r")):
            rendered[-1] += preferred_newline
        for spec in missing:
            rendered.append(
                f"{spec.key} = {values[spec.identity]}{preferred_newline}"
            )
            found.add((spec.section, spec.key))

    for line in text.splitlines(keepends=True):
        array_section_match = re.match(
            r"^\s*\[\[([^]]+)\]\]\s*(?:#.*)?(?:\r?\n)?$",
            line,
        )
        section_match = re.match(r"^\s*\[([^]]+)]\s*(?:#.*)?(?:\r?\n)?$", line)
        if array_section_match or section_match:
            append_missing(section)
            if array_section_match:
                # Editable values live only in ordinary top-level tables.
                section = ""
            else:
                assert section_match is not None
                section = section_match.group(1).strip()
                seen_sections.add(section)
            rendered.append(line)
            continue

        replacement = None
        for target_section, key in targets:
            if section != target_section:
                continue
            line_newline = (
                "\r\n"
                if line.endswith("\r\n")
                else "\n" if line.endswith("\n") else ""
            )
            body = line[: -len(line_newline)] if line_newline else line
            match = re.match(
                rf"^(\s*{re.escape(key)}\s*=\s*)[^#]*?(\s*(?:#.*)?)$",
                body,
            )
            if match:
                replacement = (
                    f"{match.group(1)}{values[f'{target_section}.{key}']}"
                    f"{match.group(2)}{line_newline}"
                )
                found.add((target_section, key))
                break
        rendered.append(replacement if replacement is not None else line)

    append_missing(section)
    for target_section in dict.fromkeys(spec.section for spec in SETTINGS):
        if target_section in seen_sections:
            continue
        if rendered and not rendered[-1].endswith(("\n", "\r")):
            rendered[-1] += preferred_newline
        if rendered and rendered[-1].strip():
            rendered.append(preferred_newline)
        rendered.append(f"[{target_section}]{preferred_newline}")
        append_missing(target_section)

    missing = set(targets).difference(found)
    if missing:  # Defensive: every editable field must be replaced or inserted.
        names = ", ".join(
            f"{target_section}.{key}"
            for target_section, key in sorted(missing)
        )
        raise ValueError(f"无法写入可编辑字段: {names}")
    return "".join(rendered)


def save_toml_settings(config_path: Path, values: Mapping[str, int]) -> Path:
    path = config_path.expanduser().resolve()
    if path.suffix.lower() != ".toml":
        raise ValueError("TUI 当前只支持保存 TOML；YAML 配置仍可只读查看")
    original = path.read_text(encoding="utf-8")
    rendered = _render_toml_settings(original, values)
    if rendered == original:
        return path.with_name(f"{path.name}.bak")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".toml", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(path.stat().st_mode & 0o777)
        load_config(temporary)
        backup = path.with_name(f"{path.name}.bak")
        shutil.copy2(path, backup)
        os.replace(temporary, path)
        return backup
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@dataclass
class IterationRow:
    iteration: int
    generation: int
    task: int | None
    statistics: str
    notification: str
    absorption: str
    blind_spot: str
    committee_replay: str
    stats: dict[str, int | float] | None = None


TASK_LABELS = {
    0: "准备训练",
    1: "训练",
    2: "训练后处理",
    3: "准备探索",
    4: "模型探索",
    5: "探索后处理",
    6: "准备 FP",
    7: "运行 FP",
    8: "FP 后处理",
}

STATISTICS_LABELS = {
    "ready": "就绪",
    "pending": "等待",
    "partial": "写入中",
    "unrecognized": "待识别",
    "log_missing": "无日志",
    "read_error": "读取失败",
    "unknown": "未知",
}


def display_width(text: str) -> int:
    """Return terminal cell width without requiring wcwidth."""
    return sum(
        0 if unicodedata.combining(char)
        else 2 if unicodedata.east_asian_width(char) in {"W", "F"}
        else 1
        for char in text
    )


def fit_cells(text: str, width: int, *, align: str = "left") -> str:
    """Clip and pad text to an exact number of terminal cells."""
    if width <= 0:
        return ""
    result: list[str] = []
    used = 0
    clipped = False
    for char in str(text):
        char_width = display_width(char)
        if used + char_width > width:
            clipped = True
            break
        result.append(char)
        used += char_width
    if clipped and width >= 1:
        while result and used + 1 > width:
            removed = result.pop()
            used -= display_width(removed)
        result.append("…")
        used += 1
    padding = " " * max(0, width - used)
    return padding + "".join(result) if align == "right" else "".join(result) + padding


class MonitorTui:
    TABS = ("总览", "事件", "设置")

    def __init__(self, config_path: Path):
        self.config_path = config_path.expanduser().resolve()
        self.config = load_config(self.config_path)
        self.tab = 0
        self.selected_row = 0
        self.selected_setting = 0
        self.rows: list[IterationRow] = []
        self.events: list[dict] = []
        self.settings = config_setting_values(self.config)
        self.saved_settings = dict(self.settings)
        self.message = "只读监控；设置保存后，后台 run 进程需重启"
        self.last_refresh = 0.0
        self.next_refresh = 0.0
        self.refresh()

    def refresh(self) -> None:
        self.config = load_config(self.config_path)
        observer = DpgenObserver(self.config.project.run_dir, self.config.dpgen)
        snapshots, inspections = observer.scan()
        state_path = self.config.project.output_dir / "monitor.sqlite3"
        store = StateStore(state_path) if state_path.exists() else None
        try:
            self.rows = [
                self._iteration_row(snapshot, inspections, store)
                for snapshot in reversed(snapshots)
            ]
            self.events = store.list_deliveries(100) if store else []
        finally:
            if store:
                store.close()
        if self.settings == self.saved_settings:
            self.settings = config_setting_values(self.config)
            self.saved_settings = dict(self.settings)
        self.selected_row = min(self.selected_row, max(0, len(self.rows) - 1))
        self.last_refresh = time.monotonic()
        self.next_refresh = self.last_refresh + self.config.project.check_interval

    def _iteration_row(
        self,
        snapshot: IterationSnapshot,
        inspections: dict[int, StatisticsInspection],
        store: StateStore | None,
    ) -> IterationRow:
        inspection = inspections.get(snapshot.iteration)
        statistics = inspection.status if inspection else "unknown"
        generation = store.get_iteration_generation(snapshot.iteration) if store else 0
        notification = "—"
        if (
            inspection
            and inspection.status == "ready"
            and snapshot.iteration >= self.config.dpgen.statistics_start_iteration
        ):
            key = f"statistics:iter.{snapshot.iteration:06d}"
            enabled = [item.name for item in self.config.notifications if item.enabled]
            delivered = bool(enabled) and bool(store) and all(
                store.is_delivered(key, name) for name in enabled
            )
            notification = "已发送" if delivered else "待发送"

        return IterationRow(
            snapshot.iteration,
            generation,
            snapshot.task,
            statistics,
            notification,
            self._phase_status(store, snapshot.iteration, "absorption"),
            self._phase_status(store, snapshot.iteration, "blind_spot"),
            self._committee_replay_status(store, snapshot.iteration),
            inspection.stats if inspection else None,
        )

    def _committee_replay_status(
        self, store: StateStore | None, model_iteration: int
    ) -> str:
        config = self.config.committee_replay
        if not config.enabled:
            return "关闭"
        if model_iteration < config.start_iteration:
            return "—"
        if not store:
            return "待运行"
        statuses = []
        for offset in config.source_offsets:
            source_iteration = model_iteration - offset
            if source_iteration < 0:
                continue
            row = store.get_committee_replay(model_iteration, source_iteration)
            statuses.append((row or {}).get("status"))
        if not statuses:
            return "—"
        if statuses.count("complete") == len(statuses):
            return "完成"
        if "failed" in statuses:
            return "失败"
        if "running" in statuses:
            return "运行中"
        if "cancelled" in statuses:
            return "已取消"
        return "等待"

    def _phase_status(
        self, store: StateStore | None, iteration: int, phase: str
    ) -> str:
        if iteration < self.config.evaluation.start_iteration:
            return "—"
        if phase == "blind_spot" and not self.config.evaluation.blind_spot_enabled:
            return "关闭"
        if not store:
            return "待运行"
        statuses = [
            (store.get_evaluation(iteration, phase, model_id) or {}).get("status")
            for model_id in self.config.evaluation.model_ids
        ]
        complete = statuses.count("complete")
        if complete == len(statuses):
            return "完成"
        if "failed" in statuses:
            return "失败"
        if "running" in statuses:
            return f"运行 {complete}/{len(statuses)}"
        if "cancelled" in statuses:
            return f"已取消 {complete}/{len(statuses)}"
        return f"等待 {complete}/{len(statuses)}"

    @property
    def dirty(self) -> bool:
        return self.settings != self.saved_settings

    def run(self, screen) -> None:
        locale.setlocale(locale.LC_ALL, "")
        curses.curs_set(0)
        screen.keypad(True)
        screen.timeout(1000)
        self._init_colors()
        while True:
            if time.monotonic() >= self.next_refresh and not self.dirty:
                try:
                    self.refresh()
                except Exception as exc:
                    self.message = f"刷新失败: {exc}"
            self.draw(screen)
            key = screen.getch()
            if key in (ord("q"), ord("Q")):
                return
            if key in (ord("1"), ord("2"), ord("3")):
                self.tab = key - ord("1")
            elif key in (curses.KEY_LEFT, curses.KEY_BTAB):
                self.tab = (self.tab - 1) % len(self.TABS)
            elif key in (curses.KEY_RIGHT, 9):
                self.tab = (self.tab + 1) % len(self.TABS)
            elif key in (curses.KEY_UP, ord("k")):
                self._move(-1)
            elif key in (curses.KEY_DOWN, ord("j")):
                self._move(1)
            elif key in (10, 13, curses.KEY_ENTER) and self.tab == 2:
                self._edit_setting(screen)
            elif key in (ord("w"), ord("W")) and self.tab == 2:
                self._save_settings()
            elif key == 27 and self.tab == 2:
                self.settings = dict(self.saved_settings)
                self.message = "已放弃未保存的设置"
            elif key in (ord("r"), ord("R")):
                try:
                    self.refresh()
                    self.message = "已刷新"
                except Exception as exc:
                    self.message = f"刷新失败: {exc}"

    @staticmethod
    def _init_colors() -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)

    def _move(self, delta: int) -> None:
        if self.tab == 0 and self.rows:
            self.selected_row = max(0, min(len(self.rows) - 1, self.selected_row + delta))
        elif self.tab == 2:
            self.selected_setting = max(
                0, min(len(SETTINGS) - 1, self.selected_setting + delta)
            )

    def draw(self, screen) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        if height < 20 or width < 76:
            self._put(
                screen,
                0,
                0,
                f"终端至少需要 76×20；当前 {width}×{height}",
                curses.A_BOLD,
            )
            screen.refresh()
            return
        self._put(screen, 0, 0, " DP-GEN MONITOR ", curses.A_REVERSE | curses.A_BOLD)
        safe_mode = "● SAFE MODE"
        safe_x = width - display_width(safe_mode) - 1
        config_width = max(0, safe_x - 22)
        self._put(
            screen,
            0,
            20,
            fit_cells(self.config_path.name, config_width),
            curses.A_BOLD,
        )
        self._put(screen, 0, safe_x, safe_mode, self._color(2) | curses.A_BOLD)
        tab_text = "  ".join(
            f"[{index + 1}] {name}" for index, name in enumerate(self.TABS)
        )
        self._put(screen, 2, 2, tab_text, curses.A_BOLD)
        selected_offset = sum(
            display_width(f"[{i + 1}] {name}") + 2
            for i, name in enumerate(self.TABS[: self.tab])
        )
        self._put(
            screen,
            3,
            2 + selected_offset,
            "─" * display_width(f"[{self.tab + 1}] {self.TABS[self.tab]}"),
            self._color(1),
        )

        if self.tab == 0:
            self._draw_overview(screen, height, width)
        elif self.tab == 1:
            self._draw_events(screen, height, width)
        else:
            self._draw_settings(screen, height, width)

        remaining = max(0, int(self.next_refresh - time.monotonic()))
        footer = f" q 退出  ←→ 切页  ↑↓ 选择  r 刷新  下次扫描 {remaining}s "
        if self.tab == 2:
            footer = " Enter 编辑  w 保存  Esc 放弃  " + footer
        self._put(screen, height - 2, 0, " " * (width - 1), curses.A_REVERSE)
        self._put(screen, height - 2, 1, footer, curses.A_REVERSE)
        self._put(screen, height - 1, 1, self.message)
        screen.refresh()

    def _draw_overview(self, screen, height: int, width: int) -> None:
        latest = self.rows[0] if self.rows else None
        pending = sum(row.notification == "待发送" for row in self.rows)
        stage = "尚未发现迭代"
        if latest:
            task = f"{latest.task:02d}" if latest.task is not None else "??"
            stage = f"iter.{latest.iteration:06d} · task {task}"

        generation = max((row.generation for row in self.rows), default=0)
        metrics = (
            ("活动迭代", str(len(self.rows))),
            ("当前阶段", stage),
            ("待推送", str(pending)),
            ("恢复代次", f"generation {generation}"),
        )
        if width >= 110:
            available = width - 5
            card_width = available // 4
            for index, (label, value) in enumerate(metrics):
                left = 1 + index * (card_width + 1)
                right = width - 2 if index == 3 else left + card_width - 1
                self._box(screen, 5, left, 8, right, label)
                self._put(
                    screen,
                    7,
                    left + 2,
                    fit_cells(value, max(1, right - left - 3)),
                    curses.A_BOLD | (self._color(1) if index == 1 else 0),
                )
        else:
            self._box(screen, 5, 1, 8, width - 2, "运行总览")
            summary = "  ".join(f"{label} {value}" for label, value in metrics)
            self._put(screen, 7, 3, summary, curses.A_BOLD)

        if generation:
            notice = (
                "▲ 已检测到运行恢复：旧结果已隔离，"
                f"当前最高为 generation {generation}"
            )
            notice_color = self._color(3) | curses.A_BOLD
        else:
            notice = "● 运行目录连续；未发现迭代回退或同编号目录重建"
            notice_color = self._color(2)
        self._put(screen, 9, 2, notice, notice_color)

        content_top = 11
        content_bottom = height - 4
        if width >= 124:
            table_right = min(width - 43, max(78, int(width * 0.64)))
            detail_left = table_right + 2
            self._draw_iteration_table(
                screen, content_top, 1, content_bottom, table_right
            )
            self._draw_iteration_detail(
                screen, content_top, detail_left, content_bottom, width - 2
            )
        else:
            self._draw_iteration_table(
                screen, content_top, 1, content_bottom, width - 2
            )

    def _draw_iteration_table(
        self, screen, top: int, left: int, bottom: int, right: int
    ) -> None:
        self._box(screen, top, left, bottom, right, "迭代列表")
        columns = (11, 4, 6, 8, 8, 11, 11)
        labels = ("ITERATION", "GEN", "TASK", "统计", "通知", "吸收评估", "盲区评估")
        header = " ".join(
            fit_cells(label, size) for label, size in zip(labels, columns)
        )
        self._put(screen, top + 1, left + 2, header, curses.A_DIM)
        visible = max(1, bottom - top - 3)
        start = max(0, self.selected_row - visible + 1)
        for line_index, row in enumerate(self.rows[start : start + visible]):
            absolute_index = start + line_index
            task = "??" if row.task is None else f"{row.task:02d}"
            values = (
                f"iter.{row.iteration:06d}",
                str(row.generation),
                task,
                STATISTICS_LABELS.get(row.statistics, row.statistics),
                row.notification,
                row.absorption,
                row.blind_spot,
            )
            text = " ".join(
                fit_cells(value, size, align="right" if index in {1, 2} else "left")
                for index, (value, size) in enumerate(zip(values, columns))
            )
            selected = absolute_index == self.selected_row
            marker = "›" if selected else " "
            attribute = curses.A_BOLD | self._color(1) if selected else 0
            if row.notification == "待发送":
                attribute |= self._color(3)
            self._put(screen, top + 2 + line_index, left + 1, marker, attribute)
            self._put(screen, top + 2 + line_index, left + 2, text, attribute)
        if not self.rows:
            self._put(screen, top + 3, left + 3, "尚未发现 DP-GEN 迭代")

    def _draw_iteration_detail(
        self, screen, top: int, left: int, bottom: int, right: int
    ) -> None:
        self._box(screen, top, left, bottom, right, "选中迭代")
        if not self.rows:
            self._put(screen, top + 2, left + 2, "无可显示内容")
            return
        row = self.rows[self.selected_row]
        inner_width = max(1, right - left - 3)
        task = "??" if row.task is None else f"{row.task:02d}"
        stage_label = TASK_LABELS.get(row.task, "未知阶段")
        lines = [
            (f"iter.{row.iteration:06d}", curses.A_BOLD | self._color(1)),
            (f"generation {row.generation}", curses.A_DIM),
            ("", 0),
            (f"阶段      task {task} · {stage_label}", 0),
            (f"探索统计  {STATISTICS_LABELS.get(row.statistics, row.statistics)}", 0),
            (f"通知      {row.notification}", 0),
            (f"吸收评估  {row.absorption}", 0),
            (f"盲区评估  {row.blind_spot}", 0),
            (f"委员会回放  {row.committee_replay}", 0),
        ]
        if row.stats:
            lines.extend(
                [
                    ("", 0),
                    ("探索统计比例", curses.A_BOLD),
                    (f"candidate  {float(row.stats['candidate_percent']):8.4f}%", 0),
                    (f"failed     {float(row.stats['failed_percent']):8.4f}%", 0),
                    (f"accurate   {float(row.stats['accurate_percent']):8.4f}%", 0),
                ]
            )
        path = self.config.project.run_dir / f"iter.{row.iteration:06d}"
        lines.extend(
            [
                ("", 0),
                ("运行目录", curses.A_BOLD),
                (str(path), curses.A_DIM),
                ("", 0),
                (
                    "TUI 只读监控；评估与推送由后台 run 进程执行。",
                    self._color(1),
                ),
            ]
        )
        y = top + 2
        for text, attribute in lines:
            if y >= bottom:
                break
            for wrapped in self._wrap_cells(text, inner_width):
                if y >= bottom:
                    break
                self._put(screen, y, left + 2, wrapped, attribute)
                y += 1

    def _draw_events(self, screen, height: int, width: int) -> None:
        self._put(screen, 5, 2, "最近通知记录", curses.A_BOLD)
        self._put(
            screen,
            7,
            1,
            " 时间                  状态       通道       事件",
            curses.A_UNDERLINE,
        )
        for index, event in enumerate(self.events[: max(1, height - 11)]):
            timestamp = str(event["updated_at"]).replace("T", " ")[:19]
            status = str(event["status"])
            color = self._color(2 if status == "delivered" else 4)
            text = (
                f" {timestamp:<20}  {status:<10} {str(event['notifier']):<10} "
                f"{event['event_key']}"
            )
            self._put(screen, 8 + index, 1, text, color)
        if not self.events:
            self._put(screen, 8, 2, "尚无通知记录")

    def _draw_settings(self, screen, height: int, width: int) -> None:
        self._put(screen, 5, 2, "运行参数", curses.A_BOLD)
        self._put(
            screen,
            5,
            max(28, width - 38),
            "已修改" if self.dirty else "与磁盘配置一致",
            self._color(3 if self.dirty else 2),
        )
        self._put(
            screen,
            7,
            1,
            " 参数                         当前值      说明",
            curses.A_UNDERLINE,
        )
        for index, spec in enumerate(SETTINGS):
            value = self.settings[spec.identity]
            marker = "*" if value != self.saved_settings[spec.identity] else " "
            text = f"{marker} {spec.label:<26} {value:>8}      {spec.help_text}"
            selected = index == self.selected_setting
            attribute = curses.A_BOLD | self._color(1) if selected else 0
            self._put(screen, 8 + index, 1, "›" if selected else " ", attribute)
            self._put(screen, 8 + index, 3, text, attribute)
        self._put(
            screen,
            min(height - 5, 10 + len(SETTINGS)),
            2,
            "保存会生成同目录 .bak 备份并原子替换 TOML；"
            "不会显示或修改通知凭证。",
            self._color(1),
        )
        self._put(
            screen,
            min(height - 4, 11 + len(SETTINGS)),
            2,
            "独立运行中的 dpgen-monitor run 需重启后加载新参数。",
            self._color(3),
        )

    def _edit_setting(self, screen) -> None:
        spec = SETTINGS[self.selected_setting]
        height, width = screen.getmaxyx()
        prompt = f"{spec.label} [{spec.minimum}–{spec.maximum}]: "
        screen.timeout(-1)
        curses.echo()
        curses.curs_set(1)
        try:
            self._put(screen, height - 1, 0, " " * (width - 1))
            self._put(screen, height - 1, 1, prompt)
            raw = screen.getstr(
                height - 1,
                min(width - 2, 1 + display_width(prompt)),
                16,
            )
            value = int(raw.decode("utf-8").strip())
            candidate = dict(self.settings)
            candidate[spec.identity] = value
            validate_setting_values(candidate)
            self.settings = candidate
            self.message = f"已修改 {spec.label}={value}；按 w 保存"
        except (ValueError, UnicodeDecodeError) as exc:
            self.message = f"输入无效: {exc}"
        finally:
            curses.noecho()
            curses.curs_set(0)
            screen.timeout(1000)

    def _save_settings(self) -> None:
        if not self.dirty:
            self.message = "没有需要保存的设置"
            return
        try:
            backup = save_toml_settings(self.config_path, self.settings)
            self.saved_settings = dict(self.settings)
            self.refresh()
            self.message = f"设置已保存；备份 {backup.name}；请重启后台 run"
        except Exception as exc:
            self.message = f"保存失败: {exc}"

    @staticmethod
    def _color(pair: int) -> int:
        return curses.color_pair(pair) if curses.has_colors() else 0

    @classmethod
    def _box(
        cls,
        screen,
        top: int,
        left: int,
        bottom: int,
        right: int,
        title: str = "",
    ) -> None:
        if bottom <= top or right <= left:
            return
        cls._put(screen, top, left, "╭" + "─" * (right - left - 1) + "╮", curses.A_DIM)
        for y in range(top + 1, bottom):
            cls._put(screen, y, left, "│", curses.A_DIM)
            cls._put(screen, y, right, "│", curses.A_DIM)
        cls._put(
            screen,
            bottom,
            left,
            "╰" + "─" * (right - left - 1) + "╯",
            curses.A_DIM,
        )
        if title:
            cls._put(screen, top, left + 2, f" {title} ", curses.A_BOLD)

    @staticmethod
    def _wrap_cells(text: str, width: int) -> list[str]:
        if not text:
            return [""]
        lines: list[str] = []
        remaining = text
        while remaining:
            chunk = fit_cells(remaining, width).rstrip()
            if not chunk:
                break
            lines.append(chunk)
            remaining = remaining[len(chunk.rstrip("…")) :]
        return lines or [""]

    @staticmethod
    def _put(screen, y: int, x: int, text: str, attribute: int = 0) -> None:
        height, width = screen.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width - 1:
            return
        try:
            clipped = fit_cells(text, max(0, width - x - 1)).rstrip()
            screen.addstr(y, x, clipped, attribute)
        except curses.error:
            pass


def run_tui(config_path: Path) -> int:
    app = MonitorTui(config_path)
    try:
        curses.wrapper(app.run)
    except curses.error as exc:
        raise RuntimeError(f"无法启动 TUI；请确认当前是交互式终端: {exc}") from exc
    return 0
