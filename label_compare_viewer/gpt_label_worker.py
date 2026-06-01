from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    from .board_geometry_hints import compact_board_x_range_hints, compute_board_x_range_hints, create_board_hint_overlay
    from .diff_overlay import create_diff_overlay
    from .openai_batch_labeler import OpenAIBatchLabeler, estimate_openai_cost
    from .prompt_manager import CORRECTION_PROMPT, load_prompt, prompt_path, render_prompt
    from .report_writer import (
        create_yolo_label_overlay,
        image_metrics_to_row,
        timestamp_id,
        write_debug_manifest,
        write_parsed_output,
        write_run_summary,
    )
    from .yolo_metrics import compare_records, object_to_box_record, read_box_records, truth_label_path_for_image
    from .yolo_parser import YoloObject, parse_yolo_label_text, yolo_object_to_detect_line
    from .yolo_validation import split_validation_messages, validate_pallet_yolo_labels
except ImportError:
    from board_geometry_hints import compact_board_x_range_hints, compute_board_x_range_hints, create_board_hint_overlay
    from diff_overlay import create_diff_overlay
    from openai_batch_labeler import OpenAIBatchLabeler, estimate_openai_cost
    from prompt_manager import CORRECTION_PROMPT, load_prompt, prompt_path, render_prompt
    from report_writer import (
        create_yolo_label_overlay,
        image_metrics_to_row,
        timestamp_id,
        write_debug_manifest,
        write_parsed_output,
        write_run_summary,
    )
    from yolo_metrics import compare_records, object_to_box_record, read_box_records, truth_label_path_for_image
    from yolo_parser import YoloObject, parse_yolo_label_text, yolo_object_to_detect_line
    from yolo_validation import split_validation_messages, validate_pallet_yolo_labels


STRUCTURE_ALLOWED = {0, 1, 2, 4}
DEFECT_ALLOWED = {6, 7, 13}
ALL_ALLOWED = {0, 1, 2, 4, 6, 7, 13}


class GPTLabelWorker:
    def __init__(
        self,
        working_dir: Path,
        image_paths: list[Path],
        prompt_template: str,
        classes: list[str],
        settings: dict[str, Any],
        options: dict[str, Any],
        progress_queue: queue.Queue,
    ):
        self.working_dir = working_dir
        self.image_paths = image_paths
        self.prompt_template = prompt_template
        self.classes = classes
        self.settings = settings
        self.options = options
        self.progress_queue = progress_queue
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.run_dir: Path | None = None
        self.raw_dir: Path | None = None
        self.parsed_dir: Path | None = None
        self.error_dir: Path | None = None
        self.validation_dir: Path | None = None
        self.hints_dir: Path | None = None
        self.debug_root: Path | None = None

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _put(self, event_type: str, **payload):
        self.progress_queue.put({"type": event_type, **payload})

    def run(self):
        run_id = timestamp_id("run")
        self.run_dir = self.working_dir / "_runs" / run_id
        self.raw_dir = self.run_dir / "raw_outputs"
        self.parsed_dir = self.run_dir / "parsed_outputs"
        self.error_dir = self.run_dir / "errors"
        self.validation_dir = self.run_dir / "validation"
        self.hints_dir = self.run_dir / "board_hints"
        self.debug_root = self.run_dir / "debug"
        for folder in (self.raw_dir, self.parsed_dir, self.error_dir, self.validation_dir, self.hints_dir, self.debug_root):
            folder.mkdir(parents=True, exist_ok=True)

        source_name = self.options["source_name"]
        run_config = {
            "run_id": run_id,
            "source_name": source_name,
            "model": self.options.get("model"),
            "reasoning_effort": self.options.get("reasoning_effort"),
            "image_detail": self.options.get("image_detail"),
            "max_output_tokens": self.options.get("max_output_tokens"),
            "use_background": self.options.get("use_background"),
            "use_two_pass": self.options.get("use_two_pass"),
            "structure_model": self.options.get("structure_model"),
            "structure_reasoning_effort": self.options.get("structure_reasoning_effort"),
            "structure_max_output_tokens": self.options.get("structure_max_output_tokens"),
            "defect_model": self.options.get("defect_model"),
            "defect_reasoning_effort": self.options.get("defect_reasoning_effort"),
            "defect_max_output_tokens": self.options.get("defect_max_output_tokens"),
            "max_cost_per_image": self.options.get("max_cost_per_image"),
            "image_count": len(self.image_paths),
            "prompt_comparison": bool(self.options.get("prompt_comparison")),
        }
        (self.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        (self.run_dir / "prompt_template.txt").write_text(self.prompt_template, encoding="utf-8")

        rows: list[dict[str, Any]] = []
        self._put("started", run_id=run_id, run_dir=str(self.run_dir), total=len(self.image_paths))

        for index, image_path in enumerate(self.image_paths, start=1):
            if self.stop_event.is_set():
                self._put("log", message="Stop requested; remaining images skipped.")
                break
            image_path = Path(image_path)
            if self.options.get("prompt_comparison"):
                image_rows = self._run_prompt_comparison_image(image_path)
            else:
                image_rows = [self._run_standard_image(image_path)]
            rows.extend(image_rows)
            write_run_summary(self.run_dir, rows)
            status = image_rows[-1]["status"] if image_rows else "error"
            self._put("progress", index=index, total=len(self.image_paths), image_stem=image_path.stem, status=status)
            self._maybe_warn_budget(image_path, image_rows)

        write_run_summary(self.run_dir, rows)
        self._put(
            "done",
            run_id=run_id,
            run_dir=str(self.run_dir),
            rows=rows,
            prompt_comparison=bool(self.options.get("prompt_comparison")),
            image_path=str(self.image_paths[0]) if self.image_paths else "",
        )

    def _run_standard_image(self, image_path: Path) -> dict[str, Any]:
        source_name = self.options["source_name"]
        saved_label_path = image_path.parent / f"{source_name}.txt"
        if self.options.get("skip_existing") and saved_label_path.exists():
            row = self._base_row(image_path, source_name, "skipped_existing")
            row["saved_label_path"] = str(saved_label_path)
            return row

        started = time.monotonic()
        row = self._base_row(image_path, source_name, "running")
        try:
            width, height = self._image_size(image_path)
            row["width"] = width
            row["height"] = height
            hints, hints_text = self._compute_and_save_hints(image_path)
            if self.options.get("use_two_pass"):
                result = self._run_two_pass(image_path, source_name, width, height, hints, hints_text)
            else:
                result = self._run_single_pass(
                    image_path,
                    source_name,
                    self.prompt_template,
                    width,
                    height,
                    hints,
                    hints_text,
                    model=self.options.get("model", "gpt-5.5"),
                    reasoning_effort=self.options.get("reasoning_effort", "medium"),
                    max_output_tokens=int(self.options.get("max_output_tokens", 3000)),
                    allowed_classes=ALL_ALLOWED,
                    validation_config=self._validation_config(ALL_ALLOWED, warn_defects_absent=True),
                )
            row.update(result["row"])
            row["width"] = width
            row["height"] = height
            self._write_debug_report(
                image_path=image_path,
                source_name=source_name,
                hints=hints,
                raw_outputs=result.get("raw_outputs", {}),
                objects=result.get("objects", []),
                validation_messages=result.get("validation_messages", []),
            )
            if row["status"] == "saved":
                self._put("label_saved", image_path=str(image_path), label_path=row["saved_label_path"])
        except Exception as exc:
            self._write_error(image_path, str(exc))
            row["status"] = "error"
            row["error"] = str(exc)
            self._put("log", message=f"{image_path.stem}: {exc}")
        row["elapsed_seconds"] = time.monotonic() - started
        return row

    def _run_prompt_comparison_image(self, image_path: Path) -> list[dict[str, Any]]:
        started = time.monotonic()
        rows: list[dict[str, Any]] = []
        try:
            width, height = self._image_size(image_path)
            hints, hints_text = self._compute_and_save_hints(image_path)
            old_template = load_prompt(prompt_path(self.working_dir, "pallet_prompt.default.txt"))
            v2_template = load_prompt(prompt_path(self.working_dir, "pallet_prompt.v2_board_split.txt"))

            variants = [
                ("gpt_old", old_template),
                ("gpt_v2_single", v2_template),
            ]
            for source_name, template in variants:
                row = self._base_row(image_path, source_name, "running")
                row["width"] = width
                row["height"] = height
                result = self._run_single_pass(
                    image_path,
                    source_name,
                    template,
                    width,
                    height,
                    hints,
                    hints_text,
                    model=self.options.get("model", "gpt-5.5"),
                    reasoning_effort=self.options.get("reasoning_effort", "medium"),
                    max_output_tokens=int(self.options.get("max_output_tokens", 3000)),
                    allowed_classes=ALL_ALLOWED,
                    validation_config=self._validation_config(ALL_ALLOWED, warn_defects_absent=True),
                )
                row.update(result["row"])
                row["width"] = width
                row["height"] = height
                row["elapsed_seconds"] = time.monotonic() - started
                rows.append(row)
                self._write_debug_report(
                    image_path=image_path,
                    source_name=source_name,
                    hints=hints,
                    raw_outputs=result.get("raw_outputs", {}),
                    objects=result.get("objects", []),
                    validation_messages=result.get("validation_messages", []),
                )
                if row["status"] == "saved":
                    self._put("label_saved", image_path=str(image_path), label_path=row["saved_label_path"])

            source_name = "gpt_v2_twopass"
            row = self._base_row(image_path, source_name, "running")
            row["width"] = width
            row["height"] = height
            result = self._run_two_pass(image_path, source_name, width, height, hints, hints_text)
            row.update(result["row"])
            row["width"] = width
            row["height"] = height
            row["elapsed_seconds"] = time.monotonic() - started
            rows.append(row)
            self._write_debug_report(
                image_path=image_path,
                source_name=source_name,
                hints=hints,
                raw_outputs=result.get("raw_outputs", {}),
                objects=result.get("objects", []),
                validation_messages=result.get("validation_messages", []),
            )
            if row["status"] == "saved":
                self._put("label_saved", image_path=str(image_path), label_path=row["saved_label_path"])
        except Exception as exc:
            row = self._base_row(image_path, "prompt_comparison", "error")
            row["error"] = str(exc)
            row["elapsed_seconds"] = time.monotonic() - started
            rows.append(row)
            self._write_error(image_path, str(exc))
        return rows

    def _run_single_pass(
        self,
        image_path: Path,
        source_name: str,
        template_text: str,
        width: int,
        height: int,
        hints: dict[str, Any],
        hints_text: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        allowed_classes: set[int],
        validation_config: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = render_prompt(
            template_text,
            image_path,
            width,
            height,
            self.classes,
            sorted(allowed_classes),
            self.settings.get("expected_counts", {}),
            board_x_range_hints=hints_text,
        )
        call = self._call_model(image_path, source_name, "single", prompt, model, reasoning_effort, max_output_tokens)
        objects, parse_errors = self._parse_and_save(
            image_path,
            source_name,
            "single",
            call["raw_text"],
            allowed_classes,
            allow_empty=False,
        )
        validation_messages: list[str] = []
        retry_count = 0
        if not parse_errors:
            valid, validation_messages = validate_pallet_yolo_labels(objects, validation_config)
            errors, _warnings = split_validation_messages(validation_messages)
            if not valid and self.options.get("retry_on_validation_failure", True):
                retry_count = 1
                retry = self._run_correction_retry(
                    image_path,
                    source_name,
                    prompt,
                    call["raw_text"],
                    errors,
                    hints_text,
                    model,
                    reasoning_effort,
                    allowed_classes,
                    validation_config,
                )
                if retry is not None:
                    call = self._merge_call_usage(call, retry["call"])
                    objects = retry["objects"]
                    parse_errors = retry["parse_errors"]
                    validation_messages = retry["validation_messages"]

        row = self._row_from_result(
            image_path=image_path,
            source_name=source_name,
            model=model,
            reasoning_effort=reasoning_effort,
            call=call,
            objects=objects,
            parse_errors=parse_errors,
            validation_messages=validation_messages,
            retry_count=retry_count,
        )
        if row["status"] == "saved":
            self._save_final_labels(image_path, source_name, objects)
            row["saved_label_path"] = str(image_path.parent / f"{source_name}.txt")
        else:
            self._write_validation_file(image_path, source_name, parse_errors + validation_messages)
        return {
            "row": row,
            "objects": objects,
            "raw_outputs": {"single": call.get("raw_text", "")},
            "validation_messages": parse_errors + validation_messages,
        }

    def _run_two_pass(
        self,
        image_path: Path,
        source_name: str,
        width: int,
        height: int,
        hints: dict[str, Any],
        hints_text: str,
    ) -> dict[str, Any]:
        structure_template = load_prompt(prompt_path(self.working_dir, "pallet_prompt.v2_structure_only.txt"))
        defect_template = load_prompt(prompt_path(self.working_dir, "pallet_prompt.v2_defects_only.txt"))
        structure_model = self.options.get("structure_model", "gpt-5.5")
        structure_effort = self.options.get("structure_reasoning_effort", "medium")
        structure_tokens = int(self.options.get("structure_max_output_tokens", 2000))
        defect_model = self.options.get("defect_model", "gpt-5.5")
        defect_effort = self.options.get("defect_reasoning_effort", "high")
        defect_tokens = int(self.options.get("defect_max_output_tokens", 1500))

        structure_prompt = render_prompt(
            structure_template,
            image_path,
            width,
            height,
            self.classes,
            sorted(STRUCTURE_ALLOWED),
            self.settings.get("expected_counts", {}),
            board_x_range_hints=hints_text,
        )
        structure_call = self._call_model(
            image_path,
            source_name,
            "structure",
            structure_prompt,
            structure_model,
            structure_effort,
            structure_tokens,
        )
        structure_objects, structure_parse_errors = self._parse_and_save(
            image_path,
            source_name,
            "structure",
            structure_call["raw_text"],
            STRUCTURE_ALLOWED,
            allow_empty=False,
        )
        validation_messages: list[str] = []
        retry_count = 0
        if not structure_parse_errors:
            valid, validation_messages = validate_pallet_yolo_labels(
                structure_objects,
                self._validation_config(STRUCTURE_ALLOWED, warn_defects_absent=False),
            )
            errors, _warnings = split_validation_messages(validation_messages)
            if not valid and self.options.get("retry_on_validation_failure", True):
                retry_count = 1
                retry = self._run_correction_retry(
                    image_path,
                    source_name,
                    structure_prompt,
                    structure_call["raw_text"],
                    errors,
                    hints_text,
                    structure_model,
                    structure_effort,
                    STRUCTURE_ALLOWED,
                    self._validation_config(STRUCTURE_ALLOWED, warn_defects_absent=False),
                )
                if retry is not None:
                    structure_call = self._merge_call_usage(structure_call, retry["call"])
                    structure_objects = retry["objects"]
                    structure_parse_errors = retry["parse_errors"]
                    validation_messages = retry["validation_messages"]

        structure_errors, _structure_warnings = split_validation_messages(validation_messages)
        if structure_parse_errors or structure_errors:
            row = self._row_from_result(
                image_path=image_path,
                source_name=source_name,
                model=structure_model,
                reasoning_effort=structure_effort,
                call=structure_call,
                objects=structure_objects,
                parse_errors=structure_parse_errors,
                validation_messages=validation_messages,
                retry_count=retry_count,
            )
            self._write_validation_file(image_path, source_name, structure_parse_errors + validation_messages)
            return {
                "row": row,
                "objects": structure_objects,
                "raw_outputs": {"structure": structure_call.get("raw_text", "")},
                "validation_messages": structure_parse_errors + validation_messages,
            }

        defect_prompt = render_prompt(
            defect_template,
            image_path,
            width,
            height,
            self.classes,
            sorted(DEFECT_ALLOWED),
            {},
            board_x_range_hints=hints_text,
        )
        defect_call = self._call_model(
            image_path,
            source_name,
            "defects",
            defect_prompt,
            defect_model,
            defect_effort,
            defect_tokens,
        )
        defect_objects, defect_parse_errors = self._parse_and_save(
            image_path,
            source_name,
            "defects",
            defect_call["raw_text"],
            DEFECT_ALLOWED,
            allow_empty=True,
        )

        merged_objects = structure_objects + defect_objects
        final_validation_messages = list(validation_messages)
        if not structure_parse_errors and not defect_parse_errors:
            _valid_final, final_validation_messages = validate_pallet_yolo_labels(
                merged_objects,
                self._validation_config(ALL_ALLOWED, warn_defects_absent=True),
            )

        call = self._merge_call_usage(structure_call, defect_call)
        row = self._row_from_result(
            image_path=image_path,
            source_name=source_name,
            model=f"{structure_model}+{defect_model}",
            reasoning_effort=f"{structure_effort}+{defect_effort}",
            call=call,
            objects=merged_objects,
            parse_errors=structure_parse_errors + defect_parse_errors,
            validation_messages=final_validation_messages,
            retry_count=retry_count,
        )
        if row["status"] == "saved":
            self._save_final_labels(image_path, source_name, merged_objects)
            self._save_final_labels(image_path, f"{source_name}_structure", structure_objects)
            self._save_final_labels(image_path, f"{source_name}_defects", defect_objects)
            row["saved_label_path"] = str(image_path.parent / f"{source_name}.txt")
        else:
            self._write_validation_file(image_path, source_name, structure_parse_errors + defect_parse_errors + final_validation_messages)
        return {
            "row": row,
            "objects": merged_objects,
            "raw_outputs": {
                "structure": structure_call.get("raw_text", ""),
                "defects": defect_call.get("raw_text", ""),
            },
            "validation_messages": structure_parse_errors + defect_parse_errors + final_validation_messages,
        }

    def _run_correction_retry(
        self,
        image_path: Path,
        source_name: str,
        original_prompt: str,
        previous_output: str,
        validation_errors: list[str],
        hints_text: str,
        model: str,
        reasoning_effort: str,
        allowed_classes: set[int],
        validation_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not validation_errors:
            return None
        self._put("log", message=f"{image_path.stem}: validation failed; running one correction retry")
        correction = render_prompt(
            CORRECTION_PROMPT,
            image_path,
            0,
            0,
            self.classes,
            sorted(allowed_classes),
            self.settings.get("expected_counts", {}),
            board_x_range_hints=hints_text,
            extra_replacements={
                "VALIDATION_ERRORS": "\n".join(validation_errors),
                "PREVIOUS_OUTPUT": previous_output,
            },
        )
        prompt = original_prompt.rstrip() + "\n\n" + correction
        call = self._call_model(image_path, source_name, "retry1", prompt, model, reasoning_effort, 2000)
        objects, parse_errors = self._parse_and_save(
            image_path,
            source_name,
            "retry1",
            call["raw_text"],
            allowed_classes,
            allow_empty=False,
        )
        validation_messages: list[str] = []
        if not parse_errors:
            _valid, validation_messages = validate_pallet_yolo_labels(objects, validation_config)
        return {
            "call": call,
            "objects": objects,
            "parse_errors": parse_errors,
            "validation_messages": validation_messages,
        }

    def _call_model(
        self,
        image_path: Path,
        source_name: str,
        pass_name: str,
        prompt: str,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        labeler = OpenAIBatchLabeler(
            api_key=self.options["api_key"],
            model=model,
            reasoning_effort=reasoning_effort,
            image_detail=self.options.get("image_detail", "original"),
            max_output_tokens=max_output_tokens,
            use_background=bool(self.options.get("use_background", False)),
        )
        response = labeler.label_image(image_path, prompt)
        raw_text = response.get("output_text", "")
        raw_path = self.raw_dir / f"{image_path.stem}.{source_name}.{pass_name}.raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")
        usage = response.get("usage", {})
        estimated_cost = estimate_openai_cost(
            model,
            usage,
            self.settings.get("cost_rates_per_million"),
        )
        warnings = response.get("warnings", [])
        reasoning_tokens = usage.get("reasoning_tokens", "")
        warning_limit = self.options.get("max_reasoning_tokens_warning")
        if warning_limit not in (None, "") and reasoning_tokens not in (None, ""):
            try:
                if int(reasoning_tokens) > int(warning_limit):
                    warnings.append(f"reasoning tokens {reasoning_tokens} exceeded warning limit {warning_limit}")
            except (TypeError, ValueError):
                pass
        return {
            "raw_text": raw_text,
            "raw_path": str(raw_path),
            "response_id": response.get("response_id", ""),
            "usage": usage,
            "warnings": warnings,
            "estimated_cost": estimated_cost,
            "elapsed_seconds": response.get("elapsed_seconds", ""),
        }

    def _parse_and_save(
        self,
        image_path: Path,
        source_name: str,
        pass_name: str,
        raw_text: str,
        allowed_classes: set[int],
        allow_empty: bool,
    ) -> tuple[list[YoloObject], list[str]]:
        objects, errors = parse_yolo_label_text(
            raw_text,
            allowed_class_ids=allowed_classes,
            allow_segments=False,
            require_detect_only=True,
            allow_empty=allow_empty,
        )
        payload = {
            "image_stem": image_path.stem,
            "source_name": source_name,
            "pass": pass_name,
            "objects": [obj.__dict__ for obj in objects],
            "errors": errors,
        }
        parsed_path = self.parsed_dir / f"{image_path.stem}.{source_name}.{pass_name}.parsed.json"
        write_parsed_output(parsed_path, payload)
        return objects, errors

    def _row_from_result(
        self,
        image_path: Path,
        source_name: str,
        model: str,
        reasoning_effort: str,
        call: dict[str, Any],
        objects: list[YoloObject],
        parse_errors: list[str],
        validation_messages: list[str],
        retry_count: int,
    ) -> dict[str, Any]:
        row = self._base_row(image_path, source_name, "running")
        row["model"] = model
        row["reasoning_effort"] = reasoning_effort
        row["raw_output_path"] = call.get("raw_path", "")
        row["response_id"] = call.get("response_id", "")
        row["input_tokens"] = call.get("usage", {}).get("input_tokens", "")
        row["output_tokens"] = call.get("usage", {}).get("output_tokens", "")
        row["reasoning_tokens"] = call.get("usage", {}).get("reasoning_tokens", "")
        estimated_cost = call.get("estimated_cost")
        row["estimated_cost"] = f"{estimated_cost:.6f}" if isinstance(estimated_cost, (int, float)) else ""
        row["retry_count"] = retry_count
        errors, warnings = split_validation_messages(validation_messages)
        all_errors = parse_errors + errors
        row["warnings"] = "; ".join([*call.get("warnings", []), *warnings])
        row["validation_errors"] = "; ".join(validation_messages)
        row["validation_status"] = "passed" if not all_errors else "failed"
        if parse_errors:
            row["status"] = "parser_failed"
            row["error"] = "; ".join(parse_errors)
        elif all_errors:
            row["status"] = "validation_failed"
            row["error"] = "; ".join(all_errors)
        else:
            row["status"] = "saved"
            row["label_count"] = len(objects)
        return row

    def _merge_call_usage(self, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        first_usage = first.get("usage", {}) or {}
        second_usage = second.get("usage", {}) or {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"):
            value = self._sum_token_values(first_usage.get(key), second_usage.get(key))
            if value != "":
                usage[key] = value
        cost = 0.0
        cost_known = False
        for item in (first, second):
            item_cost = item.get("estimated_cost")
            if isinstance(item_cost, (int, float)):
                cost += float(item_cost)
                cost_known = True
        return {
            "raw_text": "\n".join(part for part in (first.get("raw_text", ""), second.get("raw_text", "")) if part),
            "raw_path": ";".join(part for part in (first.get("raw_path", ""), second.get("raw_path", "")) if part),
            "response_id": ";".join(part for part in (first.get("response_id", ""), second.get("response_id", "")) if part),
            "usage": usage,
            "warnings": list(first.get("warnings", [])) + list(second.get("warnings", [])),
            "estimated_cost": cost if cost_known else None,
        }

    def _sum_token_values(self, first: Any, second: Any) -> int | str:
        if first in (None, "") and second in (None, ""):
            return ""
        try:
            return int(first or 0) + int(second or 0)
        except (TypeError, ValueError):
            return ""

    def _validation_config(self, allowed_classes: set[int], warn_defects_absent: bool) -> dict[str, Any]:
        config = dict(self.settings)
        config["allowed_class_ids"] = sorted(allowed_classes)
        config["strict_structure"] = True
        config["warn_defects_absent"] = warn_defects_absent
        return config

    def _save_final_labels(self, image_path: Path, source_name: str, objects: list[YoloObject]) -> Path:
        label_path = image_path.parent / f"{source_name}.txt"
        clean_lines = [yolo_object_to_detect_line(obj) for obj in objects]
        label_path.write_text("\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8")
        return label_path

    def _write_validation_file(self, image_path: Path, source_name: str, messages: list[str]):
        if not messages:
            return
        path = self.validation_dir / f"{image_path.stem}.{source_name}.validation.txt"
        path.write_text("\n".join(messages), encoding="utf-8")

    def _write_error(self, image_path: Path, message: str):
        path = self.error_dir / f"{image_path.stem}.error.txt"
        path.write_text(message, encoding="utf-8")

    def _image_size(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            return image.size

    def _compute_and_save_hints(self, image_path: Path) -> tuple[dict[str, Any], str]:
        hints = compute_board_x_range_hints(image_path)
        hints_path = self.hints_dir / f"{image_path.stem}.json"
        hints_path.write_text(json.dumps(hints, indent=2), encoding="utf-8")
        return hints, compact_board_x_range_hints(hints)

    def _write_debug_report(
        self,
        image_path: Path,
        source_name: str,
        hints: dict[str, Any],
        raw_outputs: dict[str, str],
        objects: list[YoloObject],
        validation_messages: list[str],
    ):
        stem_dir = self.debug_root / image_path.stem
        debug_dir = stem_dir / source_name if self.options.get("prompt_comparison") else stem_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(image_path, debug_dir / f"original{image_path.suffix.lower()}")
        except OSError:
            pass
        try:
            create_board_hint_overlay(image_path, hints, debug_dir / "board_x_range_hints.jpg")
        except Exception as exc:
            (debug_dir / "board_x_range_hints.error.txt").write_text(str(exc), encoding="utf-8")
        (debug_dir / "gpt_raw_output.txt").write_text(
            "\n\n".join(f"## {name}\n{text}" for name, text in raw_outputs.items()),
            encoding="utf-8",
        )
        (debug_dir / "parsed_labels.txt").write_text(
            "\n".join(yolo_object_to_detect_line(obj) for obj in objects) + ("\n" if objects else ""),
            encoding="utf-8",
        )
        (debug_dir / "validation_errors.txt").write_text("\n".join(validation_messages), encoding="utf-8")
        try:
            create_yolo_label_overlay(image_path, objects, self.classes, debug_dir / "gpt_labels_overlay.jpg")
        except Exception as exc:
            (debug_dir / "gpt_labels_overlay.error.txt").write_text(str(exc), encoding="utf-8")

        comparison: dict[str, Any] = {"truth_available": False}
        truth_path = truth_label_path_for_image(image_path)
        if truth_path is not None:
            try:
                width, height = self._image_size(image_path)
                allowed = set(self.settings.get("allowed_class_ids", [])) or None
                truth_records, truth_errors = read_box_records(truth_path, image_path, "truth", self.classes, allowed)
                pred_records = [
                    object_to_box_record(obj, image_path.stem, source_name, self.classes, width, height)
                    for obj in objects
                ]
                metric = compare_records(
                    image_path.stem,
                    source_name,
                    truth_records,
                    pred_records,
                    self.settings.get("iou_thresholds", {}),
                    self.settings.get("tiny_object_center_match_px", {}),
                    set(self.settings.get("defect_class_ids", [6, 7, 13])),
                )
                comparison = {"truth_available": True, "truth_errors": truth_errors, "scores": image_metrics_to_row(metric)}
                create_diff_overlay(image_path, metric, debug_dir / "truth_vs_gpt_diff.jpg")
            except Exception as exc:
                comparison = {"truth_available": True, "error": str(exc)}

        write_debug_manifest(
            debug_dir,
            {
                "image_path": str(image_path),
                "source_name": source_name,
                "board_hints": hints,
                "raw_outputs": {name: f"gpt_raw_output.txt#{name}" for name in raw_outputs},
                "parsed_label_count": len(objects),
                "validation_messages": validation_messages,
                "comparison_to_truth": comparison,
            },
        )

    def _maybe_warn_budget(self, image_path: Path, rows: list[dict[str, Any]]):
        try:
            budget = float(self.options.get("max_cost_per_image", 0.25))
        except (TypeError, ValueError):
            return
        for row in rows:
            try:
                cost = float(row.get("estimated_cost") or 0.0)
            except (TypeError, ValueError):
                continue
            if cost > budget:
                self._put(
                    "budget_exceeded",
                    image_stem=image_path.stem,
                    source_name=row.get("source_name", ""),
                    estimated_cost=cost,
                    max_cost_per_image=budget,
                )

    def _base_row(self, image_path: Path, source_name: str, status: str) -> dict[str, Any]:
        return {
            "image_stem": image_path.stem,
            "image_path": str(image_path),
            "source_name": source_name,
            "model": self.options.get("model", ""),
            "reasoning_effort": self.options.get("reasoning_effort", ""),
            "image_detail": self.options.get("image_detail", ""),
            "status": status,
            "error": "",
            "raw_output_path": "",
            "saved_label_path": "",
            "label_count": "",
            "warnings": "",
            "width": "",
            "height": "",
            "response_id": "",
            "input_tokens": "",
            "output_tokens": "",
            "reasoning_tokens": "",
            "estimated_cost": "",
            "elapsed_seconds": "",
            "validation_status": "",
            "validation_errors": "",
            "retry_count": "",
        }
