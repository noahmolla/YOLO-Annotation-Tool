from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


STRICT_YOLO_REMINDER = "Every output row must be: class_id x_center y_center width height"

DEFAULT_PROMPT = """You are creating high-precision YOLO object-detection labels for a top-down pallet image.

Return ONLY YOLO label lines in this exact format:
<class_id> <x_center> <y_center> <width> <height>

Use normalized coordinates from 0 to 1.
Do not output explanations, comments, markdown, headers, blank lines, confidence scores, or JSON.
One object per line.
Every row must have exactly 5 fields.

Allowed classes:
0 = entire pallet
1 = regular top deck board
2 = horizontal stringer
4 = lead board / outside edge board
6 = broken board or broken stringer
7 = major defect crack or hole
13 = protruding nail

Image and pallet type:
The image is always top-down.
The image is expected to be 1024x1024 pixels.
This pallet is rotated 90 degrees.
The top deck boards appear vertical in the image and are arranged from left to right.
This specific pallet type should have exactly 9 top deck-board boxes:
- far-left lead board / outside edge board = class 4
- far-right lead board / outside edge board = class 4
- the 7 boards between them = class 1

Labeling rules:
- Label the entire visible pallet as class 0.
- Label every regular top deck board as class 1.
- Label every horizontal stringer as class 2.
- Label the outside edge / lead boards as class 4.
- Label broken board or broken stringer regions as class 6.
- Label major defect cracks or holes as class 7.
- Label protruding nails only as class 13.
- Do not label flush nails unless the prompt is later expanded with a flush nail class.
- Do not label conveyor rails, machinery, floor, hoses, gloves, warning labels, sawdust, shadows, knots, stamps, or wood grain.
- Do not invent hidden or occluded objects.
- Boxes should be tight around the visible object or visible defect.
- If there are no objects for a class, omit that class.

Final answer must contain only valid YOLO label rows.
Allowed class IDs are: 0, 1, 2, 4, 6, 7, 13.
Every row must have exactly five values:
class_id x_center y_center width height
"""

BOARD_SPLIT_PROMPT_V2 = """You are creating high-precision YOLO object-detection labels for a top-down pallet image.

Return ONLY YOLO label lines in this exact format:
<class_id> <x_center> <y_center> <width> <height>

Use normalized coordinates from 0 to 1.
Do not output explanations, comments, markdown, headers, blank lines, confidence scores, or JSON.
One object per line.
Every row must have exactly 5 fields.

Allowed classes:
0 = entire pallet
1 = regular top deck board
2 = horizontal stringer
4 = lead board / outside edge board
6 = broken board or broken stringer
7 = major defect crack or hole
13 = protruding nail

Image and pallet type:
The image is always top-down.
The image is expected to be 1024x1024 pixels.
This pallet is rotated 90 degrees.
The top deck boards appear vertical in the image and are arranged from left to right.

Critical board-count rule:
This specific pallet type must have exactly 9 vertical top deck-board boxes total.
The 9 vertical deck boards must be ordered left to right as:

Board 1: class 4 = far-left lead board / outside edge board
Board 2: class 1 = regular board immediately next to the far-left lead board
Board 3: class 1 = regular board
Board 4: class 1 = regular board
Board 5: class 1 = regular board
Board 6: class 1 = regular board
Board 7: class 1 = regular board
Board 8: class 1 = regular board immediately next to the far-right lead board
Board 9: class 4 = far-right lead board / outside edge board

Therefore:
- class 4 count must be exactly 2
- class 1 count must be exactly 7
- class 1 plus class 4 count must be exactly 9

Critical left-edge rule:
The far-left visible wide area is not one board.
It is a pair of two adjacent vertical boards:
- left board of the pair = class 4
- right board of the pair = class 1

Even if the seam is subtle and there is no large dark gap, split this left pair into two separate boxes.
Use visual cues such as:
- subtle vertical seam
- change in grain direction
- nail-column spacing
- board color transition
- consistent expected 9-board layout

Do NOT merge the far-left lead board with the adjacent regular board.

Critical right-edge rule:
The far-right visible wide area is not three boards.
It is a pair of two adjacent vertical boards:
- left board of the pair = class 1
- right board of the pair = class 4

Split the right pair at the visible seam.
Do NOT split the right pair into three boxes.
Do NOT merge the right regular board with the far-right lead board.

Board geometry rules:
- All 9 deck-board boxes are vertical.
- All 9 deck-board boxes should have similar y extents.
- Their x centers must increase left to right.
- Adjacent board boxes should not significantly overlap.
- Gaps between boards should not be included in board boxes.
- Do not label horizontal stringers as vertical boards.
- Do not label machine rails or conveyor parts as boards.

Stringer rules:
- Label horizontal stringers as class 2.
- This pallet type usually has three horizontal stringers visible across the pallet:
  top stringer, middle stringer, bottom stringer.
- Stringers run horizontally in the image.
- Use tight boxes around the visible stringer extent.
- Do not confuse conveyor rails or machine parts with stringers.

Pallet rule:
- Label the entire visible pallet as class 0.
- The class 0 box should include the visible pallet wood only.
- Do not include conveyor rails, floor, hoses, gloves, warning labels, or machinery.

Defect rules:
- Label broken board or broken stringer regions as class 6 only when clearly visible.
- Label major defect cracks or holes as class 7 only when clearly visible.
- Label protruding nails only as class 13.
- Do not label flush nails.
- Do not label knots, stains, stamps, shadows, sawdust, or wood grain as defects.
- Do not invent hidden or occluded defects.

Optional local board hints:
The app may provide candidate vertical board x-ranges from image preprocessing.
These are hints only, not ground truth.
Use them to avoid merging or over-splitting the board pairs.
If the hints conflict with the visual image, prefer the image, but still obey the required 9-board layout.

Candidate vertical board x-ranges:
{{BOARD_X_RANGE_HINTS}}

Final self-check before output:
Before giving the final YOLO lines, silently verify:
- exactly one class 0 line
- exactly seven class 1 lines
- exactly two class 4 lines
- exactly nine total class 1/class 4 deck-board lines
- leftmost deck board is class 4
- rightmost deck board is class 4
- second-leftmost deck board is class 1
- second-rightmost deck board is class 1
- three class 2 stringer lines unless a stringer is not visible
- no markdown
- no explanations
- no JSON
- no confidence scores

Final answer must contain only valid YOLO label rows.

End prompt.
"""

STRUCTURE_ONLY_PROMPT_V2 = """You are creating high-precision YOLO object-detection labels for a top-down pallet image.

Return ONLY YOLO label lines in this exact format:
<class_id> <x_center> <y_center> <width> <height>

Use normalized coordinates from 0 to 1.
Do not output explanations, comments, markdown, headers, blank lines, confidence scores, or JSON.
One object per line.
Every row must have exactly 5 fields.

Allowed classes:
0 = entire pallet
1 = regular top deck board
2 = horizontal stringer
4 = lead board / outside edge board

Ignore all defects and nails in this pass.
Do not output classes 6, 7, or 13.

Image and pallet type:
The image is always top-down.
The image is expected to be 1024x1024 pixels.
This pallet is rotated 90 degrees.
The top deck boards appear vertical in the image and are arranged from left to right.

Critical board-count rule:
This specific pallet type must have exactly 9 vertical top deck-board boxes total.
The 9 vertical deck boards must be ordered left to right as:

Board 1: class 4 = far-left lead board / outside edge board
Board 2: class 1 = regular board immediately next to the far-left lead board
Board 3: class 1 = regular board
Board 4: class 1 = regular board
Board 5: class 1 = regular board
Board 6: class 1 = regular board
Board 7: class 1 = regular board
Board 8: class 1 = regular board immediately next to the far-right lead board
Board 9: class 4 = far-right lead board / outside edge board

Therefore:
- class 0 count must be exactly 1
- class 4 count must be exactly 2
- class 1 count must be exactly 7
- class 1 plus class 4 count must be exactly 9
- class 2 usually has exactly 3 visible horizontal stringers

Critical left-edge rule:
The far-left visible wide area is not one board.
It is a pair of two adjacent vertical boards:
- left board of the pair = class 4
- right board of the pair = class 1

Even if the seam is subtle and there is no large dark gap, split this left pair into two separate boxes.
Use visual cues such as:
- subtle vertical seam
- change in grain direction
- nail-column spacing
- board color transition
- consistent expected 9-board layout

Do NOT merge the far-left lead board with the adjacent regular board.

Critical right-edge rule:
The far-right visible wide area is not three boards.
It is a pair of two adjacent vertical boards:
- left board of the pair = class 1
- right board of the pair = class 4

Split the right pair at the visible seam.
Do NOT split the right pair into three boxes.
Do NOT merge the right regular board with the far-right lead board.

Board geometry rules:
- All 9 deck-board boxes are vertical.
- All 9 deck-board boxes should have similar y extents.
- Their x centers must increase left to right.
- Adjacent board boxes should not significantly overlap.
- Gaps between boards should not be included in board boxes.
- Do not label horizontal stringers as vertical boards.
- Do not label machine rails or conveyor parts as boards.

Stringer rules:
- Label horizontal stringers as class 2.
- This pallet type usually has three horizontal stringers visible across the pallet:
  top stringer, middle stringer, bottom stringer.
- Stringers run horizontally in the image.
- Use tight boxes around the visible stringer extent.
- Do not confuse conveyor rails or machine parts with stringers.

Pallet rule:
- Label the entire visible pallet as class 0.
- The class 0 box should include the visible pallet wood only.
- Do not include conveyor rails, floor, hoses, gloves, warning labels, or machinery.

Optional local board hints:
The app may provide candidate vertical board x-ranges from image preprocessing.
These are hints only, not ground truth.
Use them to avoid merging or over-splitting the board pairs.
If the hints conflict with the visual image, prefer the image, but still obey the required 9-board layout.

Candidate vertical board x-ranges:
{{BOARD_X_RANGE_HINTS}}

Final self-check before output:
Before giving the final YOLO lines, silently verify:
- exactly one class 0 line
- exactly seven class 1 lines
- exactly two class 4 lines
- exactly nine total class 1/class 4 deck-board lines
- leftmost deck board is class 4
- rightmost deck board is class 4
- second-leftmost deck board is class 1
- second-rightmost deck board is class 1
- three class 2 stringer lines unless a stringer is not visible
- no markdown
- no explanations
- no JSON
- no confidence scores

Final answer must contain only valid YOLO label rows.
"""

DEFECT_ONLY_PROMPT_V2 = """You are creating high-precision YOLO object-detection labels for safety defects in a top-down pallet image.

Return ONLY YOLO label lines in this exact format:
<class_id> <x_center> <y_center> <width> <height>

Use normalized coordinates from 0 to 1.
Do not output explanations, comments, markdown, headers, blank lines, confidence scores, or JSON.
One object per line.
Every row must have exactly 5 fields.

Allowed defect classes:
6 = broken board or broken stringer
7 = major defect crack or hole
13 = protruding nail

Rules:
- Label class 6 only for clearly broken board or stringer regions.
- Label class 7 only for major cracks or holes.
- Label class 13 only for protruding nails, bent nails, hook-shaped metal fasteners, U-shaped fasteners, or pulled-out/sharp metal.
- Do not label flush nails.
- Do not label normal nail heads.
- Do not label knots, stamps, shadows, sawdust, dirt, gaps, or wood grain.
- Do not label pallet boards or stringers in this pass.
- If there are no visible defects, return an empty response with no lines.

Final answer must contain only valid YOLO rows or be empty.
"""

CORRECTION_PROMPT = """Your previous YOLO label output failed validation.

Validation errors:
{{VALIDATION_ERRORS}}

Previous output:
{{PREVIOUS_OUTPUT}}

Fix the labels.
Return ONLY corrected YOLO rows.
Do not output explanations, markdown, JSON, comments, or confidence scores.
The corrected output must satisfy:
- class 0 count = 1
- class 1 count = 7
- class 4 count = 2
- class 1 + class 4 deck boards = 9
- sorted deck board classes left to right = [4,1,1,1,1,1,1,1,4]
- left pair must be split as class 4 then class 1
- right pair must be split as class 1 then class 4

Use the original instructions and image again, including the candidate board x-range hints:
{{BOARD_X_RANGE_HINTS}}
"""

PROMPT_FILES = {
    "pallet_prompt.default.txt": DEFAULT_PROMPT,
    "pallet_prompt.v2_board_split.txt": BOARD_SPLIT_PROMPT_V2,
    "pallet_prompt.v2_structure_only.txt": STRUCTURE_ONLY_PROMPT_V2,
    "pallet_prompt.v2_defects_only.txt": DEFECT_ONLY_PROMPT_V2,
}

PREFERRED_PROMPT_NAME = "pallet_prompt.v2_board_split.txt"


def ensure_default_prompt(working_dir: Path) -> Path:
    prompt_dir = working_dir / "_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in PROMPT_FILES.items():
        prompt_path = prompt_dir / filename
        if not prompt_path.exists():
            prompt_path.write_text(content, encoding="utf-8")
    return prompt_dir / PREFERRED_PROMPT_NAME


def prompt_path(working_dir: Path, prompt_name: str) -> Path:
    ensure_default_prompt(working_dir)
    return working_dir / "_prompts" / prompt_name


def list_prompt_templates(working_dir: Path) -> list[Path]:
    preferred = ensure_default_prompt(working_dir)
    prompt_dir = working_dir / "_prompts"
    paths = sorted(prompt_dir.glob("*.txt"), key=lambda path: path.name.lower())
    return [preferred] + [path for path in paths if path != preferred]


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def class_map_text(classes: list[str]) -> str:
    lines = []
    for index, name in enumerate(classes):
        if name:
            lines.append(f"{index} = {name}")
    return "\n".join(lines)


def render_prompt(
    template_text: str,
    image_path: Path,
    image_width: int,
    image_height: int,
    classes: list[str],
    allowed_class_ids: list[int],
    expected_counts: dict[str, int],
    board_x_range_hints: str | None = None,
    extra_replacements: dict[str, Any] | None = None,
) -> str:
    replacements: dict[str, Any] = {
        "IMAGE_WIDTH": image_width,
        "IMAGE_HEIGHT": image_height,
        "IMAGE_STEM": image_path.stem,
        "CLASS_MAP": class_map_text(classes),
        "ALLOWED_CLASS_IDS": ", ".join(str(value) for value in allowed_class_ids),
        "EXPECTED_COUNTS": yaml.safe_dump(expected_counts, sort_keys=True).strip(),
        "BOARD_X_RANGE_HINTS": board_x_range_hints or "No candidate x-range hints were available.",
    }
    if extra_replacements:
        replacements.update(extra_replacements)

    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))

    normalized = rendered.lower()
    if "class_id x_center y_center width height" not in normalized and "exactly 5" not in normalized:
        rendered = rendered.rstrip() + "\n\n" + STRICT_YOLO_REMINDER + "\n"
    return rendered
