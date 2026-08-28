from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from gear_sonic.utils.data_collection.lerobot_xlsx_annotations import (
    AnnotationError,
    load_annotations,
)


HEADERS = [
    "episode",
    "Valid",
    "subtask1",
    "time1",
    "subtask2",
    "time2",
    "subtask3",
    "time3",
    "subtask4",
    "Full Prompt",
]


@dataclass(frozen=True)
class InlineText:
    value: str


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def write_xlsx(path: Path, rows: list[list[object]]) -> Path:
    shared: list[str] = []
    shared_indices: dict[str, int] = {}

    def shared_index(value: str) -> int:
        if value not in shared_indices:
            shared_indices[value] = len(shared)
            shared.append(value)
        return shared_indices[value]

    row_xml: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_number, value in enumerate(row, start=1):
            if value is None or value == "":
                continue
            reference = f"{_column_name(column_number)}{row_number}"
            if isinstance(value, InlineText):
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is><t>{escape(value.value)}</t></is></c>'
                )
            elif isinstance(value, str):
                cells.append(f'<c r="{reference}" t="s"><v>{shared_index(value)}</v></c>')
            elif isinstance(value, bool):
                cells.append(f'<c r="{reference}" t="b"><v>{int(value)}</v></c>')
            else:
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>'
    )
    shared_strings = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared)}" uniqueCount="{len(shared)}">'
        + "".join(f"<si><t>{escape(value)}</t></si>" for value in shared)
        + "</sst>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Annotations" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared_strings)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    return path


def valid_row(*, episode: object = 1, valid: object = 1) -> list[object]:
    return [
        episode,
        valid,
        "approach brown table",
        1.0,
        "pick the cup",
        2.0,
        "turn left and approach the trash bin",
        3.0,
        "put it in to the trash bin",
        InlineText("  exact full prompt  "),
    ]


def test_load_annotations_selects_valid_rows_and_preserves_prompts(tmp_path: Path) -> None:
    xlsx = write_xlsx(
        tmp_path / "annotations.xlsx",
        [HEADERS, [0, 0], valid_row(valid=True), [None] * 10],
    )

    rows = load_annotations(xlsx, expected_episodes={0, 1})

    assert [row.episode for row in rows] == [1]
    assert rows[0].subtasks == (
        "approach brown table",
        "pick the cup",
        "turn left and approach the trash bin",
        "put it in to the trash bin",
    )
    assert rows[0].boundaries_s == (1.0, 2.0, 3.0)
    assert rows[0].full_prompt == "exact full prompt"


@pytest.mark.parametrize(
    ("rows", "expected_episodes", "match"),
    [
        ([HEADERS[:-1], valid_row()[:-1]], {1}, "missing required columns"),
        (
            [[*HEADERS[:-1], "episode"], valid_row()],
            {1},
            "duplicate header",
        ),
        ([HEADERS, valid_row(), valid_row()], {1}, "duplicate episode"),
        ([HEADERS, valid_row(episode=1.5)], {1}, "episode must be an integer"),
        ([HEADERS, valid_row(valid=2)], {1}, "valid must be 0 or 1"),
        ([HEADERS, [*valid_row()[:2], None, *valid_row()[3:]]], {1}, "subtask1"),
        (
            [HEADERS, [*valid_row()[:3], "nan", *valid_row()[4:]]],
            {1},
            "time1 must be finite",
        ),
        (
            [HEADERS, [*valid_row()[:5], 0.5, *valid_row()[6:]]],
            {1},
            "strictly increasing",
        ),
        ([HEADERS, valid_row()], {0, 1}, "missing episode rows"),
    ],
)
def test_load_annotations_rejects_invalid_workbooks(
    tmp_path: Path,
    rows: list[list[object]],
    expected_episodes: set[int],
    match: str,
) -> None:
    xlsx = write_xlsx(tmp_path / "invalid.xlsx", rows)

    with pytest.raises(AnnotationError, match=match):
        load_annotations(xlsx, expected_episodes=expected_episodes)
