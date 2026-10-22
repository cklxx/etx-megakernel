"""Keep the owner-facing GLM review checklist compact and executable."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / 'docs/glm/PARALLEL-DESIGN.md'


def test_review_checklist_format():
    text = DOCUMENT.read_text()
    lines = text.splitlines()
    assert len(lines) < 120
    assert '—' not in text
    assert not re.search(r'^#{4}', text, re.MULTILINE)
    assert '## Contents' in lines[:5]
    assert '| step | command | what it proves | PASS number | failure action | GPU minutes |' in lines
    rows = [line for line in lines if re.match(r'^\| \d+ ', line)]
    assert [int(row.split()[1]) for row in rows] == list(range(1, len(rows) + 1))
    assert len(rows) == 11
    for row in rows:
        fields = row.split('|')[1:-1]
        assert len(fields) == 6
        assert all(field.strip() for field in fields)
    inside_code = False
    for line in lines:
        if line.startswith('```'):
            inside_code = not inside_code
        elif line and not inside_code:
            assert line.startswith(('#', '|', '- ')), line
    assert not inside_code
    conditions = text.split('## Conditions\n', 1)[1].split('## Commands', 1)[0]
    assert '| scope | condition |' in conditions
    for match in re.finditer(r'U \(unverified\)', text.split('## Conditions')[0]):
        assert text[match.end():].startswith(':'), 'Every unknown needs its reason'


def test_review_command_blocks_are_complete_shell():
    text = DOCUMENT.read_text()
    blocks = re.findall(r'```sh\n(.*?)\n```', text, re.DOTALL)
    assert len(blocks) == 4
    for number, block in enumerate(blocks, 1):
        assert f'### C{number}\n```sh\n{block}\n```' in text
        assert f'[C{number}](#c{number})' in text
        assert '...' not in block and '…' not in block
        subprocess.run(['sh', '-n'], input=block, text=True, check=True)
    assert (ROOT / 'docs/glm/PARALLEL-CAPACITY-2026-10-22.md').is_file()
