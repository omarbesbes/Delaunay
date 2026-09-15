"""The shell scripts must be shell scripts.

A bad edit once prepended the text of an error message to build_tools.sh, pushing the shebang to
line 4. bash then executed the message -- which contained `bash script/build_tools.sh` -- and the
result was a fork bomb on the cluster. `bash -n` did not catch it, because the mangled file was
still valid shell. These checks would have.
"""

import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(
    os.path.join("script", f)
    for f in os.listdir(os.path.join(ROOT, "script"))
    if f.endswith((".sh", ".sbatch"))
)
# env.sh is sourced, never executed, so it carries no shebang.
SOURCED = {"script/env.sh"}


def test_scripts_are_found():
    assert len(SCRIPTS) >= 6, SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS)
def test_starts_with_shebang_or_comment(path):
    """The first line must be a shebang (executed scripts) or a comment (sourced ones).

    Anything else means the top of the file has been clobbered.
    """
    first = open(os.path.join(ROOT, path)).readline().rstrip("\n")
    if path in SOURCED:
        assert first.startswith("#"), f"{path}: sourced file should start with a comment"
    else:
        assert first == "#!/bin/bash", f"{path}: first line is {first!r}, not a shebang"


@pytest.mark.parametrize("path", SCRIPTS)
def test_parses(path):
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, path)], capture_output=True, text=True)
    assert r.returncode == 0, f"{path}: {r.stderr.strip()}"


@pytest.mark.parametrize("path", [p for p in SCRIPTS if p.endswith(".sbatch")])
def test_sbatch_directives_come_before_any_command(path):
    """#SBATCH lines are only read while the file's leading comment block lasts."""
    seen_command = None
    for n, raw in enumerate(open(os.path.join(ROOT, path)), 1):
        line = raw.strip()
        if line.startswith("#SBATCH"):
            assert seen_command is None, (
                f"{path}:{n}: #SBATCH after a command at line {seen_command} -- SLURM will ignore it"
            )
        elif line and not line.startswith("#"):
            seen_command = seen_command or n
