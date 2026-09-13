"""The SLURM jobs must send options the studies actually accept.

A job is the only place a setting can be overridden without a config file being involved, so the
same static check applied to configs is applied to the `key=value` arguments the jobs pass to
`main.py`. A typo here costs a cluster queue slot rather than a red test.
"""

import os
import re

import pytest

import main as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which study each job ultimately drives. The `experiment=` in the file may be a shell variable
# (FULL=1 picks benchmark_full, UPSAMPLE=densify picks scaling_densify), but both variants of a
# job run the same module.
JOBS = {
    "script/run_jitter.sbatch": "jitter_study",
    "script/run_scaling.sbatch": "scaling_study",
    "script/run_benchmark.sbatch": "benchmark",
    "script/run_triangulate.sbatch": "triangulate",
}


def invocations(text: str) -> list[str]:
    """The `python main.py ...` commands, with line continuations joined.

    Each is cut at the first redirection, so that what follows the pipe (`|| rc=$?`, `tee`) is not
    mistaken for an override.
    """
    joined = re.sub(r"\\\n\s*", " ", text)
    out = []
    for line in joined.splitlines():
        if "python main.py" not in line:
            continue
        command = line[line.index("python main.py") :]
        out.append(re.split(r"\s(?:2>&1|\||>|\|\|)", command)[0])
    return out


@pytest.mark.parametrize("job,module", sorted(JOBS.items()))
def test_job_overrides_are_real_options(job, module):
    path = os.path.join(ROOT, job)
    assert os.path.exists(path), f"{job} is missing"
    accepted = M.declared_flags(module)
    assert accepted, f"no argparse options found for {module}"

    checked = 0
    for command in invocations(open(path).read()):
        # `$COMMON` and `"$@"` expand to further key=value pairs; they are checked where they are
        # defined, below.
        for key in re.findall(r'(?<![\w"$-])([a-z][a-z0-9_]*)=', command):
            if key == "experiment":
                continue
            flag = "--" + key.replace("_", "-")
            assert flag in accepted, f"{job}: src/{module}.py has no {flag}"
            checked += 1
    assert checked, f"{job}: no overrides found -- did the invocation change shape?"


@pytest.mark.parametrize("job,module", sorted(JOBS.items()))
def test_jobs_do_not_append_raw_flags(job, module):
    """`--flag` after `--` appends a second copy of an option instead of replacing it.

    argparse then keeps the last one, so the overridden options win and the ones left out quietly
    keep their config values -- which once sent a run's JSON to the job directory and its CSV to
    the config's, and killed the job on the missing directory.
    """
    for command in invocations(open(os.path.join(ROOT, job)).read()):
        assert " -- " not in command, (
            f"{job}: passes raw flags through `--`; use key=value overrides instead\n  {command.strip()}"
        )
