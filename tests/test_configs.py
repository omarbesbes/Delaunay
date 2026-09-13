"""Every experiment in configs/ must be runnable.

These checks need neither torch nor a GPU: the study modules are parsed rather than imported, so
a laptop and CI catch a broken config before a cluster job does.
"""

import os

import pytest

import main as M

EXPERIMENTS = sorted(M.experiments())


def test_there_are_experiments():
    assert EXPERIMENTS, "configs/experiments/ is empty"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_experiment_names_a_known_study(name):
    cfg = M.load_yaml(os.path.join(M.CONFIG_DIR, "experiments", f"{name}.yaml"))
    assert cfg.get("study") in M.STUDIES, f"{name}: unknown study {cfg.get('study')!r}"
    assert cfg.get("description"), f"{name}: needs a one-line description for `main.py --list`"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_generated_flag_exists(name):
    """The flags an experiment produces must be accepted by the study that receives them.

    This is the check that stops a config and a study's argparse from drifting apart -- renaming an
    option in a study breaks this test rather than a job three hours in.
    """
    shared = M.load_yaml(os.path.join(M.CONFIG_DIR, "main.yaml"))
    experiment = M.load_yaml(os.path.join(M.CONFIG_DIR, "experiments", f"{name}.yaml"))
    config = {**shared, **experiment}
    module_name = M.STUDIES[config["study"]][0]

    accepted = M.declared_flags(module_name)
    assert accepted, f"no argparse options found in src/{module_name}.py"
    for key in list(config):
        if key in shared and key not in experiment:
            if "--" + key.replace("_", "-") not in accepted:
                del config[key]

    unknown = [a for a in M.to_argv(config) if a.startswith("--") and a not in accepted]
    assert not unknown, f"{name}: src/{module_name}.py has no {unknown}"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_referenced_data_files_exist(name):
    cfg = M.load_yaml(os.path.join(M.CONFIG_DIR, "experiments", f"{name}.yaml"))
    for path in cfg.get("ply") or []:
        assert os.path.exists(path), f"{name}: missing input {path}"
