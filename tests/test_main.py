"""The config-to-arguments translation done by main.py."""

import pytest

import main as M


def test_scalars_and_lists_become_flags():
    argv = M.to_argv({"repeats": 5, "ply": ["a.ply", "b.ply"], "gstar4d_grid": 512})
    assert argv == ["--repeats", "5", "--ply", "a.ply", "b.ply", "--gstar4d-grid", "512"]


def test_booleans_are_bare_flags_and_false_is_dropped():
    assert M.to_argv({"verbose": True, "resume": False}) == ["--verbose"]


def test_none_and_metadata_are_dropped():
    assert M.to_argv({"study": "jitter", "description": "x", "json": None}) == []


def test_extra_is_appended_verbatim():
    argv = M.to_argv({"output": "r.md", "_extra": ["results/a.json"]})
    assert argv == ["--output", "r.md", "results/a.json"]


def test_empty_list_produces_no_flag():
    assert M.to_argv({"methods": []}) == []


def test_parse_value_handles_the_shapes_configs_use():
    assert M.parse_value("[0, 1.0e-9, 1e-6]") == [0, 1.0e-9, 1e-6]
    assert M.parse_value("true") is True
    assert M.parse_value("null") is None
    assert M.parse_value("bin/cgal_delaunay") == "bin/cgal_delaunay"
    assert M.parse_value("1.0e-6") == 1e-6


def test_declared_flags_reads_a_real_study():
    flags = M.declared_flags("jitter_study")
    assert "--jitters" in flags and "--repeats" in flags


def test_fallback_parser_agrees_with_pyyaml_on_every_config():
    """main.py ships a tiny YAML parser for environments without PyYAML.

    The two must return the same thing, or a cluster run without PyYAML would silently drop a
    setting -- which is exactly what happened when the formatter moved a long list onto its own
    line.
    """
    import os
    import sys

    yaml = pytest.importorskip("yaml")

    paths = [os.path.join(M.CONFIG_DIR, "main.yaml")] + [
        os.path.join(M.CONFIG_DIR, "experiments", f"{name}.yaml") for name in M.experiments()
    ]
    saved = sys.modules.get("yaml")
    try:
        for path in paths:
            with open(path) as fh:
                expected = yaml.safe_load(fh) or {}
            sys.modules["yaml"] = None  # force the fallback
            got = M.load_yaml(path)
            sys.modules["yaml"] = saved
            assert got == expected, f"{os.path.basename(path)}: fallback parser disagrees"
    finally:
        sys.modules["yaml"] = saved
