"""Single entry point for every experiment in this repository.

    python main.py --list                                # what can be run
    python main.py experiment=jitter_sweep               # run one, with its recorded settings
    python main.py experiment=jitter_sweep repeats=1     # ... overriding a setting
    python main.py experiment=jitter_sweep -- --help     # the study's own options

An experiment is a YAML file in `configs/experiments/`. It names a `study` (which module runs)
and the arguments that study should receive; `configs/main.yaml` holds the defaults shared by all
of them. Everything an experiment does is therefore visible in one short file instead of being
buried in a job script, and a run can be reproduced by its name alone.

The studies keep their own `argparse` interfaces -- this only builds the argument list for them,
so `python src/jitter_study.py --ply ... --jitters ...` still works exactly as before and the two
paths cannot drift apart.
"""

from __future__ import annotations

import ast
import os
import sys

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

# The studies import one another by plain module name (`import benchmark as T`), so `src` has to
# be importable rather than be a package prefix.  pytest.ini puts it on the path the same way.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

STUDIES = {
    "benchmark": ("benchmark", "Correctness and timing against CGAL, on every dataset"),
    "scaling": ("scaling_study", "Time versus number of points (2k .. 1M)"),
    "jitter": ("jitter_study", "Sweep the size of the input perturbation"),
    "report": ("make_report", "Markdown report and charts from one or more result JSONs"),
    "triangulate": ("triangulate", "Run one method on one PLY file and write the tetrahedra"),
}


def load_yaml(path: str) -> dict:
    """Read a config file.

    Uses PyYAML when it is installed, and falls back to a small parser covering the subset these
    files use -- `key: scalar`, `key: [a, b]`, `key:` followed by `- item` lines, and a value the
    formatter has pushed onto its own indented line -- so that `--list` and a plain run work in an
    environment without PyYAML. There is no nesting in these configs, and the fallback does not
    support any. `tests/test_main.py` checks the two parsers agree on every config in the repo.
    """
    try:
        import yaml

        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        pass
    with open(path) as fh:
        lines: list[str] = []
        for raw in fh:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            # A value the formatter pushed onto its own indented line: prettier rewrites a long
            # `methods: [a, b, c]` that way, and reading it as an empty value would silently drop
            # the key.
            if (
                lines
                and lines[-1].endswith(":")
                and line.startswith((" ", "\t"))
                and not line.lstrip().startswith("- ")
            ):
                lines[-1] += " " + line.strip()
                continue
            lines.append(line)

    out: dict = {}
    key = None
    for line in lines:
        if line.lstrip().startswith("- ") and key is not None:
            out.setdefault(key, [])
            if not isinstance(out[key], list):
                out[key] = []
            out[key].append(parse_scalar(line.lstrip()[2:].strip()))
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        out[key] = None if value == "" else parse_value(value)
    return out


def parse_scalar(text: str):
    """YAML-ish scalar: booleans, null, numbers, everything else a string."""
    text = text.strip()
    # Quoted means "a string", whatever it looks like: the size ranges are written "200000" so
    # that they stay strings next to "2000:200000:10000".
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    low = text
    if low.lower() in ("true", "yes"):
        return True
    if low.lower() in ("false", "no"):
        return False
    if low.lower() in ("null", "~", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(low)
        except ValueError:
            pass
    return low


def parse_value(text: str):
    """A scalar, or a `[a, b, c]` list."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [parse_scalar(p) for p in inner.split(",")] if inner else []
    return parse_scalar(text)


def to_argv(config: dict) -> list[str]:
    """Turn a config into the argument list its study's argparse expects.

    `key: value` becomes `--key value`, with underscores spelled as dashes; a list becomes several
    values after one flag; `true` becomes a bare flag and `false` or an empty value drops it.
    `_extra` is appended verbatim, for the rare flag that does not fit this shape.
    """
    argv: list[str] = []
    for key, value in config.items():
        if key in ("study", "description", "_extra") or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            if value:
                argv += [flag] + [str(v) for v in value]
        else:
            argv += [flag, str(value)]
    argv += [str(v) for v in (config.get("_extra") or [])]
    return argv


def declared_flags(module_name: str) -> set[str] | None:
    """Option strings a study's argparse accepts, read statically from its source.

    Read rather than imported: importing a study pulls in torch and CUDA, which must not be needed
    merely to assemble an argument list. Returns None when the source cannot be parsed, in which
    case nothing is filtered and argparse reports any mistake itself.
    """
    path = os.path.join(SRC_DIR, module_name + ".py")
    try:
        with open(path) as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return None
    flags = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("-"):
                    flags.add(arg.value)
    return flags or None


def experiments() -> dict[str, str]:
    """Available experiment names and their one-line descriptions."""
    d = os.path.join(CONFIG_DIR, "experiments")
    out = {}
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.endswith((".yaml", ".yml")):
            cfg = load_yaml(os.path.join(d, name))
            out[os.path.splitext(name)[0]] = str(cfg.get("description") or cfg.get("study") or "")
    return out


def usage() -> int:
    print(__doc__.strip())
    print("\nExperiments (configs/experiments/):")
    for name, description in experiments().items():
        print(f"  {name:22s} {description}")
    print("\nStudies (the module each experiment runs):")
    for name, (module, description) in STUDIES.items():
        print(f"  {name:22s} src/{module}.py -- {description}")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "--list"):
        return usage()

    # everything after a bare `--` is handed to the study untouched
    passthrough: list[str] = []
    if "--" in args:
        cut = args.index("--")
        args, passthrough = args[:cut], args[cut + 1 :]

    overrides = {}
    for arg in args:
        if "=" not in arg:
            print(f"expected key=value, got {arg!r}\n")
            return usage()
        key, _, value = arg.partition("=")
        overrides[key.strip()] = parse_value(value)

    name = overrides.pop("experiment", None)
    if name is None:
        print("no experiment given: pass experiment=<name>\n")
        return usage()

    path = os.path.join(CONFIG_DIR, "experiments", f"{name}.yaml")
    if not os.path.exists(path):
        print(f"no such experiment: {name}\n")
        return usage()

    shared = load_yaml(os.path.join(CONFIG_DIR, "main.yaml"))
    experiment = load_yaml(path)
    config = {**shared, **experiment, **overrides}

    study = config.get("study")
    if study not in STUDIES:
        print(f"{path}: 'study' must be one of {', '.join(STUDIES)}, got {study!r}")
        return 2

    module_name = STUDIES[study][0]
    # A shared default applies only to the studies that accept it -- `make_report.py` takes none
    # of the tool paths, for instance.  Keys set by the experiment file or on the command line are
    # always passed through, so a genuine mistake surfaces as argparse's own error message.
    accepted = declared_flags(module_name)
    if accepted is not None:
        for key in list(config):
            inherited = key in shared and key not in experiment and key not in overrides
            if inherited and "--" + key.replace("_", "-") not in accepted:
                del config[key]

    argv = to_argv(config) + passthrough
    print(f"[main] {name}: python src/{module_name}.py " + " ".join(argv), flush=True)

    module = __import__(module_name)
    sys.argv = [f"src/{module_name}.py"] + argv
    return module.main() or 0


if __name__ == "__main__":
    sys.exit(main())
