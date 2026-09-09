# ruff: noqa: ISC004
"""Source-level corrections for Paragram (idempotent; works on this checkout or a fresh clone).

    python patch_paragram.py [/path/to/paragram-checkout]

1. Relative clipping pad.  Paragram clips every Voronoi cell to the point bounding box padded by
   an *absolute* 1.0 unit (`root_bounds.lower - 1.0f`).  Voronoi faces lying outside that box are
   dropped, and with them Delaunay edges near the convex hull: the loss depends on the model's
   units (severe for extents of 1-10, negligible for tiny or huge models).  This adds a
   `bbox_pad` argument to `voronoi_diagram` / `power_diagram`: pad = bbox_pad x (largest extent
   of the point set).  The default -1.0 keeps the legacy behaviour.  A larger box makes the
   BVH query of unbounded (hull) cells visit more points, so it costs time on big volumetric
   clouds; 10 recovers essentially all hull tetrahedra on the meshes tested.

2. Cell budget.  `MAX_PLANES` / `MAX_VERTS` (64) become overridable at build time through the
   environment variables PARAGRAM_MAX_PLANES / PARAGRAM_MAX_VERTS (read by _backend.py and
   passed as -D flags).  Surface samples have Delaunay degrees around 30 with a heavy tail, so
   64 overflows; 128 halves the overflow failures at the cost of scratch memory.  Vertex/plane
   indices are bytes with 255 reserved, so values up to 254 are representable.
"""

from __future__ import annotations

import os
import sys

EDITS: list[tuple[str, list[tuple[str, str]]]] = [
    # ------------------------------------------------------------------ CUDA: relative pad
    (
        "paragram/csrc/laguerre/voronoi_ultra.cu",
        [
            (
                "                    std::optional<torch::Tensor> initial_guesses_offsets_in,\n"
                "                    int leaf_size)\n",
                "                    std::optional<torch::Tensor> initial_guesses_offsets_in,\n"
                "                    int leaf_size,\n"
                "                    float bbox_pad)\n",
            ),
            (
                "    root_bounds.lower = root_bounds.lower - 1.0f;\n"
                "    root_bounds.upper = root_bounds.upper + 1.0f;\n",
                "    // Clipping box for the cells.  Legacy (bbox_pad < 0): pad by an absolute 1.0.  Otherwise pad by\n"
                "    // bbox_pad times the largest extent so that Voronoi faces of hull cells are not cut off.\n"
                "    {\n"
                "        float pad = 1.0f;\n"
                "        if (bbox_pad >= 0.0f)\n"
                "        {\n"
                "            float ext = fmaxf(root_bounds.upper.x - root_bounds.lower.x,\n"
                "                              fmaxf(root_bounds.upper.y - root_bounds.lower.y,\n"
                "                                    root_bounds.upper.z - root_bounds.lower.z));\n"
                "            pad = bbox_pad * ext;\n"
                "        }\n"
                "        root_bounds.lower = root_bounds.lower - pad;\n"
                "        root_bounds.upper = root_bounds.upper + pad;\n"
                "    }\n",
            ),
        ],
    ),
    (
        "paragram/csrc/laguerre/laguerre_ultra.cu",
        [
            (
                "                             std::optional<torch::Tensor> initial_guesses_offsets_in,\n"
                "                             int leaf_size)\n",
                "                             std::optional<torch::Tensor> initial_guesses_offsets_in,\n"
                "                             int leaf_size,\n"
                "                             float bbox_pad)\n",
            ),
            (
                "    root_bounds.lower = root_bounds.lower - 1.0f;\n"
                "    root_bounds.upper = root_bounds.upper + 1.0f;\n",
                "    // Clipping box for the cells.  Legacy (bbox_pad < 0): pad by an absolute 1.0.  Otherwise pad by\n"
                "    // bbox_pad times the largest extent so that Voronoi faces of hull cells are not cut off.\n"
                "    {\n"
                "        float pad = 1.0f;\n"
                "        if (bbox_pad >= 0.0f)\n"
                "        {\n"
                "            float ext = fmaxf(root_bounds.upper.x - root_bounds.lower.x,\n"
                "                              fmaxf(root_bounds.upper.y - root_bounds.lower.y,\n"
                "                                    root_bounds.upper.z - root_bounds.lower.z));\n"
                "            pad = bbox_pad * ext;\n"
                "        }\n"
                "        root_bounds.lower = root_bounds.lower - pad;\n"
                "        root_bounds.upper = root_bounds.upper + pad;\n"
                "    }\n",
            ),
        ],
    ),
    (
        "paragram/csrc/laguerre/laguerre.h",
        [
            (
                "                             std::optional<torch::Tensor> initial_guesses_offsets = std::nullopt,\n"
                "                             int leaf_size = -1);\n",
                "                             std::optional<torch::Tensor> initial_guesses_offsets = std::nullopt,\n"
                "                             int leaf_size = -1,\n"
                "                             float bbox_pad = -1.0f);\n",
            ),
            (
                "                    std::optional<torch::Tensor> initial_guesses_offsets = std::nullopt,\n"
                "                    int leaf_size = -1);\n",
                "                    std::optional<torch::Tensor> initial_guesses_offsets = std::nullopt,\n"
                "                    int leaf_size = -1,\n"
                "                    float bbox_pad = -1.0f);\n",
            ),
        ],
    ),
    (
        "paragram/csrc/ext.cpp",
        [
            (
                '          py::arg("initial_guesses_offsets") = py::none(),\n'
                '          py::arg("leaf_size") = -1);\n'
                "\n"
                '    m.def("build_voronoi_ultra",',
                '          py::arg("initial_guesses_offsets") = py::none(),\n'
                '          py::arg("leaf_size") = -1,\n'
                '          py::arg("bbox_pad") = -1.0f);\n'
                "\n"
                '    m.def("build_voronoi_ultra",',
            ),
            (
                '          py::arg("initial_guesses_offsets") = py::none(),\n'
                '          py::arg("leaf_size") = -1);\n'
                "}\n",
                '          py::arg("initial_guesses_offsets") = py::none(),\n'
                '          py::arg("leaf_size") = -1,\n'
                '          py::arg("bbox_pad") = -1.0f);\n'
                "}\n",
            ),
        ],
    ),
    # ------------------------------------------------------------------ Python API
    (
        "paragram/_api.py",
        [
            (
                "    initial_guesses_offsets: Optional[torch.Tensor] = None,\n"
                "    leaf_size: int = -1,\n"
                ") -> Diagram:\n"
                "    points = _validate_points(points)\n"
                "    weights = _validate_weights(weights, points.shape[0], points.device)\n",
                "    initial_guesses_offsets: Optional[torch.Tensor] = None,\n"
                "    leaf_size: int = -1,\n"
                "    bbox_pad: float = -1.0,\n"
                ") -> Diagram:\n"
                '    """bbox_pad: clipping-box padding as a multiple of the point-set extent (default -1 = legacy\n'
                '    absolute 1.0).  Use e.g. 10 so Voronoi faces of hull cells are not cut off."""\n'
                "    points = _validate_points(points)\n"
                "    weights = _validate_weights(weights, points.shape[0], points.device)\n",
            ),
            (
                "        initial_guesses_offsets=initial_guesses_offsets,\n"
                "        leaf_size=leaf_size,\n"
                "    )\n"
                "    return Diagram(adjacency=adjacency, offsets=offsets, status=status)\n"
                "\n"
                "\n"
                "def voronoi_diagram(",
                "        initial_guesses_offsets=initial_guesses_offsets,\n"
                "        leaf_size=leaf_size,\n"
                "        bbox_pad=float(bbox_pad),\n"
                "    )\n"
                "    return Diagram(adjacency=adjacency, offsets=offsets, status=status)\n"
                "\n"
                "\n"
                "def voronoi_diagram(",
            ),
            (
                "    initial_guesses_offsets: Optional[torch.Tensor] = None,\n"
                "    leaf_size: int = -1,\n"
                ") -> Diagram:\n"
                "    points = _validate_points(points)\n"
                "    initial_guesses, initial_guesses_offsets = _validate_initial_guesses(\n",
                "    initial_guesses_offsets: Optional[torch.Tensor] = None,\n"
                "    leaf_size: int = -1,\n"
                "    bbox_pad: float = -1.0,\n"
                ") -> Diagram:\n"
                '    """bbox_pad: clipping-box padding as a multiple of the point-set extent (default -1 = legacy\n'
                '    absolute 1.0).  Use e.g. 10 so Voronoi faces of hull cells are not cut off."""\n'
                "    points = _validate_points(points)\n"
                "    initial_guesses, initial_guesses_offsets = _validate_initial_guesses(\n",
            ),
            (
                "        initial_guesses_offsets=initial_guesses_offsets,\n"
                "        leaf_size=leaf_size,\n"
                "    )\n"
                "    return Diagram(adjacency=adjacency, offsets=offsets, status=status)\n",
                "        initial_guesses_offsets=initial_guesses_offsets,\n"
                "        leaf_size=leaf_size,\n"
                "        bbox_pad=float(bbox_pad),\n"
                "    )\n"
                "    return Diagram(adjacency=adjacency, offsets=offsets, status=status)\n",
            ),
        ],
    ),
    # ------------------------------------------------------------------ cell budget
    (
        "paragram/csrc/laguerre/voronoi_convex_cell.cuh",
        [
            (
                "#define MAX_PLANES 64\n#define MAX_VERTS 64\n",
                "#ifndef MAX_PLANES\n#define MAX_PLANES 64 // overridable: PARAGRAM_MAX_PLANES (build env), <= 254\n#endif\n"
                "#ifndef MAX_VERTS\n#define MAX_VERTS 64 // overridable: PARAGRAM_MAX_VERTS (build env), <= 254\n#endif\n",
            )
        ],
    ),
    (
        "paragram/csrc/laguerre/faster_convex_cell.cuh",
        [
            (
                "#define MAX_PLANES 64 // Can't change this without changing convex_cell.cuh garbage collections\n"
                "#define MAX_VERTS 64\n",
                "#ifndef MAX_PLANES\n#define MAX_PLANES 64 // overridable: PARAGRAM_MAX_PLANES (build env), <= 254\n#endif\n"
                "#ifndef MAX_VERTS\n#define MAX_VERTS 64 // overridable: PARAGRAM_MAX_VERTS (build env), <= 254\n#endif\n",
            )
        ],
    ),
    (
        "paragram/_backend.py",
        [
            (
                '        opt_level = "-O0" if (FAST_COMPILE or DEBUG_CUDA) else "-O3"\n'
                '        extra_cflags = [opt_level, "-Wno-attributes"]\n'
                '        extra_cuda_cflags = [opt_level, "--extended-lambda"]\n',
                '        opt_level = "-O0" if (FAST_COMPILE or DEBUG_CUDA) else "-O3"\n'
                '        extra_cflags = [opt_level, "-Wno-attributes"]\n'
                '        extra_cuda_cflags = [opt_level, "--extended-lambda"]\n'
                "        # Optional cell budget override (default 64/64): surface point clouds have Delaunay degrees\n"
                "        # around 30 with a heavy tail, which overflows 64 planes/vertices.\n"
                '        for env_name, macro in (("PARAGRAM_MAX_PLANES", "MAX_PLANES"), ("PARAGRAM_MAX_VERTS", "MAX_VERTS")):\n'
                "            value = os.getenv(env_name)\n"
                "            if value:\n"
                "                if not (4 <= int(value) <= 254):\n"
                '                    raise ValueError(f"{env_name} must be in [4, 254], got {value}")\n'
                '                extra_cflags.append(f"-D{macro}={int(value)}")\n'
                '                extra_cuda_cflags.append(f"-D{macro}={int(value)}")\n'
                '                name += f"_{macro.lower()}{int(value)}"\n'
                "        build_dir = _get_build_directory(name, verbose=False)\n",
            )
        ],
    ),
]


def patch(root: str) -> bool:
    ok = True
    for rel, edits in EDITS:
        path = os.path.join(root, rel)
        with open(path) as f:
            src = f.read()
        changed = 0
        for old, new in edits:
            if new in src:
                continue  # already applied
            if old not in src:
                print(f"{rel}: anchor not found (upstream changed?):\n{old}")
                ok = False
                continue
            src = src.replace(old, new, 1)
            changed += 1
        if changed:
            with open(path, "w") as f:
                f.write(src)
        print(f"{rel}: {changed} edit(s) applied" if changed else f"{rel}: already patched")
    return ok


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    sys.exit(0 if patch(root) else 1)
