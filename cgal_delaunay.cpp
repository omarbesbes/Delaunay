// Parallel 3D Delaunay triangulation with CGAL (Parallel_tag + TBB) as a command-line tool.
//
//   cgal_delaunay <points.f64> <tets.i32>
//
// points.f64 : raw little-endian float64, N*3 values (x y z per point), no header
// tets.i32   : raw little-endian int32, T*4 vertex indices of the finite cells
//
// Prints one line to stdout:
//   build_seconds=<insert> extract_seconds=<cells->array> cells=<T> parallel=<0|1> threads=<k> cgal=<version>
//
// Build (Debian/Ubuntu: apt-get install libcgal-dev libtbb-dev libgmp-dev libmpfr-dev):
//   g++ -O3 -std=c++17 -DCGAL_LINKED_WITH_TBB cgal_delaunay.cpp -o cgal_delaunay -ltbb -ltbbmalloc -lgmp -lmpfr
// Without TBB (sequential): drop -DCGAL_LINKED_WITH_TBB, -ltbb and -ltbbmalloc.
// Set CGAL_THREADS=<k> to cap the number of TBB threads, CGAL_LOCK_GRID=<g> for the lock grid resolution.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/version.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/global_control.h>
#include <tbb/info.h>
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<int32_t, K> Vb;
typedef CGAL::Triangulation_cell_base_3<K> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb, Concurrency_tag> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay;

static double seconds_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s points.f64 tets.i32\n", argv[0]);
        return 2;
    }

    // ---- read points -------------------------------------------------------------------
    FILE *fin = std::fopen(argv[1], "rb");
    if (!fin)
    {
        std::perror("open points");
        return 1;
    }
    std::vector<double> raw;
    {
        double buf[3 * 4096];
        size_t got;
        while ((got = std::fread(buf, sizeof(double), 3 * 4096, fin)) > 0)
            raw.insert(raw.end(), buf, buf + got);
        std::fclose(fin);
    }
    if (raw.size() % 3 != 0)
    {
        std::fprintf(stderr, "points file size is not a multiple of 3 doubles\n");
        return 1;
    }
    const size_t n = raw.size() / 3;

    std::vector<std::pair<K::Point_3, int32_t>> pts;
    pts.reserve(n);
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k)
    {
        lo[k] = std::numeric_limits<double>::max();
        hi[k] = -std::numeric_limits<double>::max();
    }
    for (size_t i = 0; i < n; ++i)
    {
        const double *p = &raw[3 * i];
        for (int k = 0; k < 3; ++k)
        {
            lo[k] = p[k] < lo[k] ? p[k] : lo[k];
            hi[k] = p[k] > hi[k] ? p[k] : hi[k];
        }
        pts.emplace_back(K::Point_3(p[0], p[1], p[2]), static_cast<int32_t>(i));
    }

    // ---- threads -------------------------------------------------------------------------
    int threads = 1;
    bool parallel = false;
#ifdef CGAL_LINKED_WITH_TBB
    parallel = true;
    threads = tbb::info::default_concurrency();
    const char *env_threads = std::getenv("CGAL_THREADS");
    std::unique_ptr<tbb::global_control> gc;
    if (env_threads && std::atoi(env_threads) > 0)
    {
        threads = std::atoi(env_threads);
        gc.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, threads));
    }
#endif

    // ---- triangulate ---------------------------------------------------------------------
    auto t0 = std::chrono::steady_clock::now();
#ifdef CGAL_LINKED_WITH_TBB
    // The lock data structure is a grid over the bounding box used by the parallel insertion.
    // Resolution: CGAL_LOCK_GRID cells per axis (default 50).  Finer grids reduce lock contention
    // for points concentrated on surfaces; coarser ones cost less to set up for small inputs.
    int grid = 50;
    if (const char *g = std::getenv("CGAL_LOCK_GRID"))
        if (std::atoi(g) > 0)
            grid = std::atoi(g);
    Delaunay::Lock_data_structure locks(CGAL::Bbox_3(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]), grid);
    Delaunay dt(K(), &locks);
#else
    const int grid = 0;
    Delaunay dt;
#endif
    dt.insert(pts.begin(), pts.end()); // spatial sort + (parallel) insertion, info() = index
    const double build_seconds = seconds_since(t0);

    // ---- extract finite cells --------------------------------------------------------------
    auto t1 = std::chrono::steady_clock::now();
    std::vector<int32_t> out;
    out.reserve(dt.number_of_finite_cells() * 4);
    for (auto c = dt.finite_cells_begin(); c != dt.finite_cells_end(); ++c)
        for (int i = 0; i < 4; ++i)
            out.push_back(c->vertex(i)->info());
    const double extract_seconds = seconds_since(t1);

    FILE *fout = std::fopen(argv[2], "wb");
    if (!fout)
    {
        std::perror("open tets");
        return 1;
    }
    if (!out.empty())
        std::fwrite(out.data(), sizeof(int32_t), out.size(), fout);
    std::fclose(fout);

    std::printf("build_seconds=%.6f extract_seconds=%.6f cells=%zu parallel=%d threads=%d lock_grid=%d cgal=%s\n",
                build_seconds, extract_seconds, out.size() / 4, parallel ? 1 : 0, threads, parallel ? grid : 0, CGAL_VERSION_STR);
    return 0;
}
