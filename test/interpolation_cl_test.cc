// cflags: -lOpenCL

#include <fstream>

#include "../include/comput.h"
#include "../include/test.h"

using namespace std;
using namespace gcl;

struct context_fixture {
    context ctxt;
    context_guard cg;

    context get_context() {
        std::vector<platform> ps = platform::get();
        std::vector<device> ds = device::get(ps);

        return context(ds.back());
    }

    context_fixture() :
        ctxt(get_context()),
        cg(ctxt)
    { }
};

struct mark_scanline_fixture : context_fixture {
    program prg;

    mark_scanline_fixture() :
        prg(compile(ifstream("../kernels/interpolation.cl"))) { }

    void test_mark_scanline(
            size_t num_scanlines,
            buffer<col4>& vertices,
            function<void (size_t, col4, col4, col4, col4)> pscan)
    {
        buffer<col4> scan_inf(num_scanlines, host_map);
        buffer<col4> scan_pos(num_scanlines, host_map);
        buffer<cl_uint> output_size{0};

        kernel k(prg, "mark_scanline");

        k.set_buffer(0, vertices);
        k.set_buffer(1, scan_inf);
        k.set_buffer(2, scan_pos);
        k.set_buffer(3, output_size);

        promise() <<
            push(vertices) << unpush(vertices) <<
            push(output_size) << unpush(output_size) <<
            run(k, vertices.size() / 3) <<
            pull(output_size) << unpull(output_size) <<
            pull(scan_inf) << unpull(scan_inf) <<
            pull(scan_pos) << unpull(scan_pos) <<
            wait_until_done;

        assert_true(output_size[0] == num_scanlines);

        for(size_t i = 0; i < num_scanlines; i += 2)
            pscan(i >> 1,
                scan_inf[i], scan_inf[i + 1],
                scan_pos[i], scan_pos[i + 1]);
    }

    static void assert_gs_output(
            buffer<col4>& vertices,
            size_t i, col4 i1, col4 i2, col4 p1, col4 p2)
    {
        // ensure endpoints to be very close to integer
        assert_float_equal(p1[0], float(int64_t(p1[0])));
        assert_float_equal(p2[0], float(int64_t(p2[0])));
        // ensure endpoints of a scanline has the same y
        assert_float_equal(p1[1], p2[1]);
        // ensure left endpoint has a smaller x than right endpoint
        assert_true(p1[0] <= p2[0]);

        col4 t1 =
            vertices[size_t(i1[3]) * 3 + 0] * i1[0] +
            vertices[size_t(i1[3]) * 3 + 1] * i1[1] +
            vertices[size_t(i1[3]) * 3 + 2] * i1[2];

        col4 t2 =
            vertices[size_t(i2[3]) * 3 + 0] * i2[0] +
            vertices[size_t(i2[3]) * 3 + 1] * i2[1] +
            vertices[size_t(i2[3]) * 3 + 2] * i2[2];

        for(size_t d = 0; d < 4; ++d) {
            assert_true(fabs(t1[d] - p1[d]) < 1.f);
            assert_true(fabs(t2[d] - p2[d]) < 1.f);
        }
    }
};

def_test_case_with_fixture(mark_scanline_single, mark_scanline_fixture) {
    buffer<col4> vertices {
        col4 { 2, 4, 0.1, 0 },
        col4 { 1, 6, 0.5, 0 },
        col4 { 8, 9, 0.9, 0 },
    };

    float prev_y, prev_lz, prev_rz;

    test_mark_scanline(10,
        vertices,
        [&](size_t i, col4 i1, col4 i2, col4 p1, col4 p2) {
            assert_gs_output(vertices, i, i1, i2, p1, p2);

            if(i != 0) {
                assert_float_equal(p1[1], prev_y + 1.0f);
                assert_true(p1[2] > prev_lz);
                assert_true(p2[2] > prev_rz);
            }

            prev_y = p2[1];
            prev_lz = p1[2];
            prev_rz = p2[2];
        }
    );
}

def_test_case_with_fixture(mark_scanline_batch, mark_scanline_fixture) {
    buffer<col4> vertices {
        col4 {  3, 20, 0, 0 }, col4 { 93, 48, 0, 0 }, col4 { 70, 36, 0, 0 },
        col4 { 39, 99, 0, 0 }, col4 {  3, 93, 0, 0 }, col4 {  9, 90, 0, 0 },
        col4 { 13, 47, 0, 0 }, col4 { 42, 75, 0, 0 }, col4 { 70, 54, 0, 0 },
        col4 { 59, 25, 0, 0 }, col4 { 49, 54, 0, 0 }, col4 { 94, 53, 0, 0 },
        col4 { 69,  1, 0, 0 }, col4 { 19, 66, 0, 0 }, col4 { 97, 13, 0, 0 },

        col4 { 68, 98, 0, 0 }, col4 { 63, 41, 0, 0 }, col4 { 73, 60, 0, 0 },
        col4 { 42, 69, 0, 0 }, col4 { 69, 89, 0, 0 }, col4 { 92,  7, 0, 0 },
        col4 { 58, 40, 0, 0 }, col4 { 26, 73, 0, 0 }, col4 { 95,  6, 0, 0 },
        col4 { 24, 22, 0, 0 }, col4 { 43, 65, 0, 0 }, col4 {  4, 79, 0, 0 },
        col4 {100, 48, 0, 0 }, col4 { 50, 97, 0, 0 }, col4 { 32, 31, 0, 0 },

        col4 { 80, 63, 0, 0 }, col4 { 38, 87, 0, 0 }, col4 { 79, 14, 0, 0 },
        col4 { 38, 59, 0, 0 }, col4 { 39, 52, 0, 0 }, col4 { 21, 94, 0, 0 },
        col4 { 99, 42, 0, 0 }, col4 { 37, 56, 0, 0 }, col4 {  4, 66, 0, 0 },
        col4 {  5, 43, 0, 0 }, col4 {  7, 31, 0, 0 }, col4 {100, 45, 0, 0 },
        col4 { 71, 91, 0, 0 }, col4 { 58, 18, 0, 0 }, col4 {  1, 37, 0, 0 },

        col4 { 90,  4, 0, 0 }, col4 { 90, 43, 0, 0 }, col4 { 72,  3, 0, 0 },
        col4 { 47, 65, 0, 0 }, col4 { 71, 85, 0, 0 }, col4 { 85, 34, 0, 0 },
        col4 { 79, 88, 0, 0 }, col4 { 15, 13, 0, 0 }, col4 { 38, 61, 0, 0 },
        col4 { 91, 34, 0, 0 }, col4 {  2, 42, 0, 0 }, col4 { 72,  1, 0, 0 },
        col4 { 86, 92, 0, 0 }, col4 { 44, 58, 0, 0 }, col4 { 96, 38, 0, 0 },
    };

    test_mark_scanline(1950,
        vertices,
        [&](size_t i, col4 i1, col4 i2, col4 p1, col4 p2) {
            assert_gs_output(vertices, i, i1, i2, p1, p2);
        }
    );
}

def_test_case_with_fixture(mark_scanline_floating, mark_scanline_fixture) {
    buffer<col4> vertices {
        col4 { 73.130000, 50.770000, 0.740000, 9.890000 },
        col4 { 44.460000, 10.460000, 0.240000, 2.780000 },
        col4 { 99.650000, 44.950000, 0.190000, 7.990000 },

        col4 { 89.110000, 14.890000, 0.210000, 8.840000 },
        col4 { 46.470000, 78.180000, 0.530000, 8.480000 },
        col4 { 14.650000, 73.640000, 0.290000, 1.290000 },

        col4 { 58.690000, 73.790000, 0.310000, 2.770000 },
        col4 {  4.200000, 81.050000, 0.130000, 8.880000 },
        col4 { 96.240000, 59.680000, 0.540000, 1.360000 },

        col4 { 10.990000, 91.970000, 0.890000, 2.840000 },
        col4 { 88.880000, 67.170000, 0.170000, 0.480000 },
        col4 { 53.310000, 92.080000, 0.100000, 4.500000 },

        col4 { 92.330000, 29.320000, 0.770000, 0.060000 },
        col4 { 77.660000,  8.450000, 0.790000, 9.350000 },
        col4 { 27.840000, 73.720000, 0.030000, 0.800000 },

        col4 { 53.790000, 49.210000, 0.230000, 2.550000 },
        col4 {  8.900000, 38.440000, 0.150000, 8.060000 },
        col4 {  5.690000, 73.350000, 0.950000, 2.110000 },
    };

    test_mark_scanline(502,
        vertices,
        [&](size_t i, col4 i1, col4 i2, col4 p1, col4 p2) {
            assert_gs_output(vertices, i, i1, i2, p1, p2);
        }
    );
}

struct fill_scanline_fixture : context_fixture {
    program prg;

    fill_scanline_fixture() :
        prg(compile(ifstream("../kernels/interpolation.cl"))) { }
};

int main()
{
    test_case::test_all(true);
    cout << test_case::log().str();

    return test_case::status() ? 0 : -1;
}

