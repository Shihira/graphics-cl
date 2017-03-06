// cflags: -lOpenCL

#include <fstream>

#include "promise.h"
#include "common/unit_test.h"

using namespace std;
using namespace gcl;

struct kernel_fixture {
    ifstream fprg;
    program prg;
    pipeline pl;
    kernel krn;

    kernel_fixture(string krn_name):
            fprg("../kernels/rasterizer.cl"),
            prg(compile(fprg, "-cl-kernel-arg-info")),
            krn(prg, krn_name.c_str()) {
        pl.bind_kernel(krn_name, krn);
    }
};

struct mark_scanline_fixture : kernel_fixture {
    buffer<cl_uint> gclMarkSize;
    buffer<cl_uint> gclFragmentSize;
    buffer<col4>    gclMarkPos;
    buffer<col4>    gclMarkInfo;

    mark_scanline_fixture() :
            kernel_fixture("mark_scanline"),
            gclMarkSize { 0 },
            gclFragmentSize { 0 },
            gclMarkPos(10000, host_map),
            gclMarkInfo(10000, host_map) { }

    void render(
            buffer<col4>& InterpPosition,
            buffer<float>& gclViewport) {

        gclMarkSize[0] = 0;
        gclFragmentSize[0] = 0;

        pl.auto_bind_buffer(InterpPosition);
        pl.auto_bind_buffer(gclViewport);
        pl.auto_bind_buffer(gclMarkSize);
        pl.auto_bind_buffer(gclFragmentSize);
        pl.auto_bind_buffer(gclMarkPos );
        pl.auto_bind_buffer(gclMarkInfo);

        promise() <<
            push(InterpPosition) <<
            push(gclViewport) <<
            push(gclMarkSize) <<
            push(gclFragmentSize) <<
            run(krn, 1) <<
            pull(gclMarkSize) <<
            pull(gclFragmentSize) <<
            pull(gclMarkPos) <<
            pull(gclMarkInfo) <<
            wait_until_done;
    }

    void sort_scanlines() {
        size_t scanline_size = gclMarkSize[0] / 2;

        vector<size_t> idx;
        for(size_t i = 0; i < scanline_size; i++)
            idx.push_back(i);

        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            return gclMarkPos[a * 2][1] < gclMarkPos[b * 2][1];
        });

        vector<col4> temp_pos, temp_info;
        for(size_t i : idx) {
            temp_pos.push_back(gclMarkPos[i * 2]);
            temp_pos.push_back(gclMarkPos[i * 2 + 1]);
            temp_info.push_back(gclMarkInfo[i * 2]);
            temp_info.push_back(gclMarkInfo[i * 2 + 1]);
        }

        std::copy(temp_pos.begin(), temp_pos.end(), gclMarkPos.begin());
        std::copy(temp_info.begin(), temp_info.end(), gclMarkInfo.begin());
    }
};

TEST_CASE_FIXTURE(mark_scanline_small_triangle, mark_scanline_fixture) {
    buffer<col4> InterpPosition {
        col4 { -.2, 0.4, -.1, 1 },
        col4 { 0.1, -.6, 0.5, 1 },
        col4 { 0.8, -.9, 0.9, 1 },
    };
    buffer<float> gclViewport { 0, 0, 20, 20 };

    render(InterpPosition, gclViewport);

    sort_scanlines();

    for(size_t i = 0; i < gclMarkSize[0] / 2; i++) {
        assert_true(gclMarkPos[i * 2][1] == gclMarkPos[i * 2 + 1][1]);
        assert_true(gclMarkPos[i * 2][0] <= gclMarkPos[i * 2 + 1][0]);
        if(!i) continue;
        assert_true(gclMarkPos[i * 2][1] == gclMarkPos[i * 2 - 2][1] + 1);
    }
}

TEST_CASE_FIXTURE(mark_scanline_joint_triangle, mark_scanline_fixture) {
    vector<col4> buf;

    buffer<col4> InterpPosition_1 {
        col4 { 0.217, 0.4, -.1, 1 },
        col4 { -.145, -.6, 0.5, 1 },
        col4 { 0.828, -.9, 0.9, 1 },
    };
    buffer<col4> InterpPosition_2 {
        col4 { 0.217, 0.4, -.1, 1 },
        col4 { 0.645, 0.1, 0.5, 1 },
        col4 { 0.828, -.9, 0.9, 1 },
    };
    buffer<float> gclViewport { 0, 0, 4000, 4000 };

    render(InterpPosition_1, gclViewport);
    sort_scanlines();
    size_t size_1 = gclMarkSize[0];
    buf.insert(buf.end(), gclMarkPos.begin(), gclMarkPos.begin() + size_1);

    render(InterpPosition_2, gclViewport);
    sort_scanlines();
    size_t size_2 = gclMarkSize[0];
    buf.insert(buf.end(), gclMarkPos.begin(), gclMarkPos.begin() + size_2);

    assert_true(size_1 == size_2);
    for(size_t i = 0; i < size_1 / 2; i++) {
        assert_true(buf[i * 2 + 1][1] == buf[size_1 + i * 2][1]);
        assert_true(buf[i * 2 + 1][0] == buf[size_1 + i * 2][0]);
    }
}

TEST_CASE_FIXTURE(mark_scanline_big_triangle, mark_scanline_fixture) {
    buffer<col4> InterpPosition {
        col4 { -.2, 0.4, -.1, 1 },
        col4 { 0.1, -.6, 0.5, 1 },
        col4 { 0.8, -.9, 0.9, 1 },
    };
    buffer<float> gclViewport { 0, 0, 4000, 4000 };

    render(InterpPosition, gclViewport);

    sort_scanlines();

    for(size_t i = 0; i < gclMarkSize[0] / 2; i++) {
        assert_true(gclMarkPos[i * 2][1] == gclMarkPos[i * 2 + 1][1]);
        assert_true(gclMarkPos[i * 2][0] <= gclMarkPos[i * 2 + 1][0]);
        if(!i) continue;
        assert_true(gclMarkPos[i * 2][1] == gclMarkPos[i * 2 - 2][1] + 1);
    }
}

struct fill_scanline_fixture : kernel_fixture {
    buffer<cl_uint> gclFragmentSize;
    buffer<col4> gclFragPos;
    buffer<col4> gclFragInfo;

    fill_scanline_fixture() : kernel_fixture("fill_scanline"),
            gclFragmentSize({ 0 }),
            gclFragPos(40000, host_map),
            gclFragInfo(40000, host_map) { }

    void render(
            buffer<col4>& gclMarkPos,
            buffer<col4>& gclMarkInfo,
            buffer<float>& gclViewport) {
        gclFragmentSize[0] = 0;

        pl.auto_bind_buffer(gclMarkPos);
        pl.auto_bind_buffer(gclMarkInfo);
        pl.auto_bind_buffer(gclViewport);
        pl.auto_bind_buffer(gclFragmentSize);
        pl.auto_bind_buffer(gclFragPos);
        pl.auto_bind_buffer(gclFragInfo);

        promise() <<
            push(gclMarkPos) <<
            push(gclMarkInfo) <<
            push(gclViewport) <<
            push(gclFragmentSize) <<
            run(krn, 1) <<
            pull(gclFragmentSize) <<
            pull(gclFragPos) <<
            pull(gclFragInfo) <<
            wait_until_done;
    }
};

TEST_CASE_FIXTURE(fill_single_scanline, fill_scanline_fixture) {
    buffer<col4> gclMarkPos {
        col4 { 21.20, 2, 0, 1 },
        col4 { 52.18, 2, 0, 1 },
    };
    buffer<col4> gclMarkInfo {
        col4 { 1, 0, 0, 0 },
        col4 { 0, 1, 0, 0 },
    };
    buffer<float> gclViewport { 0, 0, 100, 100 };

    render(gclMarkPos, gclMarkInfo, gclViewport);

    size_t expected_size = gclMarkPos[1][0] - gclMarkPos[0][0];
    assert_true(gclFragmentSize[0] == expected_size);

    int prev_x = gclFragPos[0][0];
    for(size_t i = 1; i < expected_size; i++) {
        assert_true(int(gclFragPos[i][0]) - prev_x <= 1);
        prev_x = gclFragPos[i][0];
    }
}

TEST_CASE(double_floating_point_comparison) {
    std::vector<double> doubles {
        -19.054817824216737, -6.80421153560839, -9.278101722725665,
        -17.18510762126227, 1.4881675474870475, -7.998945239584955,
        -9.345788325262378, -18.138063333632047, -23.980307102623037,
        -4.143562513785255, -26.39036990754327, -0.13864392080461196,
        16.49739530923419, 10.507144889251357, 9.722516732719365,
        0.2292182137569041, 3.114431761965425, 3.4898924949012784,
        8.16099389827008, 22.384804688962845, -3.115087633774495,
        21.17281832632006, -0.9001637039756167, -6.15103889239769,
        2.5030725775315976, 14.48650582302901, -12.977402040776678,
        -20.912566904765747, -5.655231707644738, 6.816473810269541,
        -1.7090607933072257, 1.7515450346373869, -1.3169856930046397,
        1.481075096934081, 22.423943228529353, -7.437925942839773,
        -26.327104248747307, 8.284113434441808, 18.33020385023054,
    };

    std::vector<double> control_group = doubles;

    assert_true(sizeof(double) == 8);

    std::sort(doubles.begin(), doubles.end(), [](double a, double b) {
        int64_t ia = *reinterpret_cast<int64_t*>(&a);
        int64_t ib = *reinterpret_cast<int64_t*>(&b);

        if(ia < 0) ia ^= ~(1UL << 63);
        if(ib < 0) ib ^= ~(1UL << 63);

        return ia < ib;
    });
    std::sort(control_group.begin(), control_group.end());

    assert_true(doubles == control_group);
}

struct depth_test_fixture : kernel_fixture {
    buffer<float> gclDepthBuffer;
    buffer<cl_uint> gclBufferSize;

    depth_test_fixture() : kernel_fixture("depth_test"),
            gclDepthBuffer(40000, host_map),
            gclBufferSize({200, 200}) { }

    void render(buffer<col4>& gclFragPos) {
        pl.auto_bind_buffer(gclDepthBuffer);
        pl.auto_bind_buffer(gclBufferSize);
        pl.auto_bind_buffer(gclFragPos);

        promise() <<
            fill(gclDepthBuffer, 1.f) <<
            push(gclBufferSize) <<
            push(gclFragPos) <<
            run(krn, gclFragPos.size()) <<
            pull(gclDepthBuffer) <<
            wait_until_done;
    }
};

TEST_CASE_FIXTURE(depth_test, depth_test_fixture) {
    buffer<col4> gclFragPos {
        col4 { 2, 3, 0.6, 1 },
        col4 { 50, 50, 0.2, 1 },
        col4 { 2, 3, 0.1, 1 },
        col4 { 6, 7, 0.5, 1 },
        col4 { 50, 50, 0.01, 1 }
    };

    render(gclFragPos);

    assert_true(gclDepthBuffer[3 * 200 + 2] == gclFragPos[2][2]);
    assert_true(gclDepthBuffer[7 * 200 + 6] == gclFragPos[3][2]);
    assert_true(gclDepthBuffer[50 * 200 + 50] == gclFragPos[4][2]);
}

int main(int argc, char** argv)
{
    std::vector<platform> ps = platform::get();
    std::vector<device> ds = device::get(ps);

    context ctxt(ds.back());
    context_guard cg(ctxt);

    return shrtool::unit_test::test_main(argc, argv);
}

