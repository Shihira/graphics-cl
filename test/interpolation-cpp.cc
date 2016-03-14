// cflags: -lOpenCL
#define __CL_ENABLE_EXCEPTIONS

#include <vector>
#include <array>
#include <string>
#include <iterator>
#include <iostream>
#include <fstream>

#include <CL/cl.hpp>

using namespace std;

int main()
{
    vector<float> triangles {
        1, 5, 0, 1,
        9, 2, 0, 1,
        4, 1, 0, 1,
    };

    size_t scanline_size = 0;
    size_t fragment_size = 0;

    cl::Buffer buf_triangles(triangles.begin(), triangles.end(), false);
    cl::Buffer buf_sl_size(&scanline_size, &scanline_size + 1, false);
    cl::Buffer buf_fm_size(&fragment_size, &fragment_size + 1, false);

    ifstream src_file("../kernels/interpolation.cl");
    cl::Program prg(string(
            istreambuf_iterator<char>(src_file),
            istreambuf_iterator<char>()), false);

    try {
        prg.build();
    } catch(cl::Error e) {
        if(e.err() == CL_BUILD_PROGRAM_FAILURE) {
            cout << prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    cl::Device::getDefault()) << endl;
        }
        throw;
    }

    cl::Kernel gen_kernel(prg, "gen_scanline");
    gen_kernel.setArg(0, buf_triangles);
    gen_kernel.setArg(2, buf_sl_size);
    cl::Kernel fill_kernel(prg, "fill_scanline");
    fill_kernel.setArg(0, buf_triangles);
    fill_kernel.setArg(3, buf_fm_size);

    cl::CommandQueue cmdq = cl::CommandQueue::getDefault();
    gen_kernel.setArg(1, cl::Memory(nullptr));
    cmdq.enqueueNDRangeKernel(gen_kernel,
            cl::NullRange,
            cl::NDRange(triangles.size() / 4 / 3),
            cl::NullRange);
    cmdq.enqueueReadBuffer(buf_sl_size, true, 0,
            sizeof(size_t), &scanline_size);
    cl::Buffer buf_sl(CL_MEM_READ_WRITE, scanline_size * 4 * sizeof(float));
    gen_kernel.setArg(1, buf_sl);
    size_t real_scanline_size = 0;
    cmdq.enqueueWriteBuffer(buf_sl_size, true, 0,
            sizeof(size_t), &real_scanline_size);
    cmdq.enqueueNDRangeKernel(gen_kernel,
            cl::NullRange,
            cl::NDRange(triangles.size() / 4 / 3),
            cl::NullRange);
    vector<float> sl_inf(scanline_size * 4);
    cmdq.enqueueReadBuffer(buf_sl, true, 0,
            sizeof(float) * sl_inf.size(), sl_inf.data());
    cmdq.enqueueReadBuffer(buf_sl_size, true, 0,
            sizeof(size_t), &real_scanline_size);
    cout << real_scanline_size << endl;

    for(size_t i = 0; i < scanline_size * 4; i++) {
        cout << sl_inf[i] << ' ';
        if(i % 4 == 3) cout << endl;
    }

    fill_kernel.setArg(1, buf_sl);
    fill_kernel.setArg(2, cl::Memory(nullptr));
    cmdq.enqueueNDRangeKernel(fill_kernel,
            cl::NullRange,
            cl::NDRange(scanline_size / 2),
            cl::NullRange);
    cmdq.enqueueReadBuffer(buf_fm_size, true, 0,
            sizeof(size_t), &fragment_size);
    cout << fragment_size << endl;
    cl::Buffer buf_fm(CL_MEM_READ_WRITE, fragment_size * 4 * sizeof(float));
    fill_kernel.setArg(2, buf_fm);

    size_t real_fragment_size = 0;
    cmdq.enqueueWriteBuffer(buf_fm_size, true, 0,
            sizeof(size_t), &real_fragment_size);
    cmdq.enqueueNDRangeKernel(fill_kernel,
            cl::NullRange,
            cl::NDRange(scanline_size / 2),
            cl::NullRange);
    cmdq.enqueueReadBuffer(buf_fm_size, true, 0,
            sizeof(size_t), &real_fragment_size);
    vector<float> frag_inf(fragment_size * 4);
    cmdq.enqueueReadBuffer(buf_fm, true, 0,
            sizeof(float) * frag_inf.size(), frag_inf.data());

    for(size_t i = 0; i < fragment_size * 4; i++) {
        cout << frag_inf[i] << ' ';
        if(i % 4 == 3) cout << endl;
    }
}

