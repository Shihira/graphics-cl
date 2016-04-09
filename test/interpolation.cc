// cflags: -lOpenCL
#define __CL_ENABLE_EXCEPTIONS

#include <ctime>
#include <fstream>
#include <memory>

#include "../include/comput.h"

using namespace std;
using namespace gcl;

int main()
{
    std::vector<platform> ps = platform::get();
    std::vector<device> ds = device::get(ps);
    context ctxt(ds.back());
    context_guard cg(ctxt);

    buffer<col4, cl_float4> buf_triangles {
        col4 { 1, 5, 0, 1 },
        col4 { 90, 20, 0, 1 },
        col4 { 40, 80, 0, 1 },

        col4 { 90, 20, 0, 1 },
        col4 { 40, 80, 0, 1 },
        col4 { 192, 222, 0, 1 },

        col4 { 192, 222, 0, 1 },
        col4 { 242, 272, 0, 1 },
        col4 { 399, 199, 0, 1 },
    };
    buffer<col4, cl_float4> buf_colors(buf_triangles.size(), host_map);

    for(size_t i = 0; i < buf_triangles.size(); i += 3) {
        buf_colors[i + 0] = col4 { 255, 0, 0, 1 };
        buf_colors[i + 1] = col4 { 0, 255, 0, 1 };
        buf_colors[i + 2] = col4 { 0, 0, 255, 1 };
    }
    buffer<size_t> buf_size { 0 };

    size_t w = 1024, h = 768;
    size_t num_vertices = buf_triangles.size(),
           num_triangles = num_vertices / 3,
           num_scanline_endpoints = 0,
           num_scanlines = 0,
           num_fragments = 0;

    buffer<col4, cl_float4>
        buf_out_pos(num_vertices),
        buf_out_iro(num_vertices),
        buf_color_buffer(w * h, host_map);

    program inter_prg = compile(ifstream("../kernels/interpolation.cl"));
    program vert_prg = compile(ifstream("../kernels/vertex_shader.cl"));
    program frag_prg = compile(ifstream("../kernels/fragment_shader.cl"));

    kernel vert_kernel(vert_prg, "vs_main");
    kernel gen_kernel(inter_prg, "gen_scanline");
    kernel fill_kernel(inter_prg, "fill_scanline");
    kernel frag_kernel(frag_prg, "fs_main");
    kernel img_kernel(frag_prg, "generate_image");
    kernel clear_kernel(frag_prg, "clear_buffer");

    promise cp;

    for(int x = 0; x < 10; x++) {

    clock_t t = clock();

    vert_kernel.setArg(0, buf_triangles.buf());
    vert_kernel.setArg(1, buf_colors.buf());
    vert_kernel.setArg(2, buf_out_pos.buf());
    vert_kernel.setArg(3, buf_out_iro.buf());

    gen_kernel.setArg(0, buf_out_pos.buf());
    gen_kernel.setArg(1, nullptr_buf);
    gen_kernel.setArg(2, nullptr_buf);
    gen_kernel.setArg(3, buf_size.buf());

    clear_kernel.setArg(0, buf_color_buffer.buf());

    promise clear_promise = cp << run(clear_kernel, w * h);

    buf_size[0] = 0;

    promise {
        cp << push(buf_triangles) << unpush(buf_triangles),
        cp << push(buf_colors) << unpush(buf_colors),
    } << wait_until_done;

    cp <<
        push(buf_size) << unpush(buf_size) <<
        run(vert_kernel, num_vertices) <<
        run(gen_kernel, num_triangles) <<
        pull(buf_size) << unpull(buf_size) <<
        wait_until_done;

    num_scanline_endpoints = buf_size[0];
    num_scanlines = num_scanline_endpoints / 2;
    buf_size[0] = 0;

    buffer<cl_float4> buf_scan_inf(num_scanline_endpoints);
    buffer<cl_float4> buf_scan_pos(num_scanline_endpoints);

    gen_kernel.setArg(1, buf_scan_inf.buf());
    gen_kernel.setArg(2, buf_scan_pos.buf());

    fill_kernel.setArg(0, buf_scan_pos.buf());
    fill_kernel.setArg(1, buf_scan_inf.buf());
    fill_kernel.setArg(2, nullptr_buf);
    fill_kernel.setArg(3, nullptr_buf);
    fill_kernel.setArg(4, buf_size.buf());

    cp <<
        push(buf_size) << unpush(buf_size) <<
        run(gen_kernel, num_triangles) <<
        push(buf_size) << unpush(buf_size) <<
        run(fill_kernel, num_scanlines) <<
        pull(buf_size) << unpull(buf_size) <<
        wait_until_done;

    num_fragments = buf_size[0];
    buf_size[0] = 0;

    buffer<cl_float4> buf_frag_inf(num_fragments);
    buffer<cl_float4> buf_frag_pos(num_fragments);
    buffer<cl_float4> buf_frag_col(num_fragments);

    fill_kernel.setArg(2, buf_frag_pos.buf());
    fill_kernel.setArg(3, buf_frag_inf.buf());

    frag_kernel.setArg(0, buf_frag_inf.buf());
    frag_kernel.setArg(1, buf_frag_pos.buf());
    frag_kernel.setArg(2, buf_out_iro.buf());
    frag_kernel.setArg(3, buf_frag_col.buf());

    img_kernel.setArg(0, buf_frag_inf.buf());
    img_kernel.setArg(1, buf_frag_pos.buf());
    img_kernel.setArg(2, buf_frag_col.buf());
    img_kernel.setArg(3, buf_color_buffer.buf());

    cp <<
        push(buf_size) << unpush(buf_size) <<
        run(fill_kernel, num_scanlines) <<
        run(frag_kernel, num_fragments) <<
        wait_until_done;

    clear_promise <<
        run(img_kernel, num_fragments) <<
        wait_until_done;

    cerr << clock() - t << endl; t = clock();

    }

    cp <<
        pull(buf_color_buffer) << unpull(buf_color_buffer) <<
        wait_until_done;

    cout << "P6\n" << w << ' ' << h << "\n255\n";
    for(size_t i = 0; i < buf_color_buffer.size(); i++) {
        cout << uint8_t(buf_color_buffer[i][0]);
        cout << uint8_t(buf_color_buffer[i][1]);
        cout << uint8_t(buf_color_buffer[i][2]);
    }
}

