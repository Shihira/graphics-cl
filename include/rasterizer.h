#ifndef RASTERIZER_H_INCLUDED
#define RASTERIZER_H_INCLUDED

#include <limits>

#include "promise.h"

namespace gcl {

struct rasterizer_pipeline : pipeline {

    buffer<float>       gclViewport;
    buffer<cl_uint>     gclMarkSize;
    buffer<cl_uint>     gclFragmentSize;
    buffer<col4>        gclMarkPos;
    buffer<col4>        gclMarkInfo;
    buffer<col4>        gclFragPos;
    buffer<col4>        gclFragInfo;
    buffer<cl_uint>     gclBufferSize;
    buffer<cl_int>      gclDepthBuffer;
    buffer<cl_float4>   gclColorBuffer;

    rasterizer_pipeline() :
        gclMarkSize         ({ 0 }),
        gclFragmentSize     ({ 0 }),
        gclMarkPos          (1000, host_map),
        gclMarkInfo         (1000, host_map),
        gclFragPos          (1000, host_map),
        gclFragInfo         (1000, host_map)
    {
        auto_bind_buffer(gclMarkSize        );
        auto_bind_buffer(gclFragmentSize    );
        auto_bind_buffer(gclMarkPos         );
        auto_bind_buffer(gclMarkInfo        );
        auto_bind_buffer(gclFragPos         );
        auto_bind_buffer(gclFragInfo        );
    }

    rasterizer_pipeline(size_t w, size_t h) :
        rasterizer_pipeline()
    {
        set_size(w, h);
    }

    void set_rasterizer_program(const program& p) {
        bind_kernel_from_program(p);

        krn_mark_scanline_ = get_kernel("mark_scanline");
        krn_fill_scanline_ = get_kernel("fill_scanline");
        krn_depth_test_    = get_kernel("depth_test");
    }

    void set_vertex_shader_program(const program& p,
            std::string krn_name = "vertex_shader") {
        bind_kernel_from_program(p);

        krn_vertex_shader_ = get_kernel(krn_name);
    }

    void set_fragment_shader_program(const program& p,
            std::string krn_name = "fragment_shader") {
        bind_kernel_from_program(p);

        krn_fragment_shader_ = get_kernel(krn_name);
    }

    void set_size(size_t w, size_t h) {
        gclViewport    = buffer<float>     ({ 0, 0, float(w), float(h) });
        gclBufferSize  = buffer<cl_uint>   ({ cl_uint(w), cl_uint(h) });
        gclDepthBuffer = buffer<cl_int>    (w * h, host_map);
        gclColorBuffer = buffer<cl_float4> (w * h, host_map);

        auto_bind_buffer(gclViewport        );
        auto_bind_buffer(gclBufferSize      );
        auto_bind_buffer(gclDepthBuffer     );
        auto_bind_buffer(gclColorBuffer     );
    }

    void set_vertex_number(size_t n) {
        vertex_number = n;
    }

public:
    /*
     * stage functions
     */
    promise clear_depth_buffer_stage() {
        return async_promise <<
            fill(gclDepthBuffer, std::numeric_limits<int>::max());
    }

    promise clear_color_buffer_stage() {
        return async_promise <<
            fill(gclColorBuffer, cl_float4 { 255, 255, 255, 255 });
    }

    promise vertex_shading_stage() {
        return async_promise <<
            run_kernel(*krn_vertex_shader_, vertex_number);
    }

    promise estimate_mark_size_stage() {
        gclMarkSize[0] = 0;
        gclFragmentSize[0] = 0;

        krn_mark_scanline_->set_null(
            krn_mark_scanline_->get_index("gclMarkInfo"));

        return async_promise <<
            push(gclViewport) <<
            push(gclMarkSize) <<
            push(gclFragmentSize) <<
            run(*krn_mark_scanline_, vertex_number / 3) <<
            pull(gclMarkSize);
    }

    void check_mark_size_stage() {
        if(gclMarkSize[0] > gclMarkPos.size() ||
                gclMarkSize[0] > gclMarkInfo.size()) {
            size_t new_mark_size = 1 <<
                    size_t(std::log2(gclMarkSize[0]) + 1);
            gclMarkPos = buffer<col4>(new_mark_size, host_map);
            gclMarkInfo = buffer<col4>(new_mark_size, host_map);
        }

        auto_bind_buffer(gclMarkPos);
        auto_bind_buffer(gclMarkInfo);
    }

    promise mark_scanline_stage() {
        gclMarkSize[0] = 0;
        gclFragmentSize[0] = 0;

        return async_promise <<
            push(gclMarkSize) <<
            push(gclFragmentSize) <<
            run(*krn_mark_scanline_, vertex_number / 3) <<
            pull(gclMarkSize) <<
            pull(gclFragmentSize);
    }

    void check_fragment_size_stage() {
        krn_fill_scanline_->range(gclMarkSize[0] / 2);

        if(gclFragmentSize[0] > gclFragPos.size() ||
                gclFragmentSize[0] > gclFragInfo.size()) {
            size_t new_frag_size = 1 <<
                    size_t(std::log2(gclFragmentSize[0]) + 1);
            gclFragPos = buffer<col4>(new_frag_size, host_map);
            gclFragInfo = buffer<col4>(new_frag_size, host_map);

            auto_bind_buffer(gclFragPos);
            auto_bind_buffer(gclFragInfo);
        }

        gclFragmentSize[0] = 0;
    }

    promise fill_scanline_stage() {
        return async_promise <<
            push(gclFragmentSize) <<
            run(*krn_fill_scanline_) <<
            pull(gclFragmentSize);
    }

    promise depth_test_stage() {
        return async_promise <<
            push(gclBufferSize) <<
            run(*krn_depth_test_, gclFragmentSize[0]);
    }

    promise fragment_shading_stage() {
        return async_promise <<
            run(*krn_fragment_shader_, gclFragmentSize[0]) <<
            pull(gclColorBuffer);
    }

#define CALLC_(memfn) callc(std::bind(&rasterizer_pipeline::memfn, this))
#define CALLP_(memfn) call(std::bind(&rasterizer_pipeline::memfn, this))

    void render() {
        main_promise <<
            CALLC_(clear_depth_buffer_stage) <<
            CALLC_(clear_color_buffer_stage) <<
            CALLC_(vertex_shading_stage) <<
            CALLC_(estimate_mark_size_stage) <<
            CALLP_(check_mark_size_stage) <<
            CALLC_(mark_scanline_stage) <<
            CALLP_(check_fragment_size_stage) <<
            CALLC_(fill_scanline_stage) <<
            CALLC_(depth_test_stage) <<
            CALLC_(fragment_shading_stage) <<
            wait_until_done;
    }

#undef CALLC_
#undef CALLP_

protected:
    kernel* krn_vertex_shader_ = nullptr;
    kernel* krn_fragment_shader_ = nullptr;
    kernel* krn_mark_scanline_ = nullptr;
    kernel* krn_fill_scanline_ = nullptr;
    kernel* krn_depth_test_ = nullptr;

    size_t vertex_number = 3;

    promise main_promise;
    promise async_promise;
};

}


#endif // RASTERIZER_H_INCLUDED
