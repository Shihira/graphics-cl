/*
 * Copyright(C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

/*
 * (DEPRECATED)
 * Use buffer to reduce times of synchronization of atomic operations
 *
#define BUFFER_SYNC_TRADEOFF_OPTIMIZATION
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define swap_(t, a, b) { t temp = a; a = b; b = temp; }

typedef float4 pos_t;
typedef float4 inf_t;

/*
 * Output indices ensure triangles is sorted in ascending order of y,x.
 */

void sort_triangle(
        global const pos_t triangle[3],
        size_t idx[3])
{
    idx[0] = 0; idx[1] = 1; idx[2] = 2;

    if(triangle[idx[0]].y > triangle[idx[1]].y)
        swap_(size_t, idx[0], idx[1]);
    if(triangle[idx[0]].y > triangle[idx[2]].y)
        swap_(size_t, idx[0], idx[2]);
    if(triangle[idx[1]].y > triangle[idx[2]].y)
        swap_(size_t, idx[1], idx[2]);
}

float4 identity_dim(size_t d)
{
    switch(d) {
    case 0:
        return (float4)(1, 0, 0, 0);
    case 1:
        return (float4)(0, 1, 0, 0);
    case 2:
        return (float4)(0, 0, 1, 0);
    case 3:
        return (float4)(0, 0, 0, 1);
    default:
        return (float4)(0);
    }
}

/*
 * This algorithm split edge of a triangle to make the triangle a quad polygon.
 */

void extract_quad(
        global const pos_t triangle[3],
        size_t idx[3],

        inf_t quad_inf[4],
        inf_t quad_pos[4])
{
    size_t item_id = get_global_id(0);

    for(int i = 0; i < 4; i++)
        quad_inf[i] = (float4)(0, 0, 0, item_id);

    inf_t comp_min = identity_dim(idx[0]),
          comp_mid = identity_dim(idx[1]),
          comp_max = identity_dim(idx[2]);

    quad_inf[0] += comp_min;
    quad_inf[3] += comp_max;
    quad_pos[0] = triangle[idx[0]];
    quad_pos[3] = triangle[idx[2]];

    float ratio =
        (triangle[idx[1]].y - triangle[idx[0]].y) /
        (triangle[idx[2]].y - triangle[idx[0]].y);

    pos_t breakpoint = ratio * quad_pos[3] + (1-ratio) * quad_pos[0];

    if(breakpoint.x > triangle[idx[1]].x) {
        quad_inf[1] += comp_mid;
        quad_inf[2] += ratio * comp_max + (1-ratio) * comp_min;
        quad_pos[1] = triangle[idx[1]];
        quad_pos[2] = breakpoint;
    } else {
        quad_inf[1] += ratio * comp_max + (1-ratio) * comp_min;
        quad_inf[2] += comp_mid;
        quad_pos[1] = breakpoint;
        quad_pos[2] = triangle[idx[1]];
    }
}

#define interpolate_segment(t, len, nsync, begs, ends, f) { \
    t init[nsync]; \
    t diff[nsync]; \
    for(size_t i = 0; i < (nsync); ++i) { \
        init[i] = (begs)[i]; \
        diff[i] = ((ends)[i] - (begs)[i]) / (float)(len); \
    } \
    for(float l = 0; l < (len); l += 1.f) { \
        f((init), (nsync)); \
        for(size_t i = 0; i < (nsync); ++i) \
            init[i] += diff[i]; \
    } \
}

kernel void mark_scanline(
        global pos_t const* vertices,

        global inf_t* scan_inf,
        global pos_t* scan_pos,
        global uint* output_size)
{
    size_t item_id = get_global_id(0);

    global pos_t const* triangle = vertices + item_id * 3;

    size_t idx[3];
    sort_triangle(triangle, idx);

    size_t y_1 = floor(triangle[idx[1]].y) - floor(triangle[idx[0]].y);
    size_t y_2 = floor(triangle[idx[2]].y) - floor(triangle[idx[1]].y);

    inf_t quad_inf[4]; 
    pos_t quad_pos[4];
    extract_quad(triangle, idx, quad_inf, quad_pos);

    if(scan_inf == NULL || scan_pos == NULL) {
        atomic_add(output_size, (y_1 + y_2) * 2);
        return;
    }

    float4 beg_1[4] = { quad_inf[0], quad_inf[0], quad_pos[0], quad_pos[0], },
           end_1[4] = { quad_inf[1], quad_inf[2], quad_pos[1], quad_pos[2], },
           beg_2[4] = { quad_inf[1], quad_inf[2], quad_pos[1], quad_pos[2], },
           end_2[4] = { quad_inf[3], quad_inf[3], quad_pos[3], quad_pos[3], };

#ifdef BUFFER_SYNC_TRADEOFF_OPTIMIZATION

#define scanline_move_to_global { \
    size_t old = atomic_add(output_size, buf_used / 2); \
    for(size_t buf_i = 0, scan_i = old; \
            buf_i < buf_used; buf_i += 4, scan_i += 2) { \
        scan_inf[scan_i + 0] = priv_buf[buf_i + 0]; \
        scan_inf[scan_i + 1] = priv_buf[buf_i + 1]; \
        scan_pos[scan_i + 0] = round(priv_buf[buf_i + 2]); \
        scan_pos[scan_i + 0].z = priv_buf[buf_i + 2].z; \
        scan_pos[scan_i + 1] = round(priv_buf[buf_i + 3]); \
        scan_pos[scan_i + 1].z = priv_buf[buf_i + 2].z; \
    } \
    buf_used = 0; \
}

#define scanline_interpolate_func(data, sz) { \
    if(buf_used > 60) \
        scanline_move_to_global; \
    for(size_t i = 0; i < 4; i++, buf_used++) \
        priv_buf[buf_used] = data[i]; \
}

    size_t buf_used = 0;
    float4 priv_buf[64];

#else

#define scanline_interpolate_func(data, sz) { \
    size_t old = atomic_add(output_size, 2); \
    scan_inf[old + 0] = data[0]; \
    scan_inf[old + 1] = data[1]; \
    scan_pos[old + 0] = round(data[2]); \
    scan_pos[old + 0].z = data[2].z; \
    scan_pos[old + 1] = round(data[3]); \
    scan_pos[old + 1].z = data[3].z; \
}

#endif

    interpolate_segment(float4, y_1, 4, beg_1, end_1,
            scanline_interpolate_func);
    interpolate_segment(float4, y_2, 4, beg_2, end_2,
            scanline_interpolate_func);

#ifdef BUFFER_SYNC_TRADEOFF_OPTIMIZATION
    if(buf_used > 0)
        scanline_move_to_global;
#endif
}

kernel void fill_scanline(
        global pos_t const* scan_pos,
        global inf_t const* scan_inf,

        global pos_t* frag_pos,
        global inf_t* frag_inf,
        global uint* output_size)
{
    size_t item_id = get_global_id(0),
           scan_id = item_id * 2;

    scan_inf += scan_id;
    scan_pos += scan_id;

    int len = (int)(scan_pos[1].x) - (int)(scan_pos[0].x);

    if(frag_inf == NULL) {
        atomic_add(output_size, len);
        return;
    }

    float4 beg[2] = { scan_pos[0], scan_inf[0] };
    float4 end[2] = { scan_pos[1], scan_inf[1] };

#ifdef BUFFER_SYNC_TRADEOFF_OPTIMIZATION

#define fragment_move_to_global { \
    size_t old = atomic_add(output_size, buf_used / 2); \
    for(size_t buf_i = 0; buf_i < buf_used; buf_i += 2, old++) { \
        frag_pos[old] = priv_buf[buf_i]; \
        frag_inf[old] = priv_buf[buf_i + 1]; \
    } \
    buf_used = 0; \
}

#define fragment_interpolate_func(data, sz) { \
    if(buf_used > 60) \
        fragment_move_to_global; \
    priv_buf[buf_used++] = data[0]; \
    priv_buf[buf_used++] = data[1]; \
}

    size_t buf_used = 0;
    float4 priv_buf[64];

#else

#define fragment_interpolate_func(data, sz) { \
    size_t old = atomic_inc(output_size); \
    frag_pos[old] = data[0]; \
    frag_inf[old] = data[1]; \
}

#endif

    interpolate_segment(float4, len, 2, beg, end,
            fragment_interpolate_func);

#ifdef BUFFER_SYNC_TRADEOFF_OPTIMIZATION
    if(buf_used > 0)
        fragment_move_to_global;
#endif
}

