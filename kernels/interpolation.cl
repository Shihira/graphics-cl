/*
 * Copyright(C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

#define swap_(t, a, b) { t temp = a; a = b; b = temp; }

#define interpolate_segment(t, len, begs, ends, nsync, out, outsz) { \
    t init[nsync]; \
    t diff[nsync]; \
    for(size_t i = 0; i < (nsync); ++i) { \
        init[i] = (begs)[i]; \
        diff[i] = ((ends)[i] - (begs)[i]) / (float)(len); \
    } \
    for(float l = 0; l < (len); ++l) { \
        size_t old = atomic_add(outsz, nsync); \
        for(size_t i = 0; i < (nsync); ++i) { \
            out[old + i] = init[i]; \
            init[i] += diff[i]; \
        } \
    } \
}

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
    case 0: return (float4)(1, 0, 0, 0);
    case 1: return (float4)(0, 1, 0, 0);
    case 2: return (float4)(0, 0, 1, 0);
    case 3: return (float4)(0, 0, 0, 1);
    default: return (float4)(0);
    }
}

/*
 * This algorithm split edge of a triangle to make the triangle a quad polygon.
 */

void extract_quad(
        global const pos_t triangle[3],
        size_t idx[3],

        inf_t quad_inf[4])
{
    size_t item_id = get_global_id(0);

    for(int i = 0; i < 4; i++)
        quad_inf[i] = (float4)(0, 0, 0, item_id);

    inf_t comp_min = identity_dim(idx[0]),
          comp_mid = identity_dim(idx[1]),
          comp_max = identity_dim(idx[2]);

    quad_inf[0] += comp_min;
    quad_inf[3] += comp_max;

    float breakpoint =
        (triangle[idx[1]].y - triangle[idx[0]].y) /
        (triangle[idx[2]].y - triangle[idx[0]].y);

    if(triangle[idx[2]].x > triangle[idx[1]].x) {
        quad_inf[1] += comp_mid;
        quad_inf[2] += breakpoint * comp_max + (1 - breakpoint) * comp_min;
    } else {
        quad_inf[2] += comp_mid;
        quad_inf[1] += breakpoint * comp_max + (1 - breakpoint) * comp_min;
    }
}

kernel void gen_scanline(
        global pos_t const* vertices,

        global inf_t* scan_inf,
        global size_t* output_size)
{
    size_t item_id = get_global_id(0);
    global pos_t const* triangle = vertices + item_id * 3;

    size_t vert_idx[3];
    sort_triangle(triangle, vert_idx);
    float y_1 = triangle[vert_idx[1]].y - triangle[vert_idx[0]].y;
    float y_2 = triangle[vert_idx[2]].y - triangle[vert_idx[1]].y;

    if(scan_inf == NULL) {
        atomic_add(output_size, 2 * (y_1 + y_2));
        return;
    }

    inf_t quad_info[4]; 
    extract_quad(triangle, vert_idx, quad_info);
    inf_t trap_beg_1[2] = { quad_info[0], quad_info[0] },
          trap_beg_2[2] = { quad_info[1], quad_info[2] },
          trap_end_1[2] = { quad_info[1], quad_info[2] },
          trap_end_2[2] = { quad_info[3], quad_info[3] };

    interpolate_segment(inf_t, y_1, trap_beg_1, trap_end_1, 2,
            scan_inf, output_size);
    interpolate_segment(inf_t, y_2, trap_beg_2, trap_end_2, 2,
            scan_inf, output_size);
}

kernel void fill_scanline(
        global pos_t const* vertices,
        global inf_t const* scanlines,

        global inf_t* frag_inf,
        global size_t* output_size)
{
    size_t item_id = get_global_id(0);
    global inf_t const* scanline = scanlines + item_id * 2;
    global pos_t const* triangle = vertices + (size_t)scanline->w * 3;

    pos_t vecl = triangle[0] * scanline[0].x +
                 triangle[1] * scanline[0].y +
                 triangle[2] * scanline[0].z;
    pos_t vecr = triangle[0] * scanline[1].x +
                 triangle[1] * scanline[1].y +
                 triangle[2] * scanline[1].z;

    if(frag_inf == NULL) {
        atomic_add(output_size, vecr.x - vecl.x);
        return;
    }

    interpolate_segment(inf_t, vecr.x - vecl.x, scanline, scanline + 1, 1,
            frag_inf, output_size);
}

