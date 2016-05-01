/*
 * Copyright(C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define swap_(t, a, b) { t temp = a; a = b; b = temp; }

#define in          __global const
#define out         __global 
#define inout       __global

#define VP_LEFT     0
#define VP_TOP      1
#define VP_WIDTH    2
#define VP_HEIGHT   3

#define BS_WIDTH    0
#define BS_HEIGHT   1

typedef float4 pos_t;
typedef float4 inf_t;

/*
 * Output indices ensure triangles is sorted in ascending order of y,x.
 */

void sort_triangle(
        const pos_t triangle[3],
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
        const pos_t triangle[3],
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

/*
 * Procedure to interpolate nsync values synchronously. begs and ends have nsync
 * elements of type t respectively, among them [begs_i, ends_i) len elements
 * will be generated, for each of which f is called with nsync currently
 * generated elements as the first argument, and nsync as the second. f is free
 * to be a function or a macro (mostly a macro).
 */

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

int is_in_viewport(
        pos_t* point,
        in float* gclViewport)
{
    if(point->x < gclViewport[VP_LEFT] ||
       point->x > gclViewport[VP_LEFT] + gclViewport[VP_WIDTH]) return 0;
    if(point->y < gclViewport[VP_TOP] ||
       point->y > gclViewport[VP_TOP] + gclViewport[VP_HEIGHT]) return 0;
    if(point->z < 0 ||
       point->z > 1) return 0;

    return 1;
}

kernel void mark_scanline(
        in      pos_t*  InterpPosition,
        in      float*  gclViewport,
        inout   uint*   gclMarkSize,
        inout   uint*   gclFragmentSize,
        out     pos_t*  gclMarkPos,
        out     inf_t*  gclMarkInfo)
{
    size_t item_id = get_global_id(0) * 3;

    pos_t triangle[3] = {
        InterpPosition[item_id],
        InterpPosition[item_id + 1],
        InterpPosition[item_id + 2],
    };

    for(int i = 0; i < 3; i++) {
        triangle[i] /= triangle[i].w;
        triangle[i].x *= gclViewport[VP_WIDTH] / 2;
        triangle[i].x += VP_LEFT + gclViewport[VP_WIDTH] / 2;
        triangle[i].y *= gclViewport[VP_HEIGHT] / 2;
        triangle[i].y += VP_TOP + gclViewport[VP_HEIGHT] / 2;
        triangle[i].z *= 0.5;
        triangle[i].z += 0.5;
    }

    size_t idx[3];
    sort_triangle(triangle, idx);

    inf_t quad_inf[4]; 
    pos_t quad_pos[4];
    extract_quad(triangle, idx, quad_inf, quad_pos);

    for(int i = 0; i < 4; i++)
        quad_pos[i].y = floor(quad_pos[i].y);

    size_t y_1 = quad_pos[1].y - quad_pos[0].y;
    size_t y_2 = quad_pos[3].y - quad_pos[2].y;

    if(gclMarkInfo == NULL || gclMarkPos == NULL || gclFragmentSize == NULL) {
        atomic_add(gclMarkSize, (y_1 + y_2) * 2);
        return;
    }

    float4 beg_1[4] = { quad_inf[0], quad_inf[0], quad_pos[0], quad_pos[0], },
           end_1[4] = { quad_inf[1], quad_inf[2], quad_pos[1], quad_pos[2], },
           beg_2[4] = { quad_inf[1], quad_inf[2], quad_pos[1], quad_pos[2], },
           end_2[4] = { quad_inf[3], quad_inf[3], quad_pos[3], quad_pos[3], };

#define scanline_interpolate_func(data, sz) { \
    /*if(is_in_viewport(&data[2], gclViewport) || \
       is_in_viewport(&data[3], gclViewport)) {*/ \
        size_t old = atomic_add(gclMarkSize, 2); \
        gclMarkInfo[old + 0] = data[0]; \
        gclMarkInfo[old + 1] = data[1]; \
        gclMarkPos[old + 0] = round(data[2]); \
        gclMarkPos[old + 1] = round(data[3]); \
        gclMarkPos[old + 0].z = data[2].z; \
        gclMarkPos[old + 1].z = data[3].z; \
        atomic_add(gclFragmentSize, gclMarkPos[old+1].x - gclMarkPos[old].x); \
    /*}*/ \
}

    interpolate_segment(float4, y_1, 4, beg_1, end_1,
            scanline_interpolate_func);
    interpolate_segment(float4, y_2, 4, beg_2, end_2,
            scanline_interpolate_func);
}

kernel void fill_scanline(
        in      pos_t*  gclMarkPos,
        in      inf_t*  gclMarkInfo,
        in      float*  gclViewport,
        inout   uint*   gclFragmentSize,
        out     pos_t*  gclFragPos,
        out     inf_t*  gclFragInfo)
{
    size_t item_id = get_global_id(0),
           mark_id = item_id * 2;

    gclMarkPos  += mark_id;
    gclMarkInfo += mark_id;

    size_t len = gclMarkPos[1].x - gclMarkPos[0].x;

    if(gclFragPos == NULL || gclFragInfo == NULL) {
        atomic_add(gclFragmentSize, len);
        return;
    }

    float4 beg[2] = { gclMarkPos[0], gclMarkInfo[0] };
    float4 end[2] = { gclMarkPos[1], gclMarkInfo[1] };

#define fragment_interpolate_func(data, sz) { \
    /*if(is_in_viewport(&data[0], gclViewport)) {*/ \
        size_t old = atomic_inc(gclFragmentSize); \
        gclFragPos[old] = data[0]; \
        gclFragInfo[old] = data[1]; \
    /*}*/ \
}

    interpolate_segment(float4, len, 2, beg, end,
            fragment_interpolate_func);
}

kernel void depth_test(
        in      pos_t* gclFragPos,
        in      uint*  gclBufferSize,
        inout   int*   gclDepthBuffer)
{
    size_t item_id = get_global_id(0);
    gclFragPos += item_id;

    size_t coord =
        (size_t)gclFragPos->y * gclBufferSize[BS_WIDTH] +
        (size_t)gclFragPos->x;
    gclDepthBuffer += coord;

    //int integral_z = round(gclFragPos->z * (1 << 24));
    float floating_z = gclFragPos->z;
    int integral_z = *(int*)&floating_z;

    atomic_min(gclDepthBuffer, integral_z);
}

