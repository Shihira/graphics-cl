// cflags: -lOpenCL

#include <fstream>
#include <string>

#include "../include/comput.h"

using namespace std;
using namespace gcl;

string vert_shader_src = R"EOF(
kernel void vertex_shader(
    global float4*  gclAbbrVertex,
    global float3*  gclAbbrNormal,
    global float4*  gclMatrix, // row-major
    global float4*  gclPosition,
    global float3*  gclNormal,
    global float4*  gclPositionWorld)
{
    size_t item_id = get_global_id(0);

    gclPosition += item_id;
    gclNormal += item_id;
    gclPositionWorld += item_id;
    gclAbbrVertex += item_id;
    gclAbbrNormal += item_id;

    gclPosition->x = dot(gclMatrix[0], *gclAbbrVertex);
    gclPosition->y = dot(gclMatrix[1], *gclAbbrVertex);
    gclPosition->z = dot(gclMatrix[2], *gclAbbrVertex);
    gclPosition->w = dot(gclMatrix[3], *gclAbbrVertex);

    *gclPosition /= gclPosition->w;
    *gclNormal = *gclAbbrNormal;
    *gclPositionWorld = *gclAbbrVertex;
}
)EOF";

string frag_shader_src = R"EOF(
float4 from_info_f4(global const float4* info, global const float4* attr_array)
{
    attr_array += ((size_t)info->w) * 3;
    return attr_array[0] * info->x +
           attr_array[1] * info->y +
           attr_array[2] * info->z;
}

float3 from_info_f3(global const float4* info, global const float3* attr_array)
{
    attr_array += ((size_t)info->w) * 3;
    return attr_array[0] * info->x +
           attr_array[1] * info->y +
           attr_array[2] * info->z;
}

kernel void fragment_shader(
    global float4*  gclPosition,
    global float3*  gclNormal,
    global float4*  gclPositionWorld,
    global float4*  gclFragPos,
    global float4*  gclFragInfo,
    global float4*  gclColorBuffer,
    global uint*    gclBufferSize,
    global int*     gclDepthBuffer)
{
    size_t item_id = get_global_id(0);
    gclFragPos += item_id;
    gclFragInfo += item_id;
    size_t coord =
        (size_t)gclFragPos->y * gclBufferSize[0] +
        (size_t)gclFragPos->x;
    gclDepthBuffer += coord;

    int integral_z = round((gclFragPos->z) * (1 << 24));
    if((*gclDepthBuffer) < integral_z) return;

    float3 normal = normalize(from_info_f3(gclFragInfo, gclNormal));
    float4 pos = from_info_f4(gclFragInfo, gclPositionWorld);
    pos /= pos.w;

    //float c = 255 * gclFragPos->z;
    float c = 255 * dot(normal, normalize((float4)(-1.5, 3, 2, 1) - pos).xyz);
    gclColorBuffer[coord] = (float4)(c, c, c, 255);
}
)EOF";

const cl_float4 vertices[] = {
    cl_float4 {  1,  1,  1, 1 },
    cl_float4 {  1,  1, -1, 1 },
    cl_float4 {  1, -1,  1, 1 },
    cl_float4 {  1, -1, -1, 1 },
    cl_float4 { -1,  1,  1, 1 },
    cl_float4 { -1,  1, -1, 1 },
    cl_float4 { -1, -1,  1, 1 },
    cl_float4 { -1, -1, -1, 1 },
};

const cl_float4 normals[] = {
    cl_float4 { 1, 0, 0 },
    cl_float4 { -1, 0, 0 },
    cl_float4 { 0, 1, 0 },
    cl_float4 { 0, -1, 0 },
    cl_float4 { 0, 0, 1 },
    cl_float4 { 0, 0, -1 },
};

const size_t vindices[] = {
    2, 3, 1,    1, 0, 2,
    4, 5, 7,    7, 6, 4,
    1, 5, 4,    4, 0, 1,
    2, 6, 7,    7, 3, 2,
    0, 4, 6,    6, 2, 0,
    3, 7, 5,    5, 1, 3,
};

const size_t nindices[] = {
    0, 0, 0,    0, 0, 0,
    1, 1, 1,    1, 1, 1,
    2, 2, 2,    2, 2, 2,
    3, 3, 3,    3, 3, 3,
    4, 4, 4,    4, 4, 4,
    5, 5, 5,    5, 5, 5,
};

int main()
{
    std::vector<platform> ps = platform::get();
    std::vector<device> ds = device::get(ps);
    context ctxt(ds.back());
    context_guard cg(ctxt);

    std::ifstream rst_src("../kernels/rasterizer.cl");
    program prg = compile(rst_src, "-cl-kernel-arg-info");
    program vert_prg = compile(vert_shader_src);
    program frag_prg = compile(frag_shader_src);

    kernel krn_vs(vert_prg, "vertex_shader");
    kernel krn_fr(frag_prg, "fragment_shader");
    kernel krn_ms(prg, "mark_scanline");
    kernel krn_fs(prg, "fill_scanline");
    kernel krn_dt(prg, "depth_test");
    kernel krn_ia(prg, "image_assembly");

    size_t num_vertices = end(vindices) - begin(vindices);
    size_t w = 200, h = 200;

    buffer<cl_float4> gclAbbrVertex(num_vertices, host_map);
    buffer<cl_float3> gclAbbrNormal(num_vertices, host_map);
    buffer<col4> gclPosition(num_vertices, host_map);
    buffer<col3> gclNormal(num_vertices, host_map);
    buffer<col3> gclPositionWorld(num_vertices, host_map);
    buffer<row4> gclMatrix(4, host_map);
    buffer<float> gclViewport { 0, 0, float(w), float(h), };
    buffer<cl_uint> gclMarkSize { 0 };
    buffer<cl_uint> gclFragmentSize { 0 };
    buffer<col4> gclMarkPos(4000, host_map);
    buffer<col4> gclMarkInfo(4000, host_map);
    buffer<col4> gclFragPos(20000, host_map);
    buffer<col4> gclFragInfo(20000, host_map);
    buffer<cl_uint> gclBufferSize { cl_uint(w), cl_uint(h) };
    buffer<cl_int> gclDepthBuffer(w * h, host_map);
    buffer<cl_float4> gclColorBuffer(w * h, host_map);

    for(size_t i = 0; i < num_vertices; i++) {
        gclAbbrVertex[i] = vertices[vindices[i]];
        gclAbbrNormal[i] = normals[nindices[i]];
    }

    mat4 pmat = tf::perspective(M_PI / 4, 1. / 1, 1, 10);
    mat4 mmat = tf::identity();
    mmat *= tf::translate(col4{ 0, 0, -3, 1 });
    mmat *= tf::rotate(-M_PI / 6, tf::yOz);
    mmat *= tf::rotate(-M_PI / 6, tf::zOx);
    mat4 rmat = pmat * mmat;

    cout << rmat << endl;
    for(size_t i = 0; i < 4; i++)
        gclMatrix[i] = rmat.row(i);

    promise cp;

    krn_vs.set_buffer(0, gclAbbrVertex);
    krn_vs.set_buffer(1, gclAbbrNormal);
    krn_vs.set_buffer(2, gclMatrix);
    krn_vs.set_buffer(3, gclPosition);
    krn_vs.set_buffer(4, gclNormal);
    krn_vs.set_buffer(5, gclPositionWorld);

    cp <<
        push(gclAbbrVertex) <<
        push(gclAbbrNormal) <<
        push(gclMatrix) <<
        run(krn_vs, num_vertices) <<
        pull(gclPosition) <<
        wait_until_done;

    for(size_t i = 0; i < num_vertices; i++)
        cout << gclPosition[i] << endl;

    krn_ms.set_buffer(0, gclPosition);
    krn_ms.set_buffer(1, gclViewport);
    krn_ms.set_buffer(2, gclMarkSize);
    krn_ms.set_buffer(3, gclFragmentSize);
    krn_ms.set_buffer(4, gclMarkPos);
    krn_ms.set_buffer(5, gclMarkInfo);

    cp <<
        //push(gclPosition) <<
        push(gclViewport) <<
        push(gclMarkSize) <<
        push(gclFragmentSize) <<
        run(krn_ms, num_vertices / 3) <<
        pull(gclMarkSize) <<
        pull(gclFragmentSize) <<
        pull(gclMarkPos) <<
        pull(gclMarkInfo) <<
        wait_until_done;

    cout << gclMarkSize[0] << endl;
    cout << gclFragmentSize[0] << endl;
    for(size_t i = 0; i < gclMarkSize[0]; i++) {
        cout << gclMarkPos[i] << '\t';
        cout << gclMarkInfo[i] << endl;
    }   cout << endl << endl;

    gclFragmentSize[0] = 0;

    krn_fs.set_buffer(0, gclMarkPos);
    krn_fs.set_buffer(1, gclMarkInfo);
    krn_fs.set_buffer(2, gclViewport);
    krn_fs.set_buffer(3, gclFragmentSize);
    krn_fs.set_buffer(4, gclFragPos);
    krn_fs.set_buffer(5, gclFragInfo);

    cp <<
        push(gclFragmentSize) <<
        run(krn_fs, gclMarkSize[0] / 2) <<
        pull(gclFragmentSize) <<
        pull(gclFragPos) <<
        pull(gclFragInfo) <<
        wait_until_done;

    cout << gclFragmentSize[0] << endl;
    for(size_t i = 0; i < gclFragmentSize[0]; i++) {
        cout << gclFragPos[i] << '\t';
        cout << gclFragInfo[i] << endl;
    }

    krn_dt.set_buffer(0, gclFragPos);
    krn_dt.set_buffer(1, gclBufferSize);
    krn_dt.set_buffer(2, gclDepthBuffer);

    cp <<
        push(gclBufferSize) <<
        fill(gclDepthBuffer, (1 << 24)) <<
        run(krn_dt, gclFragmentSize[0]) <<
        pull(gclDepthBuffer) <<
        wait_until_done;

    krn_fr.set_buffer(0, gclPosition);
    krn_fr.set_buffer(1, gclNormal);
    krn_fr.set_buffer(2, gclPositionWorld);
    krn_fr.set_buffer(3, gclFragPos);
    krn_fr.set_buffer(4, gclFragInfo);
    krn_fr.set_buffer(5, gclColorBuffer);
    krn_fr.set_buffer(6, gclBufferSize);
    krn_fr.set_buffer(7, gclDepthBuffer);

    cp <<
        fill(gclColorBuffer, cl_float4 { 255, 255, 255, 255 }) <<
        run(krn_fr, gclFragmentSize[0]) <<
        pull(gclColorBuffer) <<
        wait_until_done;

    ofstream f("./test.ppm");
    f << "P6\n" << w << "\n" << h << "\n255\n";
    for(size_t i = 0; i < h; i++)
    for(size_t j = 0; j < w; j++) {
        size_t coord = (h - i - 1) * w + j;
        f <<
            uint8_t(gclColorBuffer[coord].s[0]) <<
            uint8_t(gclColorBuffer[coord].s[1]) <<
            uint8_t(gclColorBuffer[coord].s[2]);
    }
}

