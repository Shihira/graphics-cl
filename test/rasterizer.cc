// cflags: -lOpenCL

#include <fstream>
#include <string>
#include <limits>

#include "../include/comput.h"

using namespace std;
using namespace gcl;

string vert_shader_src = R"EOF(
void mul_mat4_vec4(global float4* out/*row-major*/, global float4 mat4[4], float4 in)
{
    out->x = dot(mat4[0], in);
    out->y = dot(mat4[1], in);
    out->z = dot(mat4[2], in);
    out->w = dot(mat4[3], in);
}

kernel void vertex_shader(
    global float4*  AttributeVertex,
    global float3*  AttributeNormal,
    global float4*  UniformMatrix, // row-major
    global float4*  InterpPosition,
    global float3*  InterpNormal,
    global float4*  InterpPositionWorld)
{
    size_t item_id = get_global_id(0);

    InterpPosition += item_id;
    InterpNormal += item_id;
    InterpPositionWorld += item_id;
    AttributeVertex += item_id;
    AttributeNormal += item_id;

    mul_mat4_vec4(InterpPosition, UniformMatrix, *AttributeVertex);
    *InterpPosition /= InterpPosition->w;
    *InterpNormal = *AttributeNormal;
    *InterpPositionWorld = *AttributeVertex;
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

void frag_main(
    float4 position,
    float3 normal,
    float4 positionWorld,
    float4* color)
{
    normal = normalize(normal);
    positionWorld /= positionWorld.w;

    float c = dot(normal, normalize(
        (float4)(-1.5, 3, 2, 1) - positionWorld).xyz);
    *color = (float4)(c, c, c, 1);
}

kernel void fragment_shader(
    global float3*  InterpNormal,
    global float4*  InterpPositionWorld,
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

    //int integral_z = round((gclFragPos->z) * (1 << 24));
    float floating_z = gclFragPos->z;
    int integral_z = *(int*)&floating_z;
    if(*gclDepthBuffer != integral_z) return;

    float4 color = (float4)(0, 0, 0, 1);
    frag_main(*gclFragPos,
        from_info_f3(gclFragInfo, InterpNormal),
        from_info_f4(gclFragInfo, InterpPositionWorld),
        &color);

    gclColorBuffer[coord] = color * 255;
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
    try {

    std::vector<platform> ps = platform::get();
    std::vector<device> ds = device::get(ps);
    context ctxt(ds.back());
    context_guard cg(ctxt);

    size_t num_vertices = end(vindices) - begin(vindices);
    size_t w = 200, h = 200;

    std::ifstream rast_src("../kernels/rasterizer.cl");

    program rast_prg = compile(rast_src, "-cl-kernel-arg-info");
    program vert_prg = compile(vert_shader_src, "-cl-kernel-arg-info");
    program frag_prg = compile(frag_shader_src, "-cl-kernel-arg-info");

    pipeline pl;

    pl.bind_kernel_from_program(frag_prg);
    pl.bind_kernel_from_program(vert_prg);
    pl.bind_kernel_from_program(rast_prg);

    kernel& krn_vs = *pl.get_kernel("vertex_shader");
    kernel& krn_fr = *pl.get_kernel("fragment_shader");
    kernel& krn_ms = *pl.get_kernel("mark_scanline");
    kernel& krn_fs = *pl.get_kernel("fill_scanline");
    kernel& krn_dt = *pl.get_kernel("depth_test");

    buffer<cl_float4>   AttributeVertex     (num_vertices, host_map);
    buffer<cl_float3>   AttributeNormal     (num_vertices, host_map);
    buffer<col4>        InterpPosition      (num_vertices, host_map);
    buffer<col3>        InterpNormal        (num_vertices, host_map);
    buffer<col4>        InterpPositionWorld (num_vertices, host_map);
    buffer<row4>        UniformMatrix       (4, host_map);
    buffer<float>       gclViewport         ({ 0, 0, float(w), float(h), });
    buffer<cl_uint>     gclMarkSize         ({ 0 });
    buffer<cl_uint>     gclFragmentSize     ({ 0 });
    buffer<col4>        gclMarkPos          (1000, host_map);
    buffer<col4>        gclMarkInfo         (1000, host_map);
    buffer<col4>        gclFragPos          (1000, host_map);
    buffer<col4>        gclFragInfo         (1000, host_map);
    buffer<cl_uint>     gclBufferSize       ({ cl_uint(w), cl_uint(h) });
    buffer<cl_int>      gclDepthBuffer      (w * h, host_map);
    buffer<cl_float4>   gclColorBuffer      (w * h, host_map);

    pl.auto_bind_buffer(AttributeVertex    );
    pl.auto_bind_buffer(AttributeNormal    );
    pl.auto_bind_buffer(InterpPosition     );
    pl.auto_bind_buffer(InterpNormal       );
    pl.auto_bind_buffer(InterpPositionWorld);
    pl.auto_bind_buffer(UniformMatrix      );
    pl.auto_bind_buffer(gclViewport        );
    pl.auto_bind_buffer(gclMarkSize        );
    pl.auto_bind_buffer(gclFragmentSize    );
    pl.auto_bind_buffer(gclMarkPos         );
    pl.auto_bind_buffer(gclMarkInfo        );
    pl.auto_bind_buffer(gclFragPos         );
    pl.auto_bind_buffer(gclFragInfo        );
    pl.auto_bind_buffer(gclBufferSize      );
    pl.auto_bind_buffer(gclDepthBuffer     );
    pl.auto_bind_buffer(gclColorBuffer     );

    for(size_t i = 0; i < num_vertices; i++) {
        AttributeVertex[i] = vertices[vindices[i]];
        AttributeNormal[i] = normals[nindices[i]];
    }

    mat4 pmat = tf::perspective(M_PI / 4, 1. / 1, 1, 10);
    mat4 mmat = tf::identity();
    mmat *= tf::translate(col4{ 0, 0, -3, 1 });
    mmat *= tf::rotate(-M_PI / 6, tf::yOz);
    mmat *= tf::rotate(-M_PI / 6, tf::zOx);
    mat4 rmat = pmat * mmat;

    cout << rmat << endl;
    for(size_t i = 0; i < 4; i++)
        UniformMatrix[i] = rmat.row(i);

    promise cp;

    cp <<
        fill(gclDepthBuffer, numeric_limits<int>::max()) <<
        fill(gclColorBuffer, cl_float4 { 255, 255, 255, 255 }) <<
        push(AttributeVertex) <<
        push(AttributeNormal) <<
        push(UniformMatrix) <<
        run(krn_vs, num_vertices) <<
        pull(InterpPosition) <<
        [&]() {
            for(size_t i = 0; i < num_vertices; i++)
                cout << InterpPosition[i] << endl;
        } <<
        push(gclViewport) <<
        push(gclMarkSize) <<
        push(gclFragmentSize) <<
        [&]() { krn_ms.set_null(krn_ms.get_index("gclMarkInfo")); } <<
        run(krn_ms, num_vertices / 3) <<
        pull(gclMarkSize) <<
        pull(gclFragmentSize) <<
        [&]() {
            if(gclMarkSize[0] > gclMarkPos.size() ||
                    gclMarkSize[0] > gclMarkInfo.size()) {
                size_t new_mark_size = 1 << size_t(std::log2(gclMarkSize[0])+1);
                gclMarkPos = buffer<col4>(new_mark_size, host_map);
                gclMarkInfo = buffer<col4>(new_mark_size, host_map);

                cout << "New Mark: " << new_mark_size << endl;
            }

            gclMarkSize[0] = 0;
            gclFragmentSize[0] = 0;

            pl.auto_bind_buffer(gclMarkPos);
            pl.auto_bind_buffer(gclMarkInfo);
        } <<
        wait_until_done <<
        push(gclMarkSize) <<
        push(gclFragmentSize) <<
        run(krn_ms, num_vertices / 3) <<
        pull(gclMarkSize) <<
        pull(gclFragmentSize) <<
        pull(gclMarkPos) <<
        pull(gclMarkInfo) <<
        [&]() {
            cout << gclMarkSize[0] << endl;
            cout << gclFragmentSize[0] << endl;

            for(size_t i = 0; i < gclMarkSize[0]; i++) {
                cout << gclMarkPos[i] << '\t';
                cout << gclMarkInfo[i] << endl;
            }   cout << endl << endl;

            krn_fs.range(gclMarkSize[0] / 2);

            if(gclFragmentSize[0] > gclFragPos.size() ||
                    gclFragmentSize[0] > gclFragInfo.size()) {
                size_t new_frag_size = 1 << size_t(std::log2(gclFragmentSize[0])+1);
                gclFragPos = buffer<col4>(new_frag_size, host_map);
                gclFragInfo = buffer<col4>(new_frag_size, host_map);

                cout << "New Frag: " << new_frag_size << endl;

                pl.auto_bind_buffer(gclFragPos);
                pl.auto_bind_buffer(gclFragInfo);
            }

            gclFragmentSize[0] = 0;
        } <<
        wait_until_done <<
        push(gclFragmentSize) <<
        run(krn_fs) <<
        pull(gclFragmentSize) <<
        pull(gclFragPos) <<
        pull(gclFragInfo) <<
        [&]() {
            cout << gclFragmentSize[0] << endl;
            for(size_t i = 0; i < gclFragmentSize[0]; i++) {
                cout << gclFragPos[i] << '\t';
                cout << gclFragInfo[i] << endl;
            }

            krn_dt.range(gclFragmentSize[0]);
            krn_fr.range(gclFragmentSize[0]);
        } <<
        wait_until_done <<
        push(gclBufferSize) <<
        run(krn_dt) <<
        pull(gclDepthBuffer) <<
        run(krn_fr) <<
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

    } catch(cl::Error e) { cout << e.what() << ' ' << e.err() << endl; }
}

