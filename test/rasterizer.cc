// cflags: -lOpenCL

#include <fstream>
#include <string>

#include "../include/rasterizer.h"

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
    size_t w = 1024, h = 768;

    std::ifstream rast_src("../kernels/rasterizer.cl");

    program rast_prg = compile(rast_src, "-cl-kernel-arg-info");
    program vert_prg = compile(vert_shader_src, "-cl-kernel-arg-info");
    program frag_prg = compile(frag_shader_src, "-cl-kernel-arg-info");

    rasterizer_pipeline rp;
    rp.set_size(w, h);
    rp.set_vertex_number(num_vertices);
    rp.set_rasterizer_program(rast_prg);
    rp.set_vertex_shader_program(vert_prg);
    rp.set_fragment_shader_program(frag_prg);

    buffer<cl_float4>   AttributeVertex     (num_vertices, host_map);
    buffer<cl_float3>   AttributeNormal     (num_vertices, host_map);
    buffer<col4>        InterpPosition      (num_vertices, host_map);
    buffer<col3>        InterpNormal        (num_vertices, host_map);
    buffer<col4>        InterpPositionWorld (num_vertices, host_map);
    buffer<row4>        UniformMatrix       (4, host_map);

    for(size_t i = 0; i < num_vertices; i++) {
        AttributeVertex[i] = vertices[vindices[i]];
        AttributeNormal[i] = normals[nindices[i]];
    }

    mat4 pmat = tf::perspective(M_PI / 4, 4. / 3, 1, 10);
    mat4 mmat = tf::identity();
    mmat *= tf::translate(col4{ 0, 0, -3, 1 });
    mmat *= tf::rotate(-M_PI / 6, tf::yOz);
    mmat *= tf::rotate(-M_PI / 6, tf::zOx);
    mat4 rmat = pmat * mmat;

    cout << rmat << endl;
    for(size_t i = 0; i < 4; i++)
        UniformMatrix[i] = rmat.row(i);

    rp.auto_bind_buffer(AttributeVertex    );
    rp.auto_bind_buffer(AttributeNormal    );
    rp.auto_bind_buffer(InterpPosition     );
    rp.auto_bind_buffer(InterpNormal       );
    rp.auto_bind_buffer(InterpPositionWorld);
    rp.auto_bind_buffer(UniformMatrix      );

    promise() <<
        push(AttributeVertex) <<
        push(AttributeNormal) <<
        push(UniformMatrix) <<
        wait_until_done;

    clock_t c = clock();
    rp.render();
    cout << (clock() - c) / 1000.0 << endl;
    c = clock();
    rp.render();
    cout << (clock() - c) / 1000.0 << endl;

    ofstream f("./test.ppm");
    f << "P6\n" << w << "\n" << h << "\n255\n";
    for(size_t i = 0; i < h; i++)
    for(size_t j = 0; j < w; j++) {
        size_t coord = (h - i - 1) * w + j;
        f <<
            uint8_t(rp.gclColorBuffer[coord].s[0]) <<
            uint8_t(rp.gclColorBuffer[coord].s[1]) <<
            uint8_t(rp.gclColorBuffer[coord].s[2]);
    }

    } catch(cl::Error e) { cout << e.what() << ' ' << e.err() << endl; }
}

