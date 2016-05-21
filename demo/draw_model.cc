// cflags: -lOpenCL -lSDL2

#include <fstream>
#include <string>

#include "../include/rasterizer.h"
#include "../include/model.h"
#include "../include/gui.h"

using namespace std;
using namespace gcl;

string vert_shader_src = R"EOF(
void mul_mat4_vec4(global float4* out/*row-major*/, global const float4 mat4[4], float4 in)
{
    out->x = dot(mat4[0], in);
    out->y = dot(mat4[1], in);
    out->z = dot(mat4[2], in);
    out->w = dot(mat4[3], in);
}

kernel void vertex_shader(
    global const float4*  AttributeVertex,
    global const float3*  AttributeNormal,
    global const float4*  UniformMatrix, // row-major
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
        (float4)(-15, 30, 100, 1) - positionWorld).xyz);
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

mat4 calculate_matrix(indexed_model& m)
{
    // find barycenter and max
    col4 barycenter = { 0, 0, 0, 0 };
    float max_coord = 0;
    for(auto v : m.attr_vertex) {
        v /= v[3];
        barycenter += v / v[3];
        for(int i = 0; i < 3; i++)
            if(max_coord < fabs(v[i])) max_coord = fabs(v[i]);
    }
    barycenter /= barycenter[3];

    mat4 mat =
        tf::perspective(M_PI / 4, 4. / 3, 10, 1000) *
        tf::translate(col4 { 0, 0, - max_coord - 100, 1 }) *
        tf::rotate(-M_PI / 6, tf::yOz); // *
        //tf::translate(-barycenter);

    return mat;
}

int main(int argc, char** argv)
{
    try {

    std::vector<platform> ps = platform::get();
    std::vector<device> ds = device::get(ps);
    context ctxt(ds.back());
    context_guard cg(ctxt);

    std::string obj_path;
    if(argc < 2)
        cerr << "Please provide the path of a Wavefront OBJ." << endl;
    else
        obj_path = argv[1];

    std::ifstream mod_src(obj_path);
    auto mod = wavefront_loader(mod_src);

    //size_t num_vertices = end(vindices) - begin(vindices);
    size_t num_vertices = mod.attr_vertex.size();
    //size_t num_vertices = 6;
    size_t w = 800, h = 600;

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

    buffer<col4>    AttributeVertex     (num_vertices, host_map);
    buffer<col3>    AttributeNormal     (num_vertices, host_map);
    buffer<col4>    InterpPosition      (num_vertices, host_map);
    buffer<col3>    InterpNormal        (num_vertices, host_map);
    buffer<col4>    InterpPositionWorld (num_vertices, host_map);
    buffer<row4>    UniformMatrix       (4, host_map);

    std::copy_n(mod.attr_vertex.begin(), num_vertices,
            AttributeVertex.begin());
    std::copy_n(mod.attr_normal.begin(), num_vertices,
            AttributeNormal.begin());

    mat4 rmat = calculate_matrix(mod);//pmat * mmat;

    rp.auto_bind_buffer(AttributeVertex    );
    rp.auto_bind_buffer(AttributeNormal    );
    rp.auto_bind_buffer(InterpPosition     );
    rp.auto_bind_buffer(InterpNormal       );
    rp.auto_bind_buffer(InterpPositionWorld);
    rp.auto_bind_buffer(UniformMatrix      );

    promise() <<
        push(AttributeVertex) <<
        push(AttributeNormal) <<
        wait_until_done;

    window win("Demo", w, h);

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    app().register_on_paint([&]() {
        struct timespec tmpts;
        clock_gettime(CLOCK_REALTIME, &tmpts);
        cout << "Interval: " << (tmpts.tv_nsec - ts.tv_nsec) / 1000000.0 << endl;

        rmat = rmat * tf::rotate(M_PI / 30, tf::zOx);
        for(size_t i = 0; i < 4; i++)
            UniformMatrix[i] = rmat.row(i);

        promise(true) <<
            push(UniformMatrix) <<
            wait_until_done;

        rp.render(true);

        clock_gettime(CLOCK_REALTIME, &tmpts);
        cout << "Render: " << (tmpts.tv_nsec - ts.tv_nsec) / 1000000.0 << endl;

        SDL_Surface* surf = win.sdl_surface();
        SDL_LockSurface(surf);
        uint32_t* p_end = ((uint32_t*)surf->pixels) + w * h;
        cl_uint* dev = rp.gclPixelBuffer.device_data();
        for(uint32_t* p = (uint32_t*)surf->pixels; p != p_end; ++p, ++dev) {
            *p = *dev;
        }
        SDL_UnlockSurface(surf);
        SDL_UpdateWindowSurface(win.sdl_window());

        clock_gettime(CLOCK_REALTIME, &tmpts);
        cout << "Sum: " << (tmpts.tv_nsec - ts.tv_nsec) / 1000000.0 << endl;
        cout << "============================================================" << endl;
        clock_gettime(CLOCK_REALTIME, &ts);
    });

    app().run();

    } catch(cl::Error e) { cout << e.what() << ' ' << e.err() << endl; }
}

