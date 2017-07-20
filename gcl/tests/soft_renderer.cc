#include <vector>
#include <fstream>

#include "common/exception.h"
#include "common/mesh.h"
#include "common/image.h"

#include "gui.h"
#include "omp.h"

using namespace std;
using namespace gcl;
using namespace shrtool;
using namespace shrtool::math;

////////////////////////////////////////////////////////////////////////////////
// structures

struct constants {
    math::mat4 mvp_matrix;
    math::mat4 m_matrix;
    math::mat4 m_matrix_inv_t;

    int viewport_w;
    int viewport_h;
    color* color_buffer;
    math::col4 light_pos;
    math::col4 camera_pos;

    image* texture;
};

struct vertex_in {
    math::col4 position;
    math::col3 normal;
    math::col3 uvs;
};

struct vertex_out {
    math::col4 scrpos;
    math::col4 worldpos;
    math::col3 normal;
    math::col3 uvs;

    static vertex_out from_coef(
            const vertex_out& v0, double c0,
            const vertex_out& v1, double c1,
            const vertex_out& v2, double c2) {
        vertex_out o;

        o.scrpos = v0.scrpos * c0 + v1.scrpos * c1 + v2.scrpos * c2;

        double w = 1 / o.scrpos[3];
        double pw = fabs(w);

        o.worldpos = v0.worldpos * c0 + v1.worldpos * c1 + v2.worldpos * c2;
        o.worldpos *= pw;
        o.normal = v0.normal * c0 + v1.normal * c1 + v2.normal * c2;
        o.normal *= pw;
        o.uvs = v0.uvs * c0 + v1.uvs * c1 + v2.uvs * c2;
        o.uvs *= pw;

        return o;
    }
};

fcolor sampler(const image& img, float x, float y)
{
    x *= img.width();
    y *= img.height();
    x = clamp<double>(x, 0, img.width() - 2);
    y = clamp<double>(y, 0, img.height() - 2);

    fcolor color00, color10, color01, color11;
    img.quad(x, y, color00, color10, color01, color11);

    float x_left = x - floor(x), x_right = 1 + floor(x) - x;
    float y_left = y - floor(y), y_right = 1 + floor(y) - y;

    fcolor color0 = color10 * x_left + color00 * x_right;
    fcolor color1 = color11 * x_left + color01 * x_right;

    return color1 * y_left + color0 * y_right;
}

////////////////////////////////////////////////////////////////////////////////
// shaders

color surface_shader(const vertex_out& vo, constants& cnst)
{
    col3 light = col3(cnst.light_pos - vo.worldpos);
    col3 view = col3(cnst.camera_pos - vo.worldpos);
    light /= norm(light);
    view /= norm(view);

    col3 refl = - light + vo.normal * dot(light, vo.normal) * 2;
    double diffuse = dot(light, vo.normal);
    double specular = dot(refl, view);

    double s = diffuse * 0.7 + specular * specular * specular * 0.3 + 0.1;

    return sampler(*cnst.texture, vo.uvs[0], vo.uvs[1]) * s;
}

////////////////////////////////////////////////////////////////////////////////
// pipeline

void vertex_transformation(
        vector<vertex_in> input,
        constants& cnst,
        vector<vertex_out>& output) {
    output.resize(input.size());

    for(size_t i = 0; i < input.size(); i++) {
        vertex_out& out = output[i];
        vertex_in& in = input[i];

        out.scrpos = cnst.mvp_matrix * in.position;

        double w = clamp(out.scrpos[3], 1e-6, 1e6);
        double pw = fabs(w);

        out.scrpos[0] = (out.scrpos[0] / pw * 0.5 + 0.5) * cnst.viewport_w;
        out.scrpos[1] = (out.scrpos[1] / pw * 0.5 + 0.5) * cnst.viewport_h;
        out.scrpos[2] = (out.scrpos[2] / pw * 0.5 + 0.5);
        out.scrpos[3] = 1 / w;

        out.worldpos = cnst.m_matrix * in.position / pw;
        out.normal = col3(cnst.m_matrix_inv_t * col4(in.normal)) / pw;
        out.uvs = in.uvs / pw;
    }
}

void rasterize(const vertex_out vs[], constants& cnst) {
    if(cross(col3(vs[1].scrpos - vs[0].scrpos),
                col3(vs[2].scrpos - vs[1].scrpos))[2] < 0)
        return;

    int min_x = clamp<int>(floor(min(
            vs[0].scrpos[0],
            min(vs[1].scrpos[0],
                vs[2].scrpos[0]))), 0, cnst.viewport_w - 1);
    int max_x = clamp<int>(ceil(max(
            vs[0].scrpos[0],
            max(vs[1].scrpos[0],
                vs[2].scrpos[0]))), 0, cnst.viewport_w - 1);
    int min_y = clamp<int>(floor(min(
            vs[0].scrpos[1],
            min(vs[1].scrpos[1],
                vs[2].scrpos[1]))), 0, cnst.viewport_h - 1);
    int max_y = clamp<int>(ceil(max(
            vs[0].scrpos[1],
            max(vs[1].scrpos[1],
                vs[2].scrpos[1]))), 0, cnst.viewport_h - 1);

    auto term0 = [&](int a, int b) -> double {
        return (vs[a].scrpos[1] - vs[b].scrpos[1]);
    };

    auto term1 = [&](int a, int b) -> double {
        return (vs[b].scrpos[0] - vs[a].scrpos[0]);
    };

    auto term2 = [&](int a, int b) -> double {
        return (vs[a].scrpos[0] * vs[b].scrpos[1]) - (vs[b].scrpos[0] * vs[a].scrpos[1]);
    };

    double term0s[3] = { term0(1, 2), term0(2, 0), term0(0, 1) };
    double term1s[3] = { term1(1, 2), term1(2, 0), term1(0, 1) };
    double term2s[3] = { term2(1, 2), term2(2, 0), term2(0, 1) };

    auto f = [&](int v, double x, double y) -> double {
        return term0s[v] * x + term1s[v] * y + term2s[v];
    };

    double f0 = f(0, vs[0].scrpos[0], vs[0].scrpos[1]);
    double f1 = f(1, vs[1].scrpos[0], vs[1].scrpos[1]);
    double f2 = f(2, vs[2].scrpos[0], vs[2].scrpos[1]);

    #pragma omp parallel for schedule(dynamic)
    for(int y = min_y; y <= max_y; y++)
    for(int x = min_x; x <= max_x; x++) {
        double coef_0 = f(0, x + 0.5, y + 0.5) / f0;
        double coef_1 = f(1, x + 0.5, y + 0.5) / f1;
        double coef_2 = f(2, x + 0.5, y + 0.5) / f2;

        if(coef_0 > 1 || coef_0 < 0 ||
           coef_1 > 1 || coef_1 < 0 ||
           coef_2 > 1 || coef_2 < 0)
            continue;

        vertex_out vo = vertex_out::from_coef(
            vs[0], coef_0,
            vs[1], coef_1,
            vs[2], coef_2);

        if(vo.scrpos[2] > 1 || vo.scrpos[2] < 0)
            continue;

        color c = surface_shader(vo, cnst);
        std::swap(c.data.channels.r, c.data.channels.b);

        cnst.color_buffer[(cnst.viewport_h - y - 1) * cnst.viewport_w + x] = c;
    }
}

void assemble_primitives(
        vector<vertex_out>& input,
        constants& cnst) {
    GUARD_(input.size() % 3 == 0);

    for(size_t i = 0; i < input.size(); i += 3) {
        rasterize(&input[i], cnst);
    }
}

void clear_screen(
        constants& cnst) {
    color* buf = cnst.color_buffer;
    for(int y = 0; y < cnst.viewport_h; y++)
    for(int x = 0; x < cnst.viewport_w; x++) {
        buf->data.rgba = 0xff333333;
        buf++;
    }
}

int main()
{
    window w("Test");

    ////////////////////////////////////////////////////////////////////////////
    // presets
    mesh_box msh(2, 2, 2);
    std::ifstream ftex("../textures/texture.ppm");
    image img = image_io_netpbm::load(ftex);
    img.make_float_cache();

    vector<vertex_in> input;
    for(size_t i = 0; i < msh.vertices(); i++) {
        input.push_back(vertex_in {
                msh.positions[i],
                msh.normals[i],
                msh.uvs[i]
            });
    }

    constants cnst;
    cnst.viewport_w = 800;
    cnst.viewport_h = 600;
    cnst.light_pos = col4 { 0, 5, 3, 1 };
    cnst.camera_pos = col4 { 0, -0.25, 3, 1 };

    mat4 model_mat = tf::identity();
    mat4 view_mat =
        tf::translate(col4 {
                -cnst.camera_pos[0],
                -cnst.camera_pos[1],
                -cnst.camera_pos[2], 1 }) *
        tf::rotate(-math::PI / 6, tf::yOz);
    mat4 proj_mat = tf::perspective(math::PI / 6, 4.0 / 3, 1, 100);

    vector<color> color_buffer(800 * 600);

    ////////////////////////////////////////////////////////////////////////////
    application::inst().register_on_paint([&]() {
        GUARD_(w.sdl_surface()->format->BitsPerPixel == 32);
        GUARD_(SDL_LockSurface(w.sdl_surface()) >= 0);

        ////////////////////////////////////////////////////////////////////////
        // pipeline

        uint32_t* buf = (uint32_t*) w.sdl_surface()->pixels;

        //model_mat *= tf::rotate(-math::PI/120, tf::yOz);
        model_mat *= tf::rotate(-math::PI/120, tf::zOx);

        cnst.color_buffer = (color*) buf;
        cnst.texture = &img;
        cnst.mvp_matrix = proj_mat * view_mat * model_mat;
        cnst.m_matrix = model_mat;
        cnst.m_matrix_inv_t = transpose(inverse(model_mat));

        vector<vertex_out> vertex_output;
        clear_screen(cnst);
        vertex_transformation(input, cnst, vertex_output);
        assemble_primitives(vertex_output, cnst);

        SDL_UnlockSurface(w.sdl_surface());
        SDL_UpdateWindowSurface(w.sdl_window());
    });

    application::inst().register_on_mouse_wheel([&](int x, int y) {
        view_mat = tf::translate(col3 { 0, 0, y / 32.0 }) * view_mat;
    });

    application::inst().run();
}
