kernel void vs_main(
        global float4 const* in_vertices,
        global float4 const* in_color,
        global float4* out_positions,
        global float4* out_color)
{
    size_t id = get_global_id(0);

    out_positions[id] = in_vertices[id];
    out_color[id] = in_color[id];
}

