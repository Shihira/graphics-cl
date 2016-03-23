float4 from_info_f4(const float4 info, global float4 const* attr_array)
{
    attr_array += ((size_t)info.w) * 3;
    return attr_array[0] * info.x +
           attr_array[1] * info.y +
           attr_array[2] * info.z;
}

kernel void fs_main(
        global float4 const* inter_inf,
        global float4 const* inter_pos,

        global float4 const* in_color,
        global float4* out_color)
{
    size_t id = get_global_id(0);

    out_color[id] = from_info_f4(inter_inf[id], in_color);
}

kernel void generate_image(
        global float4 const* inter_inf,
        global float4 const* inter_pos,
        // user attributes
        global float4 const* in_color,

        global float4* color_buffer)
{
    size_t id = get_global_id(0);
    size_t coord = (size_t)(inter_pos[id].y) * 1024 + (size_t)(inter_pos[id].x);

    color_buffer[coord] = in_color[id];
}

kernel void clear_buffer(
        global float4* color_buffer)
{
    size_t id = get_global_id(0);
    color_buffer[id] = (float4)(255, 255, 255, 255);
}

