#include <limits>
#include <iomanip>
#include <sstream>

#include "image.h"

namespace shrtool {

color* image::lazy_data_() const
{
    if(!data_) {
        if(!width_ || !height_)
            throw restriction_error("Zero size image");
        data_ = new color[width_ * height_];
    }
    return data_;
}

void image::copy_pixel(size_t offx, size_t offy, size_t w, size_t h,
        image& dest, size_t dest_x, size_t dest_y) const {
    if(offx + w > width() || offy + h > height() ||
            dest_x + w > dest.width() || dest_y + h > dest.height()) {
        throw restriction_error("Out of bound");
    }

    const color* csrc = data() + width() * offy + offx;
    color* cdest = dest.data() + dest.width() * dest_y + dest_x;

    for(size_t y = 0; y < h; y++) {
        for(size_t x = 0; x < w; x++, cdest++, csrc++) {
            *cdest = *csrc;
        }
        csrc += width() - w;
        cdest += dest.width() - w;
    }
}

/* 0    +Y
 * 1 -X +Z +X -Z
 * 2    -Y
 *   0  1  2  3
 */
image image::load_cubemap_from(const image& img) {
    if(img.width() % 4 || img.height() % 3 ||
            img.width() / 4 != img.height() / 3) {
        throw restriction_error("Size of cubmap is not regular");
    }

    size_t unit = img.width() / 4;

    static size_t coords[6][2] = {
        { 2, 1 }, { 0, 1 },
        { 1, 0 }, { 1, 2 },
        { 1, 1 }, { 3, 1 },
    };

    image new_img(unit, unit * 6);
    for(size_t i = 0; i < 6; i++) {
        img.copy_pixel(coords[i][0] * unit, coords[i][1] * unit,
                unit, unit, new_img, 0, i * unit);
    }

    return std::move(new_img);
}
static void load_netpbm_body_plain(std::istream& is, image& im, size_t space)
{
for(auto i = im.begin(); i != im.end(); ++i) {
    uint16_t channels;
    for(size_t c = 0; c < 3; ++c) {
        is >> channels >> std::ws;

        if(is.fail()) {
            if(is.eof()) throw parse_error(
                    "EOF too early while reading image");
            else throw parse_error("Bad Netpbm image: body");
        }

        uint8_t channels_byte = channels;
        if(space > 255)
            channels_byte = channels / ((space + 1) / 256);
        else if(space < 255)
            channels_byte = channels * 256 / (space + 1) + channels;
        i->data.bytes[c] = channels_byte;
    }
}
}

template<typename CompT>
static void load_netpbm_body_raw(std::istream& is, image& im, size_t space)
{
    for(auto i = im.begin(); i != im.end(); ++i) {
        CompT channels;
        for(size_t c = 0; c < 3; c++) {
            is.read((char*)(&channels), sizeof(channels));

            if(is.fail()) {
                if(is.eof()) throw parse_error(
                        "EOF too early while reading image");
                else throw parse_error("Bad Netpbm image: body");
            }

            uint8_t channels_byte = channels;
            if(space > 255)
                channels_byte = channels / ((space + 1) / 256);
            else if(space < 255)
                channels_byte = channels * 256 / (space + 1) + channels;
            i->data.bytes[c] = channels_byte;
        }
    }
}

static std::istream& load_netpbm_consume_comment(std::istream& is)
{
    while(is.peek() == '#') {
        is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        is >> std::ws;
    }

    return is;
}

void image_io_netpbm::load_into_image(std::istream& is, image& im)
{
    char P = 0; char num = 0;
    is >> P >> num >> std::ws;
    if(P != 'P' || !std::isdigit(num)) {
        throw unsupported_error("Bad Netpbm image: magic mumber");
    }


    size_t w = 0, h = 0, space = 0; char whsc = 0;
    is  >> load_netpbm_consume_comment >> w
        >> load_netpbm_consume_comment >> h
        >> load_netpbm_consume_comment >> space;

    is.get(whsc); // spec: exact single whitespace
    if(!w || !h || !space || !isspace(whsc))
        throw parse_error("Bad Netpbm image: properties");

    image_geometry_helper__::width(im) = w;
    image_geometry_helper__::height(im) = h;

    if(num == '3')
        load_netpbm_body_plain(is, im, space);
    else if(num == '6') {
        if(space > 255)
            load_netpbm_body_raw<uint16_t>(is, im, space);
        else
            load_netpbm_body_raw<uint8_t>(is, im, space);
    } else
        throw unsupported_error("Format other than PPM is unsupported");
}

void image_io_netpbm::save_image(std::ostream& os, const image& im)
{
    os << "P6\n# created by shrtool\n"
        << im.width() << ' ' << im.height() << "\n255\n";
    for(const color& c : im) {
        os.put(c.data.channels.r);
        os.put(c.data.channels.g);
        os.put(c.data.channels.b);
    }
}

void image::flip_h()
{
    for(size_t r = 0; r < height(); r++) {
        for(size_t i = 0, j = width() - 1; i < j; ++i, --j)
            std::swap(pixel(i, r), pixel(j, r));
    }
}

void image::flip_v()
{
    for(size_t c = 0; c < width(); c++) {
        for(size_t i = 0, j = height() - 1; i < j; ++i, --j)
            std::swap(pixel(c, i), pixel(c, j));
    }
}

}
