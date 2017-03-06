#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

#include <iostream>

#include "utilities.h"

#include "traits.h"
#include "reflection.h"
#include "exception.h"

namespace shrtool {

class image {
    friend struct image_geometry_helper__;

    size_t width_ = 0;
    size_t height_ = 0;

    mutable color* data_ = nullptr;
    // mark that if data_ should be deleted on destruction
    bool data_internal_ = true;

    color* lazy_data_() const;

public:
    typedef color* iterator;
    typedef color const* const_iterator;

    // create an new image
    image(size_t w = 0, size_t h = 0, color* data_ = nullptr) :
        width_(w), height_(h), data_(data_), data_internal_(!data_) { }
    image(const image& rhs) : width_(rhs.width_), height_(rhs.height_) {
        std::copy(rhs.begin(), rhs.end(), begin());
    }
    image(image&& rhs) : width_(rhs.width_), height_(rhs.height_) {
        // we cannot call data() here. just keep the current state.
        std::swap(data_, rhs.data_);
        std::swap(data_internal_, rhs.data_internal_);
    }

    image& operator=(const image& rhs) {
        width_ = rhs.width_;
        height_ = rhs.height_;
        std::copy(rhs.begin(), rhs.end(), begin());
        return *this;
    }

    image& operator=(image&& rhs) {
        width_ = rhs.width_;
        height_ = rhs.height_;
        std::swap(data_, rhs.data_);
        std::swap(data_internal_, rhs.data_internal_);
        return *this;
    }

    void flip_h();
    void flip_v();

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    iterator begin() { return data(); }
    iterator end() { return data() + width_ * height_; }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data() + width_ * height_; }
    const_iterator cbegin() const { return data(); }
    const_iterator cend() const { return data() + width_ * height_; }

    color& pixel(size_t l, size_t t) {
        return data()[t * width_ + l];
    }

    const color& pixel(size_t l, size_t t) const {
        return data()[t * width_ + l];
    }

    color* data() { return lazy_data_(); }
    color const* data() const { return lazy_data_(); }

    ~image() {
        if(data_ && data_internal_) delete[] data_;
    }

    void copy_pixel(size_t offx, size_t offy, size_t w, size_t h,
            image& dest, size_t dest_x, size_t dest_y) const;

    /* Many cubemap images have layouts as such:
     *    +Y
     * -Z -X +Z +X
     *    -Y
     */
    static image load_cubemap_from(const image& img);

    static void meta_reg_() {
        refl::meta_manager::reg_class<image>("image")
            .enable_auto_register()
            .enable_clone()
            .function("flip_h", &image::flip_h)
            .function("flip_v", &image::flip_v)
            .function("width", &image::width)
            .function("height", &image::height)
            .function("extract_cubemap", load_cubemap_from)
            .function("pixel", static_cast<color&(image::*)(size_t, size_t)>(&image::pixel));
    }
};

struct image_geometry_helper__ {
    static size_t& height(image& img) { return img.height_; }
    static size_t& width(image& img) { return img.width_; }
};

struct image_io_netpbm {
    image* img;

    image_io_netpbm(image& im) : img(&im) { }

    static image load(std::istream& is) {
        image im;
        load_into_image(is, im);
        return std::move(im);
    }

    std::istream& operator()(std::istream& is) {
        load_into_image(is, *img);
        return is;
    }

    std::ostream& operator()(std::ostream& os) {
        save_image(os, *img);
        return os;
    }

    static void load_into_image(std::istream& is, image& im);
    static void save_image(std::ostream& os, const image& im);
};

template<>
struct texture2d_trait<image> {
    typedef shrtool::raw_data_tag transfer_tag;
    typedef image input_type;

    static size_t width(const input_type& i) {
        return i.width();
    }

    static size_t height(const input_type& i) {
        return i.height();
    }

    static size_t format(const input_type& i) {
        return RGBA_U8888;
    }

    static const void* data(const input_type& i) {
        return i.data();
    }
};

}

#endif // IMAGE_H_INCLUDED
