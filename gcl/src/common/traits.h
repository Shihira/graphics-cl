#ifndef RES_TRAIT_INCLUDED
#define RES_TRAIT_INCLUDED

#include <cstddef>
#include <string>

namespace shrtool {

template<typename T>
struct plain_item_trait
{
    typedef T value_type;
    static constexpr size_t size() {
        return sizeof(value_type);
    }
    static constexpr size_t align() {
        return sizeof(value_type);
    }

    static void copy(const T& v, value_type* buf) { *buf = v; }

    static const char* glsl_type_name() {
        return "unknown";
    }
};

template<typename T>
struct item_trait : plain_item_trait<T> { };

template<>
struct item_trait<char> : plain_item_trait<char>
{
    static const char* glsl_type_name() { return "byte"; }
};

template<>
struct item_trait<int> : plain_item_trait<int>
{
    static const char* glsl_type_name() { return "int"; }
};

template<>
struct item_trait<double> : plain_item_trait<double>
{
    static const char* glsl_type_name() { return "double"; }
};

template<>
struct item_trait<float> : plain_item_trait<float>
{
    static const char* glsl_type_name() { return "float"; }
};

template<typename T, typename ... Enables>
struct item_trait_adapter {
};

template<typename T>
struct item_trait_adapter<T,
    decltype(item_trait<T>::size()),
    decltype(item_trait<T>::align())>
{
    static size_t size(const T&) {
        return item_trait<T>::size();
    }
    static size_t align(const T&) {
        return item_trait<T>::align();
    }
    static std::string glsl_type_name(const T&) {
        return item_trait<T>::glsl_type_name();
    }
    static void copy(const T& c, void* buf) {
        item_trait<T>::copy(c,
            reinterpret_cast<typename item_trait<T>::value_type*>(buf));
    }
};

template<typename T>
struct item_trait_adapter<T,
    decltype(item_trait<T>::size(*(const T*)nullptr)),
    decltype(item_trait<T>::align(*(const T*)nullptr))>
{
    static size_t size(const T& t) {
        return item_trait<T>::size(t);
    }
    static size_t align(const T& t) {
        return item_trait<T>::align(t);
    }
    static std::string glsl_type_name(const T& t) {
        return item_trait<T>::glsl_type_name(t);
    }
    static void copy(const T& c, void* buf) {
        item_trait<T>::copy(c,
            reinterpret_cast<typename item_trait<T>::value_type*>(buf));
    }
};

////////////////////////////////////////////////////////////////////////////////

struct raw_data_tag { };
struct indirect_tag { };

template<typename InputType, typename Enable = void>
struct attr_trait {
    /*
     * Please provide:
     */

    // typedef InputType input_type;
    // typedef shrtool::indirect_tag transfer_tag;
    // typedef float elem_type;
    //
    // static int slot(const input_type& i, size_t i_s);
    // static int count(const input_type& i);
    // static int dim(const input_type& i, size_t i_s);

    /*
     * For raw_data
     */
    // static elem_type const* data(const input_type& i, size_t i_s);
    /*
     * For indirect
     */
    // static void copy(const input_type& i, size_t i_s, elem_type* data);
};

template<typename InputType, typename Enable = void>
struct prop_trait {
    /*
     * Please provide:
     */

    //typedef InputType input_type;
    //typedef shrtool::indirect_tag transfer_tag;
    /* value_type determines the type of copy arguments o */
    //typedef float value_type;

    // static size_t size(const input_type& i);
    // static bool is_changed(const input_type& i);
    // static void mark_applied(input_type& i);
    /*
     * For raw_data
     */
    // static elem_type const* data(const input_type& i);
    /*
     * For indirect
     */
    // static void copy(const input_type& i, value_type* o);
};

template<typename InputType, typename Enable = void>
struct texture2d_trait {
    //typedef shrtool::raw_data_tag transfer_tag;
    //typedef InputType input_type;

    //static size_t width(const input_type& i) {
    //    return i.width();
    //}

    //static size_t height(const input_type& i) {
    //    return i.height();
    //}

    //static size_t format(const input_type& i) {
    //    return RGBA_U8888;
    //}

    //static const void* data(const input_type& i) {
    //    return i.data();
    //}
};

template<typename InputType, typename Enable = void>
struct shader_trait {
    //typedef InputType input_type;

    //static std::string source(const input_type& i, size_t e,
    //      shader::shader_type& t);
};

}

#endif // RES_TRAIT_INCLUDED

