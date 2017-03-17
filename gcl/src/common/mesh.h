#ifndef MESH_H_INCLUDED
#define MESH_H_INCLUDED

#include <vector>
#include <array>
#include <memory>
#include <type_traits>

#include "matrix.h"
#include "image.h"
#include "traits.h"
#include "reflection.h"

namespace shrtool {

/*
 * mesh_base defines some semantic functions. If you are wondering why has_***
 * is not virtual functions, the answer is: mesh_base is a CRTP base class. you
 * have known the derived class when you know the base class, so you literally
 * have no chance to use a mesh_base reference. A CRTP base class has very
 * pure purpose: adding extra functionalities.
 */
template<typename Derived>
struct mesh_base {
private:
    Derived& self() { return *reinterpret_cast<Derived*>(this); }
    const Derived& self() const {
        return *reinterpret_cast<const Derived*>(this);
    }

public:
    typedef void mesh_tag;
    /*
     * // provide at least these member variables as follow
     * // where Container supports size() and operator[](size_t)
     *
     * Container<math::col4> positions;
     * Container<math::col3> normals;
     * Container<math::col3> uvs;
     */

    // you can define your has_** if the member variable's size
    // doesn't means any HAS or HASN'T
    bool has_positions() const { return self().positions.size() > 0; }
    bool has_normals() const { return self().normals.size() > 0; }
    bool has_uvs() const { return self().uvs.size() > 0; }

    bool empty() const {
        return self().positions.size() == 0;
    }

    math::col4& get_position(size_t tri, size_t vert)
        { return self().positions[tri * 3 + vert]; }
    const math::col4& get_position(size_t tri, size_t vert) const
        { return self().positions[tri * 3 + vert]; }
    math::col3& get_normal(size_t tri, size_t vert)
        { return self().normals[tri * 3 + vert]; }
    const math::col3& get_normal(size_t tri, size_t vert) const
        { return self().normals[tri * 3 + vert]; }
    math::col3& get_uv(size_t tri, size_t vert)
        { return self().uvs[tri * 3 + vert]; }
    const math::col3& get_uv(size_t tri, size_t vert) const
        { return self().uvs[tri * 3 + vert]; }

    size_t triangles() const { return vertices() / 3; }
    size_t vertices() const { return self().positions.size(); }
};

struct mesh_basic : mesh_base<mesh_basic> {
public:
    std::vector<math::col4> positions;
    std::vector<math::col3> normals;
    std::vector<math::col3> uvs;

    mesh_basic() { }

    mesh_basic(mesh_basic&& mp) :
        positions(std::move(mp.positions)),
        normals(std::move(mp.normals)),
        uvs(std::move(mp.uvs)) { }

    mesh_basic(const mesh_basic& mp) :
        positions(mp.positions),
        normals(mp.normals),
        uvs(mp.uvs) { }
};

/*
 * indexed_attr give great convenience to people who want to operate on data
 * through indices access, and indexed_attr will do you the conversion.
 */
template<typename T, typename ContainerRef = std::vector<T>*>
struct indexed_attr {
    /*
     * built-in iterator for special purpose
     */
    template<typename ValType, typename Ref = indexed_attr&>
    struct indexed_attr_iterator :
            std::iterator<std::bidirectional_iterator_tag, ValType> {
        indexed_attr_iterator(Ref ia, size_t ioi) :
            ia_(ia), index_of_idx_(ioi) { }
        indexed_attr_iterator(const indexed_attr_iterator& other) :
            ia_(other.ia_), index_of_idx_(other.index_of_idx_) { }

        typedef indexed_attr_iterator self_type;

        self_type& operator++() { index_of_idx_++; return *this; }
        self_type operator++(int) {
            self_type other = *this; index_of_idx_++; return other; }
        self_type& operator--() { index_of_idx_--; return *this; }
        self_type operator--(int) {
            self_type other = *this; index_of_idx_--; return other; }

        bool operator!=(const self_type& rhs) const {
            return index_of_idx_ != rhs.index_of_idx_;
        }
        bool operator==(const self_type& rhs) const {
            return index_of_idx_ == rhs.index_of_idx_;
        }

        ValType& operator*() const { return (*ia_.refer)[ia_.indices[index_of_idx_]]; }

        self_type& operator=(const self_type& other) {
            index_of_idx_ = other.index_of_idx_;
        }
    private:
        Ref ia_;
        size_t index_of_idx_;
    };

    // refer is itself a reference, which binds itself with its parent, instead
    // of binding to the container.
    ContainerRef& refer;

    typedef ContainerRef& container_type_refernce;
    typedef T value_type;
    std::vector<size_t> indices;

    typedef indexed_attr_iterator<T, indexed_attr&> iterator;
    typedef indexed_attr_iterator<const T, const indexed_attr&> const_iterator;

    indexed_attr(ContainerRef& r) : refer(r) { }
    indexed_attr(ContainerRef& r, const std::vector<size_t>& i) :
        refer(r), indices(i) { }
    indexed_attr(ContainerRef& r, std::vector<size_t>&& i) :
        refer(r), indices(std::move(i)) { }

    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, indices.size()); }

    const_iterator begin() const { return const_iterator(*this, 0); }
    const_iterator end() const { return const_iterator(*this, indices.size()); }

    size_t size() const {
        return indices.size();
    }

    T& operator[](size_t i) { return (*refer)[indices[i]]; }
    const T& operator[](size_t i) const { return (*refer)[indices[i]]; }
};


struct mesh_indexed : mesh_base<mesh_indexed> {
    template<typename T>
    using stor_ptr = std::shared_ptr<std::vector<T>>;

    stor_ptr<math::col4> stor_positions;
    stor_ptr<math::col3> stor_normals;
    stor_ptr<math::col3> stor_uvs;

    indexed_attr<math::col4, stor_ptr<math::col4>> positions;
    indexed_attr<math::col3, stor_ptr<math::col3>> normals;
    indexed_attr<math::col3, stor_ptr<math::col3>> uvs;

    bool has_positions() const {
        return
            stor_positions &&
            stor_positions->size() &&
            positions.size() > 0;
    }

    bool has_normals() const {
        return
            stor_normals &&
            stor_normals->size() &&
            normals.size() > 0;
    }

    bool has_uvs() const {
        return
            stor_uvs &&
            stor_uvs->size() &&
            uvs.size() > 0;
    }

    // default constructor: initialize storage
    mesh_indexed(bool init_stor = true) :
        positions(stor_positions),
        normals(stor_normals),
        uvs(stor_uvs) {
        if(init_stor) {
            stor_positions.reset(new std::vector<math::col4>);
            stor_normals.reset(new std::vector<math::col3>);
            stor_uvs.reset(new std::vector<math::col3>);
        }
    }

    mesh_indexed(const mesh_indexed& im) :
        stor_positions(im.stor_positions),
        stor_normals(im.stor_normals),
        stor_uvs(im.stor_uvs),
        positions(stor_positions, im.positions.indices),
        normals(stor_normals, im.normals.indices),
        uvs(stor_uvs, im.uvs.indices) { }

    mesh_indexed(mesh_indexed&& im) :
        stor_positions(std::move(im.stor_positions)),
        stor_normals(std::move(im.stor_normals)),
        stor_uvs(std::move(im.stor_uvs)),
        positions(stor_positions, std::move(im.positions.indices)),
        normals(stor_normals, std::move(im.normals.indices)),
        uvs(stor_uvs, std::move(im.uvs.indices)) { }

    static mesh_indexed gen_uv_sphere(double radius,
            size_t tesel_u, size_t tesel_v, bool smooth = true);
    static mesh_indexed gen_plane(double w, double h,
            size_t tesel_u, size_t tesel_v);
    static mesh_indexed gen_box(double l, double w, double h);

    static void meta_reg_() {
        refl::meta_manager::reg_class<mesh_indexed>("mesh")
            .enable_auto_register()
            .function("has_positions", &mesh_indexed::has_positions)
            .function("has_normals", &mesh_indexed::has_normals)
            .function("has_uvs", &mesh_indexed::has_uvs)
            .function("triangles", static_cast<size_t (mesh_indexed::*)() const>(&mesh_indexed::triangles))
            .function("vertices", static_cast<size_t (mesh_indexed::*)() const>(&mesh_indexed::vertices))
            .function("gen_uv_sphere", gen_uv_sphere)
            .function("gen_box", gen_box)
            .function("gen_plane", gen_plane)
            .function("get_position", static_cast<math::col4& (mesh_indexed::*) (size_t, size_t)>(&mesh_indexed::get_position))
            .function("get_normals", static_cast<math::col3& (mesh_indexed::*) (size_t, size_t)>(&mesh_indexed::get_normal))
            .function("get_uv", static_cast<math::col3& (mesh_indexed::*) (size_t, size_t)>(&mesh_indexed::get_uv));
    }
};

struct mesh_uv_sphere : mesh_indexed {
    mesh_uv_sphere(double radius,
            size_t tesel_u, size_t tesel_v, bool smooth = true);

    mesh_uv_sphere(const mesh_uv_sphere& mp) : mesh_indexed(mp) { }
    mesh_uv_sphere(mesh_uv_sphere&& mp) : mesh_indexed(std::move(mp)) { }
};

struct mesh_plane : mesh_indexed {
    mesh_plane(double w, double h,
            size_t tesel_u, size_t tesel_v);

    mesh_plane(const mesh_plane& mp) : mesh_indexed(mp) { }
    mesh_plane(mesh_plane&& mp) : mesh_indexed(std::move(mp)) { }
};

struct mesh_box : mesh_indexed {
    mesh_box(double l, double w, double h);

    mesh_box(const mesh_box& mb) : mesh_indexed(mb) { }
    mesh_box(mesh_box&& mb) : mesh_indexed(std::move(mb)) { }
};

inline mesh_indexed mesh_indexed::gen_uv_sphere(double radius,
        size_t tesel_u, size_t tesel_v, bool smooth) {
    return mesh_uv_sphere(radius, tesel_u, tesel_v, smooth);
}
inline mesh_indexed mesh_indexed::gen_plane(double w, double h,
        size_t tesel_u, size_t tesel_v) {
    return mesh_plane(w, h, tesel_u, tesel_v);
}
inline mesh_indexed mesh_indexed::gen_box(double l, double w, double h) {
    return mesh_box(l, w, h);
}

struct mesh_io_object {
    typedef mesh_indexed mesh_type;
    typedef std::vector<mesh_type> meshes_type;

    meshes_type* meshes;

    mesh_io_object(meshes_type& ms) : meshes(&ms) { }

    static meshes_type load(std::istream& is) {
        meshes_type ms;
        load_into_meshes(is, ms);
        return std::move(ms);
    }

    std::istream& operator()(std::istream& is) {
        load_into_meshes(is, *meshes);
        return is;
    }

    static void load_into_meshes(std::istream& is, meshes_type& ms);
};

template<typename T>
struct attr_trait<T, typename T::mesh_tag> {
    typedef T input_type;
    typedef shrtool::indirect_tag transfer_tag;
    typedef float elem_type;
    
    static int slot(const input_type& i, size_t i_s) {
        switch(i_s) {
        case 0:
            if(i.has_positions()) return 0;
            break;
        case 1:
            if(i.has_normals()) return 1;
            break;
        case 2:
            if(i.has_uvs()) return 2;
            break;
        }

        return -1;
    }

    static int count(const input_type& i) {
        return i.vertices();
    }

    static int dim(const input_type& i, size_t i_s) {
        static const int dims[3] = { 4, 3, 3, };
        return dims[i_s];
    }

    template<typename Vecs>
    static void copy_cols(Vecs& vs, elem_type* data) {
        for(auto& vec : vs) {
            for(size_t i = 0; i < Vecs::value_type::rows; ++i)
                *(data++) = vec[i];
        }
    }

    static void copy(const input_type& i, size_t i_s, elem_type* data) {
        switch(i_s) {
        case 0: copy_cols(i.positions, data); break;
        case 1: copy_cols(i.normals, data); break;
        case 2: copy_cols(i.uvs, data); break;
        }
    }
};

template<typename T>
inline math::col4 find_average(const T& m)
{
    math::col4 sum = { 0, 0, 0, 0 };
    for(int i = 0; i < attr_trait<T>::count(m); i++) {
        sum += m.positions[i];
    }
    return sum / double(attr_trait<T>::count(m));
}

}

#endif // MESH_H_INCLUDED
