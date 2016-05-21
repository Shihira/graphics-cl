#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include <vector>
#include <sstream>

#include "matrix.h"

namespace gcl {

struct model {
    std::vector<col4> attr_vertex;
    std::vector<col3> attr_normal;
    std::vector<col3> attr_uv;

    model() { }
    model(model&& other) :
        attr_vertex(std::move(other.attr_vertex)),
        attr_normal(std::move(other.attr_normal)),
        attr_uv(std::move(other.attr_uv)) { }
    model(const model& other) :
        attr_vertex(other.attr_vertex),
        attr_normal(other.attr_normal),
        attr_uv(other.attr_uv) { }

    model triangle(size_t idx) {
        model tri;
        idx *= 3;

        for(size_t i = 0; i < 3; i++) {
            tri.attr_vertex.push_back(attr_vertex[idx + i]);
            tri.attr_normal.push_back(attr_normal[idx + i]);
            tri.attr_uv.push_back(attr_uv[idx + i]);
        }

        return tri;
    }
};

struct indexed_model {
private:
    template<typename T>
    struct indexed_attrlist {
    private:
        struct indexed_attrlist_iterator :
                std::iterator<std::bidirectional_iterator_tag, T> {
            indexed_attrlist_iterator(indexed_attrlist& ia, size_t ioi) :
                ia_(ia), index_of_idx_(ioi) { }
            indexed_attrlist_iterator(const indexed_attrlist_iterator& other) :
                ia_(other.ia_), index_of_idx_(other.index_of_idx_) { }

            typedef indexed_attrlist_iterator self_type;

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

            T& operator*() { return ia_.refer_[ia_.indices[index_of_idx_]]; }

            self_type& operator=(const self_type& other) {
                index_of_idx_ = other.index_of_idx_;
            }
        private:
            indexed_attrlist& ia_;
            size_t index_of_idx_;
        };

        friend struct indexed_attrlist_iterator;

        std::vector<T>& refer_;

    public:
        std::vector<size_t> indices;

        indexed_attrlist(std::vector<T>& r) : refer_(r) { }

        indexed_attrlist_iterator begin() {
            return indexed_attrlist_iterator(*this, 0);
        }

        indexed_attrlist_iterator end() {
            return indexed_attrlist_iterator(*this, indices.size());
        }

        size_t size() const {
            return indices.size();
        }

        T& operator[](size_t i) { return refer_[indices[i]]; }
        const T& operator[](size_t i) const { return refer_[indices[i]]; }
    };

public:
    std::vector<col4> stor_vertex;
    std::vector<col3> stor_normal;
    std::vector<col3> stor_uv;

    indexed_attrlist<col4> attr_vertex;
    indexed_attrlist<col3> attr_normal;
    indexed_attrlist<col3> attr_uv;

    indexed_model() :
        attr_vertex(stor_vertex),
        attr_normal(stor_normal),
        attr_uv(stor_uv) { }

    indexed_model(const indexed_model& im) :
        stor_vertex(im.stor_vertex),
        stor_normal(im.stor_normal),
        stor_uv(im.stor_uv),
        attr_vertex(stor_vertex),
        attr_normal(stor_normal),
        attr_uv(stor_uv) { }

    indexed_model(indexed_model&& im) :
        stor_vertex(std::move(im.stor_vertex)),
        stor_normal(std::move(im.stor_normal)),
        stor_uv(std::move(im.stor_uv)),
        attr_vertex(stor_vertex),
        attr_normal(stor_normal),
        attr_uv(stor_uv) { }
};

namespace wavefront_loader_details {

struct wavefront_parse_error : std::exception {
    std::string cause;
    wavefront_parse_error(const std::string& c) : cause(c) { }
    const char* what() const noexcept override { return cause.c_str(); }
};

typedef indexed_model model_type;

inline void read_v(model_type& m, const std::string& str_v) {
    col4 v;
    std::istringstream is(str_v);

    for(int i = 0; i < 4; i++) {
        is >> v[i];
        if(is.fail()) {
            if(i == 3) {
                v[3] = 1;
                is.clear();
            } else
                throw wavefront_parse_error(
                        "Vertex is not 3D.");
        }
    }

    m.stor_vertex.push_back(v);
}

inline void read_vt(model_type& m, const std::string& str_vt) {
    col3 vt;
    std::istringstream is(str_vt);

    for(int i = 0; i < 3; i++) {
        is >> vt[i];
        if(is.fail()) {
            if(i == 2) {
                vt[2] = 1;
                is.clear();
            } else
                throw wavefront_parse_error(
                        "UV Coordinates is not 2D.");
        }
    }

    m.stor_uv.push_back(vt);
}

inline void read_vn(model_type& m, const std::string& str_vn) {
    col3 vn;
    std::istringstream is(str_vn);

    for(int i = 0; i < 3; i++) {
        is >> vn[i];
        if(is.fail()) 
            throw wavefront_parse_error(
                    "Normal vector is not 3D");
    }

    m.stor_normal.push_back(vn);
}

inline void read_f(model_type& m, const std::string& str_f) {
    // 3 choices for face element:
    //   i. v1 v2 v3 ...
    //  ii. v1//vt1 ...
    // iii. v1/vn1/vt1 ...

    int v, vn, vt;
    std::istringstream is(str_f);

    std::vector<std::tuple<int, int, int>> face;

    while(true) {
        std::string str_fe;
        is >> str_fe;

        if(str_fe.empty()) break;

        if(str_fe.find("//") != str_fe.npos) { // ii.
            int assigned = std::sscanf(str_fe.c_str(), "%d//%d", &v, &vn);
            if(assigned != 2)
                throw wavefront_parse_error("Format Error.");
            vt = v;
        } else if(str_fe.find('/') != str_fe.npos) { // iii.
            int assigned = std::sscanf(str_fe.c_str(), "%d/%d/%d", &v, &vt, &vn);
            if(assigned != 3)
                throw wavefront_parse_error("Format Error.");
        } else { // i.
            int assigned = std::sscanf(str_fe.c_str(), "%d", &v);
            if(assigned != 1)
                throw wavefront_parse_error("Format Error.");
            vn = v;
            vt = v;
        }

        face.push_back(std::make_tuple(v - 1, vn - 1, vt - 1));

        is >> std::ws;
    }

    if(face.size() > 3) {
        std::vector<std::tuple<int, int, int>> new_face;
        for(size_t i = 1; i < face.size() - 1; i++) {
            new_face.push_back(face[0]);
            new_face.push_back(face[i]);
            new_face.push_back(face[i+1]);
        }
        face = std::move(new_face);
    }

    for(auto& f : face) {
        std::tie(v, vn, vt) = f;

        m.attr_vertex.indices.push_back(v);
        m.attr_normal.indices.push_back(vn);
        m.attr_uv.indices.push_back(vt);
    }
}

}

inline indexed_model wavefront_loader(std::istream& is) {
    indexed_model m;

    std::string name;

    while(!is.eof() && !is.fail()) {
        std::string cmd;
        is >> std::ws >> cmd >> std::ws;

        if(cmd == "g") {
            if(name.empty()) {
                is >> name;
                continue;
            } else {
                is.putback(' ');
                is.putback('g');
                break; // next model
            }
        }

        std::string line;
        std::getline(is, line);

        if(cmd == "v") {
            wavefront_loader_details::read_v(m, line);
        } else if(cmd == "f") {
            wavefront_loader_details::read_f(m, line);
        } else if(cmd == "vn") {
            wavefront_loader_details::read_vn(m, line);
        } else if(cmd == "vt") {
            wavefront_loader_details::read_vt(m, line);
        }
    }

    return m;
}

}

#endif // MODEL_H_INCLUDED
