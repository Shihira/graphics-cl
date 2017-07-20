/*
 * Copyright (C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <array>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <functional>
#include <memory>
#include <initializer_list>

#include "traits.h"

namespace shrtool {

namespace math {

namespace detail {

template<typename T>
constexpr T mpl_max__(T t1, T t2) { return (t1 > t2) ? t1 : t2; }
template<typename T>
constexpr T mpl_min__(T t1, T t2) { return (t1 < t2) ? t1 : t2; }

#define CSELF (*(ExtType const*)(this))
#define MSELF (*(ExtType*)(this))
#define SELF CSELF

template<typename ExtType>
struct unequal_operator_decorator
{
    template<typename AnyType>
    bool operator!=(const AnyType& e) const
        { return !(SELF == e); }
};

/*
 * Total order compares by subtractions: ExtType opertor-(const ExtType& e);
 */
template<typename ExtType>
struct total_order_operator_decorator
{
    bool operator==(const ExtType& e) const
        { return SELF - e == 0; }
    bool operator!=(const ExtType& e) const
        { return SELF - e != 0; }
    bool operator< (const ExtType& e) const
        { return SELF - e <  0; }
    bool operator> (const ExtType& e) const
        { return SELF - e >  0; }
};

template<typename ExtType>
struct plus_equal_operator_decorator
{
    template<typename DiffType>
    ExtType& operator+=(const DiffType& i)
        { MSELF = SELF + i; return MSELF; }
};

template<typename ExtType, typename DiffType, DiffType One = 1>
struct plus_extra_operator_decorator :
    plus_equal_operator_decorator<ExtType>
{
    ExtType operator++(int)
        { ExtType old = SELF; MSELF += One; return old; }
    ExtType& operator++()
        { return (MSELF += One); }
};

template<typename ExtType>
struct subs_equal_operator_decorator
{
    template<typename DiffType>
    ExtType& operator-=(const DiffType& i)
        { MSELF = SELF - i; return MSELF; }
};

template<typename ExtType, typename DiffType, DiffType One = 1>
struct subs_extra_operator_decorator :
    subs_equal_operator_decorator<ExtType>
{
    ExtType operator--(int)
        { ExtType old = SELF; MSELF -= One; return old; }
    ExtType& operator--()
        { return (MSELF -= One); }
};

template<typename ExtType>
struct mult_equal_operator_decorator
{
    template<typename FactorType>
    ExtType& operator*=(const FactorType& f)
        { MSELF = SELF * f; return MSELF; }
};

template<typename ExtType>
struct divi_equal_operator_decorator
{
    template<typename FactorType>
    ExtType& operator/=(const FactorType& f)
        { MSELF = SELF / f; return MSELF; }
};

template<typename IterType, int Step = 1>
struct step_iterator :
    plus_extra_operator_decorator<step_iterator<IterType, Step>, int>,
    subs_extra_operator_decorator<step_iterator<IterType, Step>, int>,
    total_order_operator_decorator<step_iterator<IterType, Step>>,
    std::iterator<
        typename std::enable_if<std::is_same<std::random_access_iterator_tag,
            typename std::iterator_traits<IterType>::iterator_category>::value,
            std::random_access_iterator_tag>::type,
        typename std::iterator_traits<IterType>::value_type,
        typename std::iterator_traits<IterType>::difference_type,
        typename std::iterator_traits<IterType>::pointer,
        typename std::iterator_traits<IterType>::reference>
{
public:
    step_iterator operator+(int i) const
        { return step_iterator(iter_ + i * Step); }
    step_iterator operator-(int i) const
        { return operator+(-i); }
    int operator-(const step_iterator& i) const {
        if((iter_ - i.iter_) % Step != 0)
            throw std::logic_error("Fractional distance between iterators.");
        return (iter_ - i.iter_) / Step;
    }

    typename step_iterator::reference operator*() const
        { return *iter_; }
    typename step_iterator::reference operator[](int i) const
        { return *((*this) + i); }

    template<typename AnyIterType, int AnyStep>
    step_iterator(const step_iterator<AnyIterType, AnyStep>& i) :
        iter_(i.iter_) { }

    step_iterator(const step_iterator& i) :
        iter_(i.iter_) { }

    step_iterator(const IterType& i) :
        iter_(i) { }

    template<typename AnyIterType, int AnyStep>
    step_iterator& operator=(const step_iterator<AnyIterType, AnyStep>& i) {
        iter_ = i.iter_;
        return *this;
    }

    IterType& underlying() { return iter_; }

    step_iterator& operator=(const step_iterator& i) {
        iter_ = i.iter_;
        return *this;
    }

private:
    IterType iter_;

    /*
     * ref: constructor.
     * This enables initialization with all step_iterator<...> so long as
     * AnyIterType can be used to initializing OtherType.
     */
    template<typename AnyIterType, int AnyStep>
    friend struct step_iterator;
};

/*
 * Items about vector_ref's constant properties: All members of vector_ref are
 * constant so there's no chance to modify a reference just like vanilla ones.
 * A const reference is one that points to a constant matrix.
 *
 * VecType is a type who is iterable for which we can copy data
 * directly using expression std::copy(begin(), end(), v.begin())),
 * which means its size has to been determined in compilation time.
 * - Currently available STL container is std::array only.
 */

template<typename IterType, typename VecType>
struct vector_ref :
    unequal_operator_decorator   <vector_ref<IterType, VecType>>,
    plus_equal_operator_decorator<vector_ref<IterType, VecType>>,
    subs_equal_operator_decorator<vector_ref<IterType, VecType>>,
    mult_equal_operator_decorator<vector_ref<IterType, VecType>>,
    divi_equal_operator_decorator<vector_ref<IterType, VecType>>
{
    typedef IterType iterator;
    typedef VecType vector_type;

    template<typename AnyIter>
    vector_ref(const vector_ref<AnyIter, VecType>& t):
        beg_(t.beg_), end_(t.end_) { }

    vector_ref(const vector_ref& t):
        beg_(t.beg_), end_(t.end_) { }

    vector_ref(iterator const & b, iterator const & e) :
        beg_(b), end_(e) { }

    iterator begin() const { return beg_; }
    iterator end() const { return end_; }

    template<typename AnyIter>
    bool operator==(const vector_ref<AnyIter, VecType>& v) const {
        iterator s_iter = begin(); AnyIter v_iter = v.begin();
        for(; s_iter != end() && v_iter != v.end(); ++s_iter, ++v_iter)
            if(*s_iter != *v_iter) return false;

        return s_iter == end() && v_iter == v.end();
    }

    bool operator==(const vector_type& v) const {
        typedef decltype(v.begin()) vt_iterator;
        typedef vector_ref<vt_iterator, vector_type> vt_vector_ref;

        return *this == vt_vector_ref(v.begin(), v.end());
    }

    template<typename Iterable>
    vector_type operator+(const Iterable& v) const {
        vector_type r;

        auto s_iter = begin();
        auto v_iter = v.begin();
        auto r_iter = r.begin();

        for(; s_iter != end() && v_iter != v.end();
                ++s_iter, ++v_iter, ++r_iter)
            *r_iter = *s_iter + *v_iter;

        return r;
    }

    template<typename Iterable>
    vector_type operator-(const Iterable& v) const {
        vector_type r;

        auto s_iter = begin();
        auto v_iter = v.begin();
        auto r_iter = r.begin();

        for(; s_iter != end() && v_iter != v.end();
                ++s_iter, ++v_iter, ++r_iter)
            *r_iter = *s_iter - *v_iter;

        return r;
    }

    vector_type operator-() const {
        return operator*(-1);
    }

    template<typename Numeric> typename std::enable_if<
        std::is_arithmetic<Numeric>::value, vector_type>::type
    operator*(Numeric n) const {
        vector_type r;

        auto s_iter = begin();
        auto r_iter = r.begin();

        for(; s_iter != end(); ++s_iter, ++r_iter)
            *r_iter = *s_iter * n;

        return r;
    }

    template<typename Iterable> typename std::enable_if<
        !std::is_arithmetic<Iterable>::value,
        typename vector_type::value_type>::type
    operator*(const Iterable& v) const {
        typename vector_type::value_type result = 0;

        auto s_iter = begin();
        auto v_iter = v.begin();

        for(; s_iter != end() && v_iter != v.end();
                ++s_iter, ++v_iter)
            result += (*s_iter) * (*v_iter);

        return result;
    }

    template<typename Numeric> typename std::enable_if<
        std::is_arithmetic<Numeric>::value, vector_type>::type
    operator/(Numeric n) const { return operator*(1.0 / n); }

    /*
     * Disable implicit vector_type construction by doing exact template match.
     * -- Any better solutions?
     */
    template<typename T> typename std::enable_if<
        std::is_same<const T, const vector_type>::value, vector_ref>::type
    operator=(const T& v) {
        std::copy(v.begin(), v.end(), begin());
        return *this;
    }

    template<typename AnyIter>
    vector_ref operator=(const vector_ref<AnyIter, vector_type>& v) {
        std::copy(v.begin(), v.end(), begin());
        return *this;
    }

    vector_ref operator=(const vector_ref& v) {
        std::copy(v.begin(), v.end(), begin());
        return *this;
    }

    operator vector_type() const {
        vector_type v;
        std::copy(begin(), end(), v.begin());

        return v;
    }

    auto operator[](size_t i) -> decltype(*begin()) {
        return *(begin() + i);
    }

protected:
    iterator const beg_;
    iterator const end_;

    template<typename AnyIter, typename AnyVec>
    friend struct vector_ref;
};

/*
 * class matrix represents a matrix sized to MxN (M rows by N cols).
 */


/*
 * matrix_subscript implements a trick: when matrix is vector, subscript
 * operation returns a value instead of a vector reference;
 * **NOTE**: Never use operator[] while implementing matrix algorithms, because
 * it's no more than a tool for clients' convenience and is non-systematic.
 */
template<typename ExtType, bool IsVec>
struct matrix_subscript_ { };

template<typename ExtType>
struct matrix_subscript_<ExtType, false>
{
    typedef typename std::conditional<
        std::is_const<ExtType>::value,
        typename ExtType::const_row_ref,
        typename ExtType::row_ref>::type ret_type;

    static ret_type subscript(ExtType* self, size_t i) {
        auto beg = self->begin() + i * self->cols;
        return ret_type(
                typename ret_type::iterator(beg),
                typename ret_type::iterator(beg + self->cols)
            );
    }
};

template<typename ExtType>
struct matrix_subscript_<ExtType, true>
{
    typedef typename std::conditional<
        std::is_const<ExtType>::value,
        const typename ExtType::value_type&,
        typename ExtType::value_type&>::type ret_type;

    static ret_type subscript(ExtType* self, size_t i) {
        return *(self->begin() + i);
    }
};

template<typename T, size_t M, size_t N>
struct matrix :
    unequal_operator_decorator   <matrix<T, M, N>>,
    plus_equal_operator_decorator<matrix<T, M, N>>,
    subs_equal_operator_decorator<matrix<T, M, N>>,
    mult_equal_operator_decorator<matrix<T, M, N>>,
    divi_equal_operator_decorator<matrix<T, M, N>>
{
private:
    typedef std::array<T, M * N> container_type;
    //typedef std::unique_ptr<container_type> __pointer;
    container_type data_;

public:
    typedef T value_type;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

    typedef vector_ref<step_iterator<iterator, N>, matrix<T, M, 1>> col_ref;
    typedef vector_ref<step_iterator<iterator>, matrix<T, 1, N>> row_ref;
    typedef vector_ref<step_iterator<const_iterator, N>, matrix<T, M, 1>> const_col_ref;
    typedef vector_ref<step_iterator<const_iterator>, matrix<T, 1, N>> const_row_ref;

    static constexpr size_t rows = M;
    static constexpr size_t cols = N;
    static constexpr bool is_vector = M == 1 || N == 1;

    matrix() //: data_(new container_type)
        { std::fill(begin(), end(), 0); }
    matrix(const std::initializer_list<T>& l)
        //: data_(new container_type)
        { std::copy(l.begin(), l.end(), begin()); }
    matrix(const matrix& m)
        //: data_(new container_type)
        { std::copy(m.begin(), m.end(), begin()); }
    template<typename OtherT>
    matrix(const matrix<OtherT, M, N>& m)
        //: data_(new container_type)
        { std::copy(m.begin(), m.end(), begin()); }
    //matrix(matrix&& m) : data_(std::move(m.data_)) { }
    template<typename OtherT, size_t M_, size_t N_>
    matrix(const matrix<OtherT, M_, N_>& mat) {
        T* dst = data();
        const OtherT* src = mat.data();
        for(size_t m = 0; m < mpl_min__(M_, M); m++) {
            for(size_t n = 0; n < mpl_min__(N_, N); n++)
                dst[n] = src[n];
            dst += N;
            src += N_;
        }
    }

    matrix operator+(const matrix& m) const {
        matrix result;

        auto dst = &(*result.begin());
        auto src1 = &(*m.begin());
        auto src2 = &(*begin());

        for(int i = 0; i < M * N; i++)
            dst[i] = src1[i] + src2[i];

        return result;
    }

    value_type* data() { return data_.data(); }
    const value_type* data() const { return data_.data(); }

    template<typename Numeric> typename std::enable_if<
        std::is_arithmetic<Numeric>::value, matrix>::type
    operator*(Numeric n) const {
        matrix result;

        auto dst = &*result.begin();
        auto src = &*begin();

        for(int i = 0; i < M * N; i++)
            dst[i] = src[i] * n;

        return result;
    }

    template<size_t K>
    matrix<T, M, K> operator*(const matrix<T, N, K>& mul) const {
        matrix<T, M, K> mat;

        for(size_t m = 0; m < M; m++)
        for(size_t k = 0; k < K; k++)
            mat.at(m, k) = row(m) * mul.col(k);

        return mat;
    }

    matrix operator-(const matrix& m) const { return operator+(-m); }
    matrix operator-() const { return operator*(-1); }

    template<typename Numeric> typename std::enable_if<
        std::is_arithmetic<Numeric>::value, matrix>::type
    operator/(Numeric n) const {
        matrix result;

        auto dst = &*result.begin();
        auto src = &*begin();

        for(int i = 0; i < M * N; i++)
            dst[i] = src[i] / n;

        return result;
    }

    /*
     * row-major order iterator
     */
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }

    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend() const { return data_.cend(); }

    matrix& operator=(const matrix& m) {
        std::copy(m.begin(), m.end(), begin());
        return *this;
    }

    matrix& operator=(matrix&& m) {
        data_ = std::move(m.data_);
        return *this;
    }

    
    auto operator[](size_t i) -> decltype(
            matrix_subscript_<matrix, is_vector>::subscript(this, i)) {
        return matrix_subscript_<matrix, is_vector>::subscript(this, i);
    }

    auto operator[](size_t i) const -> decltype(
            matrix_subscript_<const matrix, is_vector>::subscript(this, i)) {
        return matrix_subscript_<const matrix, is_vector>::subscript(this, i);
    }

    value_type& at(size_t r, size_t c) { return data()[r * cols + c]; }
    const value_type& at(size_t r, size_t c) const { return data()[r * cols + c]; }

    col_ref col(size_t i) {
        return col_ref(
                typename col_ref::iterator(begin() + i),
                typename col_ref::iterator(end() + i)
            );
    }

    const_col_ref col(size_t i) const {
        return const_col_ref(
                typename const_col_ref::iterator(begin() + i),
                typename const_col_ref::iterator(end() + i)
            );
    }

    row_ref row(size_t i) {
        auto beg = begin() + i * cols;
        return row_ref(
                typename row_ref::iterator(beg),
                typename row_ref::iterator(beg + cols)
            );
    }

    const_row_ref row(size_t i) const {
        auto beg = begin() + i * cols;
        return const_row_ref(
                typename const_row_ref::iterator(beg),
                typename const_row_ref::iterator(beg + cols)
            );
    }

    bool operator==(const matrix& m) const {
        auto s_i = begin(), m_i = m.begin();
        for(; s_i != end() && m_i != m.end(); ++s_i, ++m_i)
            if(*s_i != *m_i) return false;
        return true;
    }

    bool close(const matrix& m, value_type bias) const {
        auto s_i = begin(), m_i = m.begin();
        for(; s_i != end() && m_i != m.end(); ++s_i, ++m_i) {
            value_type res = *s_i - *m_i;
            if(res > value_type(0) && res > bias) return false;
            if(res < value_type(0) && res < -bias) return false;
        }
        return true;
    }
};

template<typename T>
struct is_matrix : std::false_type { };
template<typename T, size_t M, size_t N>
struct is_matrix<matrix<T, M, N>> : std::true_type { };

template<typename T, size_t M, size_t N>
std::ostream& operator<<(std::ostream& s, const matrix<T, M, N>& mat) {
    for(size_t m = 0; m < M; m++) {
        for(size_t n = 0; n < N; n++)
            s <<
                (n == 0 ? (m == 0 ? "[ " : "  ") : "") <<
                mat.at(m, n) <<
                (n == N - 1 ? (m == M - 1 ? " ]" : ";\n") : ",\t");
    }
    return s;
}

template<typename T, size_t M>
std::ostream& operator<<(std::ostream& s, const matrix<T, M, 1>& mat) {
    for(size_t m = 0; m < M; m++) {
        s <<
            (m == 0 ? "[ " : "") <<
            mat.at(m, 0) <<
            (m == M - 1 ? " ]ᵀ" : ",\t");
    }
    return s;
}

template<typename IterType, typename VecType>
std::ostream& operator<<(std::ostream& s,
        const vector_ref<IterType, VecType>& v) {
    return operator<<(s, VecType(v));
}

#undef CSELF
#undef SSELF

}

static constexpr double PI = 3.141592653589793;

template<typename IterType, typename VecType>
using vector_ref = detail::vector_ref<IterType, VecType>;

template<typename ValueType, size_t rows, size_t cols>
using matrix = detail::matrix<ValueType, rows, cols>;

typedef matrix<double, 4, 4> mat4;
typedef matrix<double, 3, 3> mat3;
typedef matrix<double, 2, 2> mat2;
typedef matrix<double, 1, 1> mat1;
typedef matrix<double, 4, 5> mat45;
typedef matrix<double, 3, 4> mat34;
typedef matrix<double, 2, 3> mat23;

typedef matrix<float, 4, 4> fmat4;
typedef matrix<float, 3, 3> fmat3;
typedef matrix<float, 2, 2> fmat2;
typedef matrix<float, 1, 1> fmat1;
typedef matrix<float, 4, 5> fmat45;
typedef matrix<float, 3, 4> fmat34;
typedef matrix<float, 2, 3> fmat23;

template<typename T, size_t M>
using col = matrix<T, M, 1>;
template<typename T, size_t N>
using row = matrix<T, 1, N>;

typedef col<double, 4> col4;
typedef col<double, 3> col3;
typedef col<double, 2> col2;
typedef col<double, 1> col1;

typedef row<double, 4> row4;
typedef row<double, 3> row3;
typedef row<double, 2> row2;
typedef row<double, 1> row1;

typedef col<float, 4> fcol4;
typedef col<float, 3> fcol3;
typedef col<float, 2> fcol2;
typedef col<float, 1> fcol1;

typedef row<float, 4> frow4;
typedef row<float, 3> frow3;
typedef row<float, 2> frow2;
typedef row<float, 1> frow1;

typedef col<int32_t, 4> icol4;
typedef col<int32_t, 3> icol3;
typedef col<int32_t, 2> icol2;
typedef col<int32_t, 1> icol1;

typedef row<int32_t, 4> irow4;
typedef row<int32_t, 3> irow3;
typedef row<int32_t, 2> irow2;
typedef row<int32_t, 1> irow1;

template<typename T, size_t M, size_t N>
matrix<T, N, M> transpose(const matrix<T, M, N>& m_) {
    matrix<T, N, M> new_m;

    for(size_t m = 0; m < M; m++)
        for(size_t n = 0; n < N; n++) {
            new_m.at(n, m) = m_.at(m, n);
        }

    return new_m;
}


template<typename T, size_t M, size_t N>
typename std::enable_if<M == N, T>::type
det(const matrix<T, M, N>& m_) {
    matrix<T, M, N> mat = m_;

    T res(1), times(1);

    for(size_t n = 0; n < N; n++) {
        if(mat.at(n, n) == 0) {
            for(size_t k = n + 1; k < M; k++)
                if(mat.at(k, n) != 0) {
                    mat.row(n) += mat.row(k);
                    break;
                }
            // singular matrix
            if(mat[n][n] == 0) return 0;
        }

        T Mnn(mat.at(n, n));
        res *= Mnn;

        for(size_t m = n + 1; m < M; m++) {
            T Mmn(mat.at(m, n));

            times *= Mnn;
            mat.row(m) *= Mnn;

            mat.row(m) -= mat.row(n) * Mmn;
        }
    }

    return res / times;
}

template<typename T, size_t M>
const matrix<T, M, M>
inverse(const matrix<T, M, M>& m_) {
    T det_val(det(m_));

    if(!det_val)
        throw std::logic_error("Attempted to find"
            "inversion of a singular matrix");

    matrix<T, M, M> adjugate;

    for(size_t m = 0; m < M; m++)
        for(size_t n = 0; n < M; n++) {
            matrix<T, M-1, M-1> minor;

            for(size_t i = 0, mnr_i = 0; i < M; i++) {
                if(i == m) continue;
                for(size_t j = 0, mnr_j = 0; j < M; j++) {
                    if(j == n) continue;

                    minor.at(mnr_i, mnr_j) = m_.at(i, j);

                    mnr_j += 1;
                }
                mnr_i += 1;
            }
            adjugate.at(m, n) = det(minor) *
                (((m + n) % 2) ? -1 : 1);
        }

    return transpose(adjugate) / det_val;
}

template<typename T, size_t M, size_t N, size_t P, size_t Q>
typename std::enable_if<
    detail::mpl_min__(M, N) == 1 && detail::mpl_min__(P, Q) == 1 &&
    detail::mpl_max__(M, N) == detail::mpl_max__(P, Q), T>::type
dot(const matrix<T, M, N>& v1, const matrix<T, P, Q>& v2) {
    T sum(0);
    const T* src1 = v1.data();
    const T* src2 = v2.data();

    for(size_t i = 0; i < M * N; i++)
        sum += src1[i] * src2[i];

    return sum;
}

template<typename T, size_t M, size_t N>
typename std::enable_if<M == 1 || N == 1, double>::type
norm(const matrix<T, M, N>& m) {
    return std::sqrt(dot(m, m));
}

template<typename IterType, typename VecType>
double norm(const vector_ref<IterType, VecType>& v) {
    return std::sqrt(v * v);
}

template<typename T, size_t M, size_t N, size_t P, size_t Q>
typename std::enable_if<
    detail::mpl_min__(M, N) == 1 && detail::mpl_min__(P, Q) &&
    detail::mpl_max__(M, N) == 3 && detail::mpl_max__(P, Q) == 3,
    col<T, 3>>::type
cross(const matrix<T, M, N>& v1, const matrix<T, P, Q>& v2) {
    col3 res;
    res[0] = det(mat2{
            v1[1], v1[2],
            v2[1], v2[2],
        });
    res[1] = -det(mat2{
            v1[0], v1[2],
            v2[0], v2[2],
        });
    res[2] = det(mat2{
            v1[0], v1[1],
            v2[0], v2[1],
        });
    return res;
}

template<typename T>
typename std::enable_if<std::is_scalar<T>::value, T>::type
clamp(T v, T min_v, T max_v) {
    return v < min_v ? min_v : v > max_v ? max_v : v;
}

////////////////////////////////////////////////////////////////////////////////
// dynmatrix: not useful c++ natively, mostly used in reflection

template<typename T>
struct dynmatrix {
    typedef T value_type;

    dynmatrix() { }
    dynmatrix(size_t rows, size_t cols) {
        assign(rows, cols);
    }
    dynmatrix(size_t rows, size_t cols,
            const std::initializer_list<value_type>& d) {
        assign(rows, cols);
        std::copy(d.begin(), d.end(), data_);
    }
    dynmatrix(const dynmatrix& d) {
        assign(d.rows_, d.cols_);
        std::copy(d.data_, d.data_ + d.elem_count(), data_);
    }
    dynmatrix(dynmatrix&& d) :
        rows_(d.rows_), cols_(d.cols_) {
        std::swap(d.data_, data_);
        std::swap(d.is_agent_, is_agent_);
    }
    template<size_t M, size_t N>
    dynmatrix(const matrix<T, M, N>& m) {
        assign(M, N);
        std::copy(m.begin(), m.end(), data_);
    }

    void assign(size_t rows, size_t cols) {
        if(data_) delete[] data_;
        cols_ = cols;
        rows_ = rows;
        data_ = new value_type[cols * rows];
    }

    ~dynmatrix() {
        if(data_ && !is_agent_) delete[] data_;
    }

    value_type* data() { return data_; }
    const value_type* data() const { return data_; }
    value_type& at(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }
    const value_type& at(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    operator bool() const {
        return data_;
    }

    static dynmatrix agent(size_t r, size_t c, value_type* v) {
        dynmatrix mat;
        mat.rows_ = r; mat.cols_ = c; mat.data_ = v;
        mat.is_agent_ = true;

        return std::move(mat);
    }

    template<size_t M, size_t N>
    static dynmatrix agent(matrix<value_type, M, N>& m) {
        return agent(m.rows, m.cols, m.data());
    }

    template<size_t M, size_t N>
    operator matrix<value_type, M, N>() const {
        matrix<value_type, M, N> m;
        std::copy(data_, data_ + elem_count(), m.begin());
        return std::move(m);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t elem_count() const { return cols_ * rows_; }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;

    value_type* data_ = nullptr;
    bool is_agent_ = false;
};

typedef dynmatrix<float> fxmat;
typedef dynmatrix<double> dxmat;

////////////////////////////////////////////////////////////////////////////////

namespace tf {

typedef enum { xOy, yOz, zOx } plane;

template<typename T = double>
inline matrix<T, 4, 4> diagonal(col<T, 4> diag)
{
    return matrix<T, 4, 4> {
        diag[0], 0, 0, 0,
        0, diag[1], 0, 0,
        0, 0, diag[2], 0,
        0, 0, 0, diag[3],
    };
}

template<typename T = double>
inline matrix<T, 4, 4> rotate(double a, plane p)
{
    if(p == xOy)
        return matrix<T, 4, 4> {
            cos(a),-sin(a), 0, 0,
            sin(a), cos(a), 0, 0,
            0,      0,      1, 0,
            0,      0,      0, 1,
        };
    else if(p == yOz)
        return matrix<T, 4, 4> {
            1, 0,      0,      0,
            0, cos(a), sin(a), 0,
            0,-sin(a), cos(a), 0,
            0, 0,      0,      1,
        };
    else
        return matrix<T, 4, 4> {
            cos(a), 0,-sin(a), 0,
            0,      1, 0,      0,
            sin(a), 0, cos(a), 0,
            0,      0, 0,      1,
        };
}

template<typename T = double>
inline matrix<T, 4, 4> translate(col<T, 4> t)
{
    return matrix<T, 4, 4> {
        t[3], 0,    0,    t[0],
        0,    t[3], 0,    t[1],
        0,    0,    t[3], t[2],
        0,    0,    0,    t[3],
    };
}

template<typename T = double>
inline matrix<T, 4, 4> translate(col<T, 3> t)
{
    return matrix<T, 4, 4> {
        1, 0,    0,    t[0],
        0,    1, 0,    t[1],
        0,    0,    1, t[2],
        0,    0,    0,    1,
    };
}

template<typename T = double>
inline matrix<T, 4, 4> scale(T x, T y, T z)
    { return diagonal({x, y, z, 1}); }

template<typename T = double>
inline matrix<T, 4, 4> identity()
    { return diagonal({1, 1, 1, 1}); }

template<typename T = double>
inline matrix<T, 4, 4> perspective(double fov, double wh, double zn, double zf)
{
    double f = 1 / tan(fov);
    double c = zn - zf;
    return matrix<T, 4, 4> {
        f / wh, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zn + zf) / c, 2 * zn * zf / c,
        0, 0, -1, 0
    };
}

template<typename T = double>
inline matrix<T, 4, 4> orthographic(
        double l, double r,
        double t, double b,
        double n, double f)
{
    double  r_l = r - l,
            t_b = t - b,
            f_n = f - n;
    return matrix<T, 4, 4> {
        2/r_l, 0, 0, -(r+l)/r_l,
        0, 2/t_b, 0, -(t+b)/t_b,
        0, 0, 2/f_n, -(f+n)/f_n,
        0, 0, 0, 1,
    };
}

} // tf

} // math

////////////////////////////////////////////////////////////////////////////////
// traits

template<typename T, size_t M, size_t N>
struct item_trait<math::matrix<T, M, N>>
{
    typedef float value_type;
    static constexpr size_t size() {
        return M * N * sizeof(value_type);
    }
    static constexpr size_t align() {
        return item_trait<math::col<value_type, M>>::align();
    }

    static void copy(const math::matrix<T, M, N>& m, value_type* buf) {
        for(size_t n = 0; n < N; ++n, buf += M) {
            const auto& c = m.col(n);
            std::copy(c.begin(), c.end(), buf);
        }
    }

    static const char* glsl_type_name() {
        static const char name_[] = {
            'm', 'a', 't', M + '0',
            M == N ? '\0' : 'x', N + '0', '\0' };
        return name_;
    }
};

template<typename T, size_t M>
struct item_trait<math::matrix<T, M, 1>>
{
    typedef T value_type;
    static constexpr size_t size() {
        return M * sizeof(value_type);
    }
    static constexpr size_t align() {
        return (M < 3 ? M : 4) * sizeof(value_type);
    }

    static void copy(const math::col<T, M>& c, value_type* buf) {
        std::copy(c.begin(), c.end(), buf);
    }

    static const char* glsl_type_name() {
        static const char name_[] = {
            std::is_same<value_type, uint8_t>::value ? 'b' :
            std::is_same<value_type, int>::value ? 'i' :
            std::is_same<value_type, double>::value ? 'd' : '\0',
            'v', 'e', 'c', M + '0', '\0' };
        static const char* name_p = name_[0] ? name_ : name_ + 1;

        return name_p;
    }
};

template<typename T>
struct item_trait<math::dynmatrix<T>>
{
    typedef T value_type;
    static size_t size(const math::dynmatrix<T>& m) {
        return sizeof(value_type) * m.elem_count();
    }
    static size_t align(const math::dynmatrix<T>& m) {
        size_t r = m.rows();
        return sizeof(value_type) * (r < 3 ? r : 4);
    }
    static void copy(const math::dynmatrix<T>& m, value_type* buf) {
        for(size_t i = 0; i < m.cols(); i++) {
            for(size_t j = 0; j < m.rows(); j++) {
                *(buf++) = m.at(j, i);
            }
        }
    }
    static std::string glsl_type_name(const math::dynmatrix<T>& m) {
        const char* tc =
            std::is_same<value_type, uint8_t>::value ? "b" :
            std::is_same<value_type, int>::value ? "i" :
            std::is_same<value_type, double>::value ? "d" : "";

        if(m.cols() == 1) {
            return std::string(tc) +
                std::string("vec") + std::to_string(m.rows());
        } else {
            return std::string(tc) +
                std::string("mat") + std::to_string(m.rows()) +
                "x" + std::to_string(m.cols());
        }
    }
};

} // shrtool

#endif // MATRIX_H_INCLUDED
