/*
 * Copyright (C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

#include <array>
#include <map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <functional>
#include <initializer_list>

namespace gcl {

namespace detail {

#define SELF (*(this->self))

// NOTE: HERE BE DRAGONS! It would probably bring about an uninitialized self,
// so watch out for nullptr exception. Need a better solution.
template<typename ExtType>
struct decorator_base
{
protected:
    typedef ExtType extended_type;
    ExtType* const self;

    decorator_base(ExtType* s = nullptr) : self(s) {
        if(self == nullptr)
            throw std::runtime_error("decorator_base is uninitialized.");
    }
};

template<typename ExtType>
struct unequal_operator_decorator : virtual decorator_base<ExtType>
{
    template<typename AnyType>
    bool operator!=(const AnyType& e) const
        { return !(SELF == e); }
};

/*
 * Total order compares by subtractions: ExtType opertor-(const ExtType& e);
 */
template<typename ExtType>
struct total_order_operator_decorator : virtual decorator_base<ExtType>
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
struct plus_equal_operator_decorator : virtual decorator_base<ExtType>
{
    template<typename DiffType>
    ExtType& operator+=(const DiffType& i)
        { SELF = SELF + i; return SELF; }
};

template<typename ExtType, typename DiffType, DiffType One = 1>
struct plus_extra_operator_decorator :
    plus_equal_operator_decorator<ExtType>
{
    ExtType operator++(int)
        { ExtType old = SELF; SELF += One; return old; }
    ExtType& operator++()
        { return (SELF += One); }
};

template<typename ExtType>
struct subs_equal_operator_decorator : virtual decorator_base<ExtType>
{
    template<typename DiffType>
    ExtType& operator-=(const DiffType& i)
        { SELF = SELF - i; return SELF; }
};

template<typename ExtType, typename DiffType, DiffType One = 1>
struct subs_extra_operator_decorator :
    subs_equal_operator_decorator<ExtType>
{
    ExtType operator--(int)
        { ExtType old = SELF; SELF -= One; return old; }
    ExtType& operator--()
        { return (SELF -= One); }
};

template<typename ExtType>
struct mult_equal_operator_decorator : virtual decorator_base<ExtType>
{
    template<typename FactorType>
    ExtType& operator*=(const FactorType& f)
        { SELF = SELF * f; return SELF; }
};

template<typename ExtType>
struct divi_equal_operator_decorator : virtual decorator_base<ExtType>
{
    template<typename FactorType>
    ExtType& operator/=(const FactorType& f)
        { SELF = SELF / f; return SELF; }
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
        decorator_base<step_iterator>(this),
        iter_(i.iter_) { }

    step_iterator(const step_iterator& i) :
        decorator_base<step_iterator>(this),
        iter_(i.iter_) { }

    step_iterator(const IterType& i) :
        decorator_base<step_iterator>(this),
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
    friend class step_iterator;
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
        decorator_base<vector_ref>(this),
        beg_(t.beg_), end_(t.end_) { }

    vector_ref(const vector_ref& t):
        decorator_base<vector_ref>(this),
        beg_(t.beg_), end_(t.end_) { }

    vector_ref(iterator const & b, iterator const & e) :
        decorator_base<vector_ref>(this),
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

    operator vector_type() {
        vector_type v;
        std::copy(begin(), end(), v.begin());

        return *this;
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

    matrix() : decorator_base<matrix>(this)
        { std::for_each(begin(), end(), [](T& e) { e = 0; }); }
    matrix(const std::initializer_list<T>& l) : decorator_base<matrix>(this)
        { std::copy(l.begin(), l.end(), begin()); }
    matrix(const matrix& m) :
        decorator_base<matrix>(this), data_(m.data_) { }
    template<typename OtherT> matrix(const matrix<OtherT, M, N>& m) :
        decorator_base<matrix>(this) { std::copy(m.begin(), m.end(), begin()); }
    matrix(matrix&& m) :
        decorator_base<matrix>(this), data_(std::move(m.data_)) { }

    matrix operator+(const matrix& m) const {
        matrix result;

        auto dst_i = result.begin(), src1_i = m.begin(), src2_i = begin();
        for(; dst_i != result.end() && src1_i != m.end() && src2_i != m.end();
                ++dst_i, ++src1_i, ++src2_i)
            *dst_i = *src1_i + *src2_i;

        return result;
    }

    template<typename Numeric> typename std::enable_if<
        std::is_arithmetic<Numeric>::value, matrix>::type
    operator*(Numeric n) const {
        matrix result;

        auto dst_i = result.begin(), src_i = begin();
        for(; dst_i != result.end() && src_i != end(); ++dst_i, ++src_i)
            *dst_i = *src_i * n;

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

        auto dst_i = result.begin(), src_i = begin();
        for(; dst_i != result.end() && src_i != end(); ++dst_i, ++src_i)
            *dst_i = *src_i / n;

        return result;
    }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }

    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend() const { return data_.cend(); }

    matrix& operator=(const matrix& m) { data_ = m.data_; return *this; }
    matrix& operator=(matrix&& m) { data_ = std::move(m.data_); return *this; }

    
    auto operator[](size_t i) -> decltype(
            matrix_subscript_<matrix, is_vector>::subscript(this, i)) {
        return matrix_subscript_<matrix, is_vector>::subscript(this, i);
    }

    auto operator[](size_t i) const -> decltype(
            matrix_subscript_<const matrix, is_vector>::subscript(this, i)) {
        return matrix_subscript_<const matrix, is_vector>::subscript(this, i);
    }

    value_type& at(size_t r, size_t c) { return data_[r * cols + c]; }
    const value_type& at(size_t r, size_t c) const { return data_[r * cols + c]; }

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
};

}

template<typename IterType, typename VecType>
using vector_ref = detail::vector_ref<IterType, VecType>;

template<typename ValueType, size_t Rows, size_t Cols>
using matrix = detail::matrix<ValueType, Rows, Cols>;

typedef matrix<double, 4, 4> mat4;
typedef matrix<double, 3, 3> mat3;
typedef matrix<double, 2, 2> mat2;
typedef matrix<double, 1, 1> mat1;
typedef matrix<double, 4, 5> mat45;
typedef matrix<double, 3, 4> mat34;
typedef matrix<double, 2, 3> mat23;

typedef matrix<double, 4, 1> col4;
typedef matrix<double, 3, 1> col3;
typedef matrix<double, 2, 1> col2;
typedef matrix<double, 1, 1> col1;

typedef matrix<double, 1, 4> row4;
typedef matrix<double, 1, 3> row3;
typedef matrix<double, 1, 2> row2;
typedef matrix<double, 1, 1> row1;

template<typename T, size_t M, size_t N>
typename std::enable_if<M == 1 || N == 1, double>::type
norm(const matrix<T, M, N>& m) {
    double result(0);
    for(const auto& e : m) result += e * e;
    return std::sqrt(result);
}

template<typename IterType, typename VecType>
double norm(const vector_ref<IterType, VecType>& v) {
    return std::sqrt(v * v);
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

}

