/*
 * Copyright (C) Shihira Fung, 2016 <fengzhiping@hotmail.com>
 */

#include "../include/matrix.h"
#include "../include/test.h"

using namespace shr_mat;
using namespace std;

template<int size>
array<int, size> int_range(int first, int last, int step = 1) {
    array<int, size> arr;

    for(int i = 0, n = first; n <= last && i < size; ++i, n += step)
        arr[i] = n;

    return arr;
}

template<typename T1, typename T2, typename Func>
void for_each_2(T1& t1, T2& t2, Func f) {
    auto t1_i = t1.begin();
    auto t2_i = t2.begin();

    for(; t1_i != t1.end() && t2_i != t2.end(); t1_i++, t2_i++)
        f(*t1_i, *t2_i);
}

template<typename IterType>
struct iterable_wrapper {
    IterType beg_;
    IterType end_;

    IterType begin() { return beg_; }
    IterType end() { return end_; }
};

////////////////////////////////////////////////////////////////////////////////
// step iterator tests

struct si_fixture {
    template<typename IterType, size_t Step>
    using step_iterator = shr_mat::detail::step_iterator<IterType, Step>;

    static constexpr int step = 3;
    static constexpr int size = 10;
    static constexpr int max_val = step * (size - 1);

    typedef array<int, size> vector_type;

    typedef step_iterator<vector_type::iterator, step> iterator;
    typedef step_iterator<vector_type::const_iterator, step> const_iterator;

    array<int, size * step> int_list;
    vector_type expected;

    si_fixture() :
        int_list(int_range<size * step>(0, max_val)),
        expected(int_range<size>(0, max_val, step))
    { }

    iterator begin()
        { return iterator(int_list.begin()); }
    iterator end()
        { return iterator(int_list.begin() + step * size); }
};

def_test_case_with_fixture(si_iterable, si_fixture)
{
    for_each_2(*this, expected, equal_to<int>());
}

def_test_case_with_fixture(si_const_assignment, si_fixture)
{
    auto const_this = iterable_wrapper<const_iterator>
        { begin(), end() };

    for_each_2(const_this, expected, equal_to<int>());
}

////////////////////////////////////////////////////////////////////////////////
// vector reference tests

struct vr_fixture {
    template<typename IterType, size_t Step>
    using step_iterator = shr_mat::detail::step_iterator<IterType, Step>;

    static constexpr int step = 3;
    static constexpr int size = 10;
    static constexpr int max_val = step * (size - 1);

    typedef array<int, size> vector_type;
    typedef vector_ref<vector_type::const_iterator, vector_type> const_vr;
    typedef vector_ref<vector_type::iterator, vector_type> mutable_vr;

    typedef step_iterator<array<int, size * step>::iterator, step> org_iter;
    typedef step_iterator<array<int, size * step>::const_iterator, step> const_org_iter;

    array<int, size * step> org;
    const vector_type expected;
    vector_ref<org_iter, vector_type> org_vr;

    template<typename T>
    const_vr make_const_vr(const T& t) {
        return const_vr(t.begin(), t.end());
    }

    template<typename T>
    mutable_vr make_mutable_vr(T& t) {
        return mutable_vr(t.begin(), t.end());
    }

    vr_fixture() :
        org(int_range<size * step>(0, max_val)),
        expected(int_range<10>(0, max_val, step)),
        org_vr(org_iter(org.begin()), org_iter(org.begin()) + size) { }
};

def_test_case_with_fixture(vr_equality, vr_fixture)
{
    const_vr cmp_vr = make_const_vr(expected);

    assert_true(cmp_vr == org_vr);
    assert_true(org_vr == expected);
    assert_false(org_vr != cmp_vr);
}

def_test_case_with_fixture(vr_const_assignment, vr_fixture)
{
    const_vr cmp_vr = make_const_vr(expected);
    vector_ref<const_org_iter, vector_type> const_org_vr = org_vr;

    assert_true(const_org_vr == cmp_vr);
    assert_true(const_org_vr == org_vr);
}

def_test_case_with_fixture(vr_evaluate, vr_fixture)
{
    const vector_type
        dif_list {  1,  2,  3,  4,  5,  5,  4,  3,  2,  1 },
        sub_list { -1,  1,  3,  5,  7, 10, 14, 18, 22, 26 },
        sum_list {  1,  5,  9, 13, 17, 20, 22, 24, 26, 28 },
        mul_list {  0,  6, 12, 18, 24, 30, 36, 42, 48, 54 },
        div_list {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9 };

    const_vr
        dif_vr = make_const_vr(dif_list),
        exp_vr = make_const_vr(expected),
        sub_vr = make_const_vr(sub_list);

    assert_true(org_vr == exp_vr);
    assert_true(org_vr + dif_vr == sum_list);
    assert_true(org_vr - dif_vr == sub_list);
    assert_true(org_vr * 2      == mul_list);
    assert_true(org_vr / 3      == div_list);

    assert_true(sub_vr == org_vr - dif_vr);
    assert_true(org_vr * dif_vr == 405);
}

def_test_case_with_fixture(vr_modify_origin, vr_fixture)
{
    const vector_type
        dif_list {  1,  2,  3,  4,  5,  5,  4,  3,  2,  1 },
        sum_list {  1,  5,  9, 13, 17, 20, 22, 24, 26, 28 },
        mul_list {  0,  6, 12, 18, 24, 30, 36, 42, 48, 54 },
        div_list {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9 };

    const_vr
        dif_vr = make_const_vr(dif_list),
        exp_vr = make_const_vr(expected);

    org_vr += dif_vr;
    assert_true(org_vr == sum_list);

    org_vr -= dif_vr;
    assert_true(org_vr == exp_vr);

    org_vr *= 2;
    assert_true(org_vr == mul_list);

    org_vr = exp_vr;
    assert_true(org_vr == exp_vr);

    org_vr /= 3;
    assert_true(org_vr == div_list);

    org_vr = expected;
    assert_true(org_vr == exp_vr);
}

////////////////////////////////////////////////////////////////////////////////
// matrix tests

struct mat_fixture {
    mat34 nsqr;
    const mat4 sqr;
    mat4 sqr2;
    row4 row;
    const col3 col;

    mat_fixture() :
        nsqr {
            6,  22, 14, 15,
            24, 15, 22, 8,
            29, 9,  26, 30,
        },
        sqr {
            17, 15, 5,  18,
            27, 10, 17, 27,
            22, 12, 13, 10,
            21, 8,  7,  19,
        },
        sqr2 {
            6,  22, 14, 15,
            24, 15, 22, 8,
            29, 9,  26, 30,
            21, 8,  7,  19,
        },
        row {
            13, 25,  8, 10
        },
        col {
            24, 2,  19
        } { }
};

def_test_case_with_fixture(mat_subscript, mat_fixture) {
    assert_true(nsqr[2][1] == 9);
    assert_true(sqr[0][3] == 18);
    assert_true(row[2] == 8);
    assert_true(col[1] == 2);

    assert_true(sqr.at(0, 3) == 18);
    assert_true(sqr.row(0)[3] == 18);
    assert_true(sqr.col(3)[0] == 18);

    assert_true(sqr.row(3) * sqr.col(0) == 1126);
}

def_test_case_with_fixture(mat_multiply, mat_fixture) {
    assert_true(nsqr * sqr == mat34({
        1319,        598,        691,       1127,
        1465,        838,        717,       1209,
        1938,       1077,        846,       1595,
    }));

    assert_true(sqr2 * sqr == mat4({
        1319,        598,        691,       1127,
        1465,        838,        717,       1209,
        1938,       1077,        846,       1595,
        1126,        631,        465,       1025,
    }));

    sqr2 *= sqr;
    assert_true(sqr2 == mat4({
        1319,        598,        691,       1127,
        1465,        838,        717,       1209,
        1938,       1077,        846,       1595,
        1126,        631,        465,       1025,
    }));
}

def_test_case_with_fixture(mat_vector_modify_origin, mat_fixture) {
    sqr2[1] -= sqr2[0] * 4;
    sqr2[2] += -sqr2[0] * (29. / 6);
    sqr2[3] -= sqr2[0] * (21. / 6);

    for(size_t i = 1; i < 4; i++)
        assert_float_equal(sqr2[i][0], 0);
}

def_test_case_with_fixture(mat_vector_operations, mat_fixture) {
    assert_float_equal(det(sqr), 18564);
    assert_float_equal(det(sqr2), -141055);
    assert_true(det(matrix<long long, 4, 4>(sqr)) == 18564);

    assert_float_equal(norm(col), sqrt(941));
    assert_float_equal(norm(sqr.col(0)), sqrt(1943));
}

int main()
{
    test_case::test_all(true);
}

