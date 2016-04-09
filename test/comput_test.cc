// cflags: -lOpenCL

#include "../include/test.h"
#include "../include/comput.h"

using namespace std;
using namespace gcl;

def_test_case(check_devices) {
    std::vector<platform> ps = platform::get();

    for(platform& p : ps) {
        std::vector<device> ds = device::get(p);
        if(ds.empty()) continue;
        for(device& d : ds) {
            ctest << p.getInfo<CL_PLATFORM_VENDOR>() <<
                " " << d.getInfo<CL_DEVICE_VERSION>() <<
                " [" << d.getInfo<CL_DEVICE_NAME>() << "]" << endl;
        }
    }
}

struct context_fixture {
    context ctxt;
    context_guard cg;

    context get_context() {
        std::vector<platform> ps = platform::get();
        std::vector<device> ds = device::get(ps);

        return context(ds.back());
    }

    context_fixture() :
        ctxt(get_context()),
        cg(ctxt)
    { }
};

def_test_case_with_fixture(kernel_test_assingment, context_fixture) {
    program prg = compile(R"EOF(
    kernel void fun(global uint* buf)
    {
        uint id = get_global_id(0);
        buf[id] = id;
    }
    )EOF", "-cl-kernel-arg-info");

    kernel krn(prg, "fun");

    buffer<cl_uint> s(100, host_map);

    krn.set_buffer(0, s);

    promise() <<
        run(krn, 100) <<
        pull(s) << unpull(s)
        << wait_until_done;

    for(size_t i = 0; i < 100; i++)
        assert_true(i == s[i]);
}

def_test_case_with_fixture(kernel_test_reflection, context_fixture) {
    program prg = compile(R"EOF(
    typedef float4 pos_t;
    kernel void fun(
            global uint* var1,
            global float4 * var2,
            global pos_t * var3) {
    }
    )EOF","-cl-kernel-arg-info");

    kernel krn(prg, "fun");

    assert_true(krn.getInfo<CL_KERNEL_NUM_ARGS>() == 3);
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_NAME>(0) == "var1");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(0) == "uint*");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_NAME>(1) == "var2");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(1) == "float4*");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_NAME>(2) == "var3");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(2) == "pos_t*");
}

def_test_case_with_fixture(kernel_test_sum_up, context_fixture) {
    program prg = compile(R"EOF(
    kernel void fun(global uint* s, global uint* res)
    {
        size_t id = get_global_id(0);
        atomic_add(res, s[id] + id);
    }
    )EOF");

    kernel krn(prg, "fun");

    buffer<cl_uint> s(500, 1);
    buffer<cl_uint> r{200};

    krn.set_buffer(0, s);
    krn.set_buffer(1, r);

    promise() <<
        push(s) << unpush(s) <<
        push(r) << unpush(r) <<
        run(krn, 500) <<
        pull(r) << unpull(r) <<
        pull(s) << unpull(s) <<
        wait_until_done;

    assert_true(r[0] == (500 + 200 + 499 * 500 / 2));
    assert_true(krn.getInfo<CL_KERNEL_NUM_ARGS>() == 2);
}

int main()
{
    test_case::test_all(true);
    cout << test_case::log().str();

    return test_case::status() ? 0 : -1;
}

