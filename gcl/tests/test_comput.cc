// cflags: -lOpenCL

#include "promise.h"
#include "common/unit_test.h"

using namespace std;
using namespace gcl;

TEST_CASE(check_devices) {
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

struct sum_up_program_fixture {
    program prg;
    kernel krn;
    buffer<cl_uint> s;
    buffer<cl_uint> r;

    sum_up_program_fixture() :
        prg(compile(R"EOF(
        kernel void fun(global uint* s, global uint* r)
        {
            size_t id = get_global_id(0);
            atomic_add(r, s[id] + id);
        }
        )EOF")),
        krn(prg, "fun"),
        s(500, host_map),
        r(1, host_map) { }

    void set_buffer() {
        krn.set_buffer(0, s);
        krn.set_buffer(1, r);
    }
};

TEST_CASE(kernel_test_assingment) {
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

    try {
        promise() <<
            run(krn, 100) <<
            pull(s) <<
            wait_until_done;
    } catch(cl::Error e) {
        ctest << e.err() << endl;
    }

    for(size_t i = 0; i < 100; i++)
        assert_true(i == s[i]);
}

TEST_CASE(kernel_test_reflection) {
    program prg = compile(R"EOF(
    typedef float4 pos_t;
    kernel void fun(
            global uint* var1,
            global float4 * var2,
            global pos_t * var3) {
        *var2 = (float4)(*var1, *var1, 0, 0);
        *var3 = *var2;
        *var1 = 1;
    }
    kernel void fun2(
            global int* var4
        ) {
        *var4 = 1;
    }
    )EOF","-cl-kernel-arg-info");

    kernel krn(prg, "fun");

    assert_true(prg.getInfo<CL_PROGRAM_KERNEL_NAMES>() == "fun;fun2");
    assert_true(krn.getInfo<CL_KERNEL_NUM_ARGS>() == 3);
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_NAME>(0) == "var1");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(0) == "uint*");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_NAME>(1) == "var2");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(1) == "float4*");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_NAME>(2) == "var3");
    assert_true(krn.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(2) == "pos_t*");
}

TEST_CASE_FIXTURE(kernel_test_sum_up, sum_up_program_fixture) {
    set_buffer();

    fill(s.begin(), s.end(), 1);
    r[0] = 200;

    promise() <<
        push(s) <<
        push(r) <<
        run(krn, 500) <<
        pull(r) <<
        pull(s) <<
        wait_until_done;

    assert_true(r[0] == (500 + 200 + 499 * 500 / 2));
    assert_true(krn.getInfo<CL_KERNEL_NUM_ARGS>() == 2);
}

TEST_CASE_FIXTURE(command_queue_run_lambda, sum_up_program_fixture) {
    set_buffer();

    r[0] = 0;

    bool first_lambda_passed = false;
    bool second_lambda_passed = false;

    promise() <<
        fill(s, 1U) <<
        push(r) <<
        run(krn, 500) <<
        pull(r) <<
        call([&] {
            first_lambda_passed = (r[0] == (500 + 499 * 500 / 2));
        }) <<
        run(krn, 500) <<
        pull(r) <<
        call([&] {
            second_lambda_passed = (r[0] == (1000 + 499 * 500));
        }) <<
        wait_until_done;

    assert_true(first_lambda_passed);
    assert_true(second_lambda_passed);
}

TEST_CASE_FIXTURE(kernel_event_listener, sum_up_program_fixture) {
    set_buffer();

    cl_uint s_val = 3;

    run krn_runner(krn);

    promise push_p;

    krn_runner.register_pre([&](const promise& p) {
        return p <<
            callc([&]() {
                fill(s.begin(), s.end(), s_val);
                r[0] = 0;

                return push_p <<
                    push(s) <<
                    push(r);
            });
    });
    krn_runner.register_post([&](const promise& p) {
        return p <<
            pull(r);
    });
    krn.range(500);

    cl_uint result[3] = {0};

    promise() <<
        krn_runner <<
        call([&]() { result[0] = r[0]; s_val = 11; }) <<
        krn_runner <<
        call([&]() { result[1] = r[0]; s_val = 100; }) <<
        krn_runner <<
        call([&]() { result[2] = r[0]; }) <<
        wait_until_done;

    assert_true(result[0] == (1500 + 499 * 500 / 2));
    assert_true(result[1] == (5500 + 499 * 500 / 2));
    assert_true(result[2] == (50000 + 499 * 500 / 2));
}

TEST_CASE(pipeline_buf_krn_bindings) {
    program prg = compile(R"EOF(
    typedef float4 pos_t;
    kernel void find_max(global int* buf_f, global int* most_f) {
        size_t id = get_global_id(0);
        atomic_max(most_f, buf_f[id]);
    }
    kernel void find_min(global int* buf_f, global int* most_f) {
        most_f += 1;

        size_t id = get_global_id(0);
        atomic_min(most_f, buf_f[id]);
    }
    )EOF","-cl-kernel-arg-info");

    buffer<cl_float> buf_f {
        0.527220, 0.455024, 0.243937, 0.569419, 0.193320,
        0.109147, 0.056309, 0.505797, 0.088849, 0.286497,
        0.810914, 0.392379, 0.516577, 0.155513, 0.749785,
        0.811884, 0.798685, 0.484439, 0.340455, 0.392970,
        0.074639, 0.763501, 0.761734, 0.521394, 0.878799,

        0.931135, 0.061400, 0.939514, 0.812183, 0.391622,
        0.383235, 0.072958, 0.280965, 0.270403, 0.140101,
        0.090971, 0.624020, 0.457334, 0.748565, 0.963742, // <-- max(39)
        0.733968, 0.875589, 0.703515, 0.667127, 0.275617,
        0.141495, 0.044850, 0.528003, 0.197100, 0.791535
        // min(46) ---^
    };
    buffer<cl_float> most_f { 0.0, 1.0 };

    pipeline pl;
    pl.auto_bind_buffer(buf_f);
    pl.bind_kernel_from_program(prg);
    pl.auto_bind_buffer(most_f);

    kernel* find_max = pl.get_kernel("find_max");
    kernel* find_min = pl.get_kernel("find_min");
    assert_true(find_max != nullptr);
    assert_true(find_min != nullptr);

    promise p;
    promise pushed = p <<
        push(buf_f) <<
        push(most_f);

    promise {
        pushed << run(*find_min, 50),
        pushed << run(*find_max, 50),
    } <<
        pull(most_f) <<
        wait_until_done;

    assert_true(most_f[0] == buf_f[39]);
    assert_true(most_f[1] == buf_f[46]);
}

int main(int argc, char** argv)
{
    std::vector<platform> ps = platform::get();
    std::vector<device> ds = device::get(ps);

    context ctxt(ds.back());
    context_guard cg(ctxt);

    shrtool::unit_test::test_main(argc, argv);
}

