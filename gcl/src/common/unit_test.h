#ifndef UNIT_TEST_H_INCLUDED
#define UNIT_TEST_H_INCLUDED

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <iostream>
#include <sstream>

#include "singleton.h"
#include "exception.h"

#ifndef TEST_SUITE
#define TEST_SUITE "Default"
#endif

namespace shrtool {

namespace unit_test {

class test_case {
public:
    typedef std::function<void()> test_case_func_type;

    test_case_func_type test_case_func;
    std::string name;

    test_case(
            const std::string& nm,
            test_case_func_type fn) :
        test_case_func(fn), name(nm) { }

    test_case(test_case&& tc) :
        test_case_func(std::move(tc.test_case_func)),
        name(std::move(tc.name)) { }

    test_case(const test_case& tc) :
        test_case_func(tc.test_case_func), name(tc.name) { }
};

class test_suite {
    typedef std::vector<test_case> test_cases_list_;
    test_cases_list_ test_cases_;

public:
    typedef test_cases_list_::iterator iterator;
    typedef test_cases_list_::const_iterator const_iterator;

    std::string name;

    test_suite(const std::string& nm = "unnamed") :
        name(nm) { }

    test_suite(test_suite&& ts) :
        test_cases_(std::move(ts.test_cases_)),
        name(std::move(ts.name)) { }

    test_suite& operator=(test_suite&& ts) {
        test_cases_ = std::move(ts.test_cases_);
        name = std::move(ts.name);
        return *this;
    }

    void add_test_case(test_case&& tc) {
        test_cases_.emplace_back(std::move(tc));
    }

    inline bool test_all();

    iterator begin() { return test_cases_.begin(); }
    iterator end() { return test_cases_.end(); }
    const_iterator cbegin() const { return test_cases_.cbegin(); }
    const_iterator cend() const { return test_cases_.cend(); }
};

class test_context : public generic_singleton<test_context> {
    std::vector<test_suite> suites_;
    std::map<std::string, test_suite*> name_suites_map_;
    bool stop_on_failure_ = false;

public:
    typedef std::function<bool()> test_func_type;
    typedef std::function<bool(test_func_type)> runner_type;

    std::stringstream ctest;
    std::stringstream full_log;
    std::vector<runner_type> runners_list;

    test_context() { }

    bool run_test(test_func_type fn) {
        runner_type cur_r = [](test_func_type f) -> bool { return f(); };
        for(runner_type r : runners_list) {
            runner_type prev_r = cur_r;
            cur_r = [r, prev_r](test_func_type f) -> bool {
                return r([prev_r, f]() -> bool { return prev_r(f); });
            };
        }

        return cur_r(fn);
    }

    static void stop_on_failure(bool b) { inst().stop_on_failure_ = b; }
    static bool stop_on_failure() { return inst().stop_on_failure_; }

    static void add_test_case(
            const std::string& test_suite_name,
            test_case&& tc) {
        auto& nsm = inst().name_suites_map_;
        auto i = nsm.find(test_suite_name);

        if(i == nsm.end()) {
            inst().suites_.emplace_back(test_suite(test_suite_name));
            nsm.insert(std::make_pair(test_suite_name, &inst().suites_.back()));
            inst().suites_.back().add_test_case(std::move(tc));
        } else {
            i->second->add_test_case(std::move(tc));
        }
    }

    static bool test_all() {
        bool state = true;
        for(auto& e : inst().suites_) {
            state &= e.test_all();

            if(stop_on_failure() && !state)
                break;
        }

        return state;
    }

    static void commit_log(const std::string name) {
        inst().full_log
            << "\033[1m----- " << name << " -----\033[0m\n"
            << inst().ctest.str();

        inst().ctest.str("");
    }

    static const test_suite& suite(const std::string& name) {
        static const test_suite empty_ts;
        auto i = inst().name_suites_map_.find(name);
        if(i == inst().name_suites_map_.end()) return empty_ts;
        return *i->second;
    }
};

#define ctest (shrtool::unit_test::test_context::inst().ctest)

struct test_case_adder__ {
    test_case_adder__(
            const std::string& test_suite_name,
            const std::string& test_case_name,
            test_case::test_case_func_type fn) {
        test_context::add_test_case(
                test_suite_name,
                test_case(test_case_name, fn));
    }
};

inline bool test_suite::test_all() {
    bool state = true;
    for(auto& tc : test_cases_) {
        std::cout << "Running " << tc.name << "..." << std::flush;

        try {
            bool res = test_context::inst().run_test([&tc]() {
                    tc.test_case_func();
                    return true;
                });
            state &= res;
            if(res)
                std::cout << "\033[1;32mPassed\033[0m" << std::endl;
        } catch(assert_error e) {
            std::cout << "\033[1;33mFailed\033[0m: "
                << e.what() << std::endl;
            state &= false;
        }

#ifndef EXPOSE_EXCEPTION
        catch(error_base e) {
            std::cout << "\033[1;31mError\033[0m: " << e.what() << std::endl;
            state &= false;
        } catch(std::exception e) {
            std::cout << "\033[1;31mError\033[0m: " << e.what() << std::endl;
            state &= false;
        } catch(...) {
            std::cout << "\033[1;31mError\033[0m: Unknown" << std::endl;
            state &= false;
        }
#endif

        if(!ctest.str().empty())
            test_context::commit_log(name + "/" + tc.name);

        if(test_context::stop_on_failure() && !state)
            break;
    }
    return state;
}

inline int test_main(int argc, char* argv[]) {
    bool status = test_context::test_all();
    std::cout << test_context::inst().full_log.str();

    return status ? 0 : -1;
}

}

}

#define TEST_CASE(func_name) \
    void func_name(); \
    shrtool::unit_test::test_case_adder__ test_case_adder_##func_name##__( \
        TEST_SUITE, #func_name, func_name); \
    void func_name()

#define TEST_CASE_FIXTURE(func_name, fix_name) \
    struct test_case_##func_name##_fixture__ : fix_name { \
        inline void operator() (); }; \
    shrtool::unit_test::test_case_adder__ test_case_adder_##func_name##__( \
        TEST_SUITE, #func_name, [](){ test_case_##func_name##_fixture__ d; d(); }); \
    void test_case_##func_name##_fixture__::operator() ()

#define assert_true(expr) { \
    if(!bool(expr)) { \
        std::stringstream ss; \
        ss << #expr << " != true"; \
        throw  shrtool::assert_error(ss.str()); \
    } \
}

#define assert_false(expr) { \
    if(bool(expr)) { \
        std::stringstream ss; \
        ss << #expr << " != false"; \
        throw  shrtool::assert_error(ss.str()); \
    } \
}

#define assert_equal(expr1, expr2) { \
    if((expr1) == (expr2)) { } \
    else { \
        std::stringstream ss; \
        ss << #expr1 << " != " << #expr2; \
        throw  shrtool::assert_error(ss.str()); \
    } \
}

#define assert_equal_print(expr1, expr2) { \
    auto val1 = (expr1); \
    auto val2 = (expr2); \
    if(val1 == val2) { } \
    else { \
        std::stringstream ss; \
        ss << #expr1 << "(" << val1 << ")" << " != " \
            << #expr2 << "(" << val2 << ")"; \
        throw  shrtool::assert_error(ss.str()); \
    } \
}

#define assert_float_close(expr1, expr2, bias) { \
    auto val1 = (expr1); \
    auto val2 = (expr2); \
    double diff = val1 - val2; \
    if(diff < 0) diff = -diff; \
    if(diff > bias) { \
        std::stringstream ss; \
        ss << expr1 << "(" << val1 << ")" << " !~ " \
            << expr2 << "(" << val2 << ")"; \
        throw  shrtool::assert_error(ss.str()); \
    } \
}

#define assert_except(expr, exc) { \
    bool correct_exc = false; \
    try { expr; } \
    catch(exc e) { correct_exc = true; } \
    /* catch(...) { } */ \
    if(!correct_exc) { \
        throw  shrtool::assert_error("Exception " #exc \
                " was not catched in `" #expr "`"); \
    } \
}

#define assert_no_except(expr) { \
    try { expr; } \
    catch(...)  { \
        throw  shrtool::assert_error("Exception catched"); \
    } \
}

#define assert_float_equal(expr1, expr2) \
    assert_float_close(expr1, expr2, 0.00001)

#endif // UNIT_TEST_H_INCLUDED

