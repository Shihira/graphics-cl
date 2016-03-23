#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

#include <stdexcept>
#include <string>
#include <functional>
#include <vector>
#include <iostream>
#include <sstream>

struct test_failure : std::logic_error {
    test_failure(const std::string& s) : std::logic_error(s) { }
};

std::stringstream ctest;

struct test_case {
    typedef std::function<void()> test_func_type;

    test_case(const std::string& name, test_func_type test_func) {
        test_cases_.push_back(std::make_pair(name, test_func));
    }

    static void test_all(bool continue_anyway = false) {
        std::stringstream all_log;
        
        for(auto& p : test_cases_) {
            std::cout << "Running " << p.first << " ... ";

            bool passed = true;

            try {
                p.second();
            } catch(const test_failure& e) {
                std::cout << "\033[1;33mFailed\033[0m: ";
                std::cout << e.what() << std::endl;
                passed = false;
            } catch(const std::exception& e) {
                std::cout << "\033[1;31mError\033[0m: ";
                std::cout << e.what() << std::endl;
                passed = false;
            }

            if(!ctest.str().empty()) {
                all_log << "\033[1m----- " << p.first << " -----\033[0m\n";
                all_log << ctest.str(); ctest.str("");
            }

            if(passed)
                std::cout << "\033[1;32mPassed\033[0m" << std::endl;
            else if(!continue_anyway) {
                std::cout << "\033[1;31mAborted.\033[0m" << std::endl;
                break;
            }
        }

        std::cout << all_log.str();
    }

private:
    static std::vector<std::pair<std::string, test_func_type>> test_cases_;
};

std::vector<std::pair<std::string, test_case::test_func_type>> 
test_case::test_cases_;

#define def_test_case(name) \
    static void name(); \
    static test_case add_test_case_##name (#name, name); \
    static void name()

#define def_test_case_with_fixture(name, fixture) \
    struct name : fixture { void operator() (); }; \
    static test_case add_test_case_##name (#name, []() { name()(); }); \
    inline void name::operator() ()

#define assert_true(expr) \
    ((expr) ? true : (throw test_failure( \
        std::string("bool(") + #expr + ") == false (true expected)")))

#define assert_float_equal(expr1, expr2) \
    ((abs((expr1) -(expr2)) < 1e-6) ? true : (throw test_failure( \
        std::string("(") + #expr1 + ") != (" + #expr2 + ")")))

#define assert_false(expr) \
    ((expr) ? (throw test_failure( \
        std::string("bool(") + #expr + ") == true (false expected)")) : false)

#endif // TEST_H_INCLUDED
