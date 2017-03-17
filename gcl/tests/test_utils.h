#include <comput.h>
#include <common/unit_test.h>

bool cl_error_handler(shrtool::unit_test::test_context::test_func_type fn)
{
    try {
        return fn();
    } catch(cl::Error e) {
        std::cout << "\033[1;31mError\033[0m: " << e.what() <<
            "(" << e.err() << ")" << std::endl;
        return false;
    }
}

