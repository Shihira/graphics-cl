#include <ctime>
#include <cstdlib>

#include "logger.h"
#include "exception.h"

#define LOG_ENV "SHRTOOL_LOG_LEVEL"

namespace shrtool {

void abstract_logger::record_prefix()
{
    std::ostream& os = get_stream();

    std::time_t t = std::time(NULL);
    std::tm* tm = std::localtime(&t);

    // *__os << std::put_time(tm, "%F %T ");
    static char time_buffer[64];
    strftime(time_buffer, 64, "%F %T ", tm);
    os << time_buffer;

    os << '[' << name_ << "] " << std::flush;
}


abstract_logger& logger_set::get_by_level(size_t lvl)
{
    auto i = loggers_.find(lvl);
    if(i == loggers_.end() || !i->second)
        throw not_found_error("No logger set for this level");
    return *i->second;
}

static void_logger regius_void_logger;

abstract_logger& logger_manager::get_by_level(size_t lvl)
{
    if(lvl < inst().current_level_)
        return regius_void_logger;
    else
        return inst().ls_.get_by_level(lvl);
}


logger_manager::logger_manager(bool leave_empty)
{
    if(leave_empty) return;
    current_level_ = INFO;

    if(char* env = std::getenv(LOG_ENV)) {
        if(*env && *env >= '0' && *env <= '9') // only check the head
            current_level_ = std::stoul(std::getenv(LOG_ENV));
    }

#define REG_LOG(lvl, type) ls_.set_level_logger(lvl, \
        std::shared_ptr<abstract_logger>(new type##_logger())).set_name(#lvl);
    REG_LOG(DEBUG, stdout);
    REG_LOG(INFO, stdout);
    REG_LOG(WARNING, stderr);
    REG_LOG(ERROR, stderr);
    REG_LOG(FATAL, stderr);
#undef REG_LOG
}

}

