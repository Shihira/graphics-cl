#ifndef LOGGER_H_INCLUDED
#define LOGGER_H_INCLUDED

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>
#include <limits>

#include "singleton.h"

namespace shrtool {

class abstract_logger {
protected:
    std::string name_ = "UNKNOWN";
    bool prefix_ = true;

public:
    void set_name(std::string n) { name_ = std::move(n); }
    const std::string& get_name() const { return name_; }
    void enable_prefix() { prefix_ = true; }
    void disable_prefix() { prefix_ = false; }

    virtual void record_prefix();
    virtual std::ostream& get_stream() = 0;
    std::ostream& start_logging() {
        record_prefix();
        return get_stream();
    }

    virtual ~abstract_logger() { }
};

class stdout_logger : public abstract_logger {
public:
    std::ostream& get_stream() override { return std::cout; }
};

class stderr_logger : public abstract_logger {
public:
    std::ostream& get_stream() override { return std::cerr; }
};

class string_logger : public abstract_logger {
    std::stringstream ss__;
public:
    std::ostream& get_stream() override { return ss__; }
};

class file_logger : public abstract_logger {
    std::ofstream fs__;
public:
    std::ostream& get_stream() override { return fs__; }
    file_logger() { }
    file_logger(const std::string& fn) : fs__(fn) { }
    void open(const std::string& fn) { fs__.open(fn); }

    ~file_logger() { fs__.close(); }
};

class void_logger : public abstract_logger {
    class void_stream : public std::streambuf {
    protected:
        int overflow(int c) override { return c; }
    } __void_streambuf;

    std::ostream void_stream__;
public:
    void_logger() : void_stream__(&__void_streambuf) { }

    void record_prefix() override { }
    std::ostream& get_stream() override {
        return void_stream__;
    }
};

enum internal_log_level : size_t {
    LOG_ALL = 0,
    LOG_DEBUG   = 10000,
    LOG_INFO    = 20000,
    LOG_WARNING = 30000,
    LOG_ERROR   = 40000,
    LOG_FATAL   = 50000,
    LOG_NONE = std::numeric_limits<size_t>::max(), // constexpr Shika Dekinai
};

class logger_set {
protected:
    std::map<size_t, std::shared_ptr<abstract_logger>> loggers_;
public:

    abstract_logger& get_by_level(size_t lvl);
    std::shared_ptr<abstract_logger>& share_logger(size_t lvl) {
        return loggers_[lvl];
    }
    abstract_logger& set_level_logger(size_t lvl,
            std::shared_ptr<abstract_logger> plog) {
        loggers_[lvl] = plog;
        return *plog;
    }
};

class logger_manager : public generic_singleton<logger_manager> {
    logger_set ls_;
    size_t current_level_;

public:
    static void set_loggers(const logger_set& s) { inst().ls_ = s; }
    static logger_set& get_logger_set() { return inst().ls_; }
    static abstract_logger& get_by_level(size_t lvl);
    static void set_current_level(size_t lvl) { inst().current_level_ = lvl; }

    logger_manager(bool leave_empty = false);
};

#define debug_log   logger_manager::get_by_level(LOG_DEBUG).start_logging()
#define info_log    logger_manager::get_by_level(LOG_INFO).start_logging()
#define warning_log logger_manager::get_by_level(LOG_WARNING).start_logging()
#define error_log   logger_manager::get_by_level(LOG_ERROR).start_logging()
#define fatal_log   logger_manager::get_by_level(LOG_FATAL).start_logging()

}

#endif // LOGGER_H_INCLUDED
