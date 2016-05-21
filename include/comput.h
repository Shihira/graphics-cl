#ifndef COMPUT_H_INCLUDED
#define COMPUT_H_INCLUDED

#include "matrix.h"

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <ctime>
#include <cassert>

#include <memory>
#include <set>
#include <iterator>
#include <vector>
#include <functional>
#include <string>
#include <initializer_list>
#include <sstream>
#include <map>

namespace gcl {

typedef cl::Event event;
typedef cl::Program program;

struct comput_error : public std::exception {
public:
    comput_error(const std::string& s) : what_arg_(s) { }

    const char* what() const noexcept override {
        return what_arg_.c_str();
    }

private:
    std::string what_arg_;
};

struct platform : cl::Platform {
    platform(const cl::Platform& p) : cl::Platform(p) { }

    static std::vector<platform> get() {
        std::vector<cl::Platform> cps;
        cl::Platform::get(&cps);
        return std::vector<platform>(cps.begin(), cps.end());
    }
};

struct device : cl::Device {
    device(const cl::Device& d) : cl::Device(d) { }

    enum device_type : cl_device_type {
        DEFAULT = CL_DEVICE_TYPE_DEFAULT,
        CPU = CL_DEVICE_TYPE_CPU,
        GPU = CL_DEVICE_TYPE_GPU,
        ALL = CL_DEVICE_TYPE_ALL
    };

    static std::vector<device> get(
            const std::vector<platform>& ps,
            device_type type = ALL) {
        std::vector<device> ds;
        for(const platform& p : ps) {
            std::vector<cl::Device> cds;
            try {
                p.getDevices(type, &cds);
            } catch(cl::Error e) {
                if(e.err() == CL_DEVICE_NOT_FOUND)
                    continue;
                else throw;
            }
            ds.insert(ds.end(), cds.begin(), cds.end());
        }
        return ds;
    }

    static std::vector<device> get(
            const platform& p,
            device_type type = ALL) {
        std::vector<cl::Device> cds;
        try {
            p.getDevices(type, &cds);
        } catch(cl::Error e) {
            if(e.err() != CL_DEVICE_NOT_FOUND) throw;
        }
        return std::vector<device>(cds.begin(), cds.end());
    }
};

struct context : cl::Context {
    context(const device& d) : cl::Context(d) { }

    void set_current() {
        if(current_context_)
            throw comput_error("Recursive context is not allowed.");
        current_context_ = this;
    }

    static void unset_current() {
        current_context_ = nullptr;
    }

    static context& current() {
        if(!current_context_)
            throw comput_error("No context exists.");
        return *current_context_;
    }

    device get_device() {
        return this->getInfo<CL_CONTEXT_DEVICES>()[0];
    }
private:
    friend struct context_guard;

    static context* current_context_;
};

struct context_guard {
    context_guard(context& ctxt) : ctxt_(ctxt) { ctxt.set_current(); }
    ~context_guard() { ctxt_.unset_current(); }

private:
    context& ctxt_;
};

context* context::current_context_ = nullptr;

template<typename FromType, typename ToType>
struct type_convertor_ {
    static void assign(ToType* t, const FromType* f, size_t sz) {
        ToType* tend = t + sz;
        for(; t != tend; ++t, ++f)
            *t = *f;
    }
};

template<> struct type_convertor_<cl_float4, col4> {
    static void assign(col4* t, const cl_float4* f, size_t sz) {
        col4* tend = t + sz;
        for(; t != tend; ++t, ++f) {
            (*t)[0] = f->s[0];
            (*t)[1] = f->s[1];
            (*t)[2] = f->s[2];
            (*t)[3] = f->s[3];
        }
    }
};

template<> struct type_convertor_<col4, cl_float4> {
    static void assign(cl_float4* t, const col4* f, size_t sz) {
        cl_float4* tend = t + sz;
        for(; t != tend; ++t, ++f) {
            t->s[0] = (*f)[0];
            t->s[1] = (*f)[1];
            t->s[2] = (*f)[2];
            t->s[3] = (*f)[3];
        }
    }
};

template<> struct type_convertor_<cl_float4, row4> {
    static void assign(row4* t, const cl_float4* f, size_t sz) {
        row4* tend = t + sz;
        for(; t != tend; ++t, ++f) {
            (*t)[0] = f->s[0];
            (*t)[1] = f->s[1];
            (*t)[2] = f->s[2];
            (*t)[3] = f->s[3];
        }
    }
};

template<> struct type_convertor_<row4, cl_float4> {
    static void assign(cl_float4* t, const row4* f, size_t sz) {
        cl_float4* tend = t + sz;
        for(; t != tend; ++t, ++f) {
            t->s[0] = (*f)[0];
            t->s[1] = (*f)[1];
            t->s[2] = (*f)[2];
            t->s[3] = (*f)[3];
        }
    }
};

template<> struct type_convertor_<cl_float3, col3> {
    static void assign(col3* t, const cl_float3* f, size_t sz) {
        col3* tend = t + sz;
        for(; t != tend; ++t, ++f) {
            (*t)[0] = f->s[0];
            (*t)[1] = f->s[1];
            (*t)[2] = f->s[2];
        }
    }
};

template<> struct type_convertor_<col3, cl_float3> {
    static void assign(cl_float3* t, const col3* f, size_t sz) {
        cl_float3* tend = t + sz;
        for(; t != tend; ++t, ++f) {
            t->s[0] = (*f)[0];
            t->s[1] = (*f)[1];
            t->s[2] = (*f)[2];
        }
    }
};

template<typename HostType>
struct default_conversion_ {
    typedef HostType type;
};

template<> struct default_conversion_<col4> { typedef cl_float4 type; };
template<> struct default_conversion_<row4> { typedef cl_float4 type; };
template<> struct default_conversion_<col3> { typedef cl_float3 type; };


/* Buffer for General Purpose
 *
 * gcl::buffer wraps operations including transfer data to and from graphic
 * card, convert data between wrapped host data type and raw data.
 *
 * Lazy allocation policy for buffer, host data and device data, so always use
 * buf(), host_data() and dev_data() instead of accessing pointers directly.
 * Furthermore, when HostType and DeviceType are the same type, host_data() is
 * equivalent, and conversion functions do nothing.
 */

enum buffer_type {
    host_map = CL_MEM_USE_HOST_PTR,
    no_access = CL_MEM_HOST_NO_ACCESS,
    direct = CL_MEM_READ_WRITE,
};

struct abstract_buffer {
    virtual cl::Buffer buf() = 0;
    virtual size_t size() const = 0;
    virtual buffer_type type() const = 0;
    virtual size_t size_in_bytes() const = 0;
    virtual void conv_dev_to_host() = 0;
    virtual void conv_host_to_dev() = 0;

    virtual const void* erased_device_data() const = 0;
    virtual void* erased_device_data() = 0;

    virtual ~abstract_buffer() { }
};

template<
    typename HostType,
    typename DeviceType = typename default_conversion_<HostType>::type>
struct buffer : abstract_buffer {
    typedef HostType host_type;
    typedef DeviceType device_type;
    typedef type_convertor_<HostType, DeviceType> convertor;
    typedef type_convertor_<DeviceType, HostType> inv_convertor;

    typedef host_type* iterator;
    typedef host_type const* const_iterator;

    buffer() { }

    // direct read/write is a better choice for small amount of data
    buffer(const std::initializer_list<host_type>& l,
            buffer_type t = direct) :
        size_(std::distance(l.begin(), l.end())),
        bt_(t)
    {
        std::copy(l.begin(), l.end(), host_data());
    }

    // device-end data chunk
    buffer(size_t count, buffer_type t = no_access) :
        size_(count), bt_(t) { }

    // large amount of duplicated data
    buffer(size_t count, host_type v, buffer_type t = host_map) :
        size_(count),
        bt_(t)
    {
        std::fill_n(host_data(), size_, v);
    }

    iterator begin() { return host_data(); }
    iterator end() { return host_data() + size_; }
    const_iterator begin() const { return host_data(); }
    const_iterator end() const { return host_data() + size_; }
    const_iterator cbegin() const { return host_data(); }
    const_iterator cend() const { return host_data() + size_; }

    cl::Buffer buf() override {
        if(dev_buf_() == NULL) {
            dev_buf_ = cl::Buffer(
                context::current(),
                static_cast<cl_device_type>(bt_),
                size() * sizeof(device_type),
                bt_ == host_map ? device_data() : nullptr
            );
        }
        return dev_buf_;
    }

    host_type* host_data() {
        // TODO: use template specialisation instead
        //if(std::is_same<host_type, device_type>::value)
        //    return reinterpret_cast<host_type*>(device_data());
        if(!host_data_)
            host_data_ = std::unique_ptr<host_type[]>
                (new host_type[size()]);
        return host_data_.get();
    }

    host_type const* host_data() const {
        //if(std::is_same<host_type, device_type>::value)
        //    return reinterpret_cast<host_type const*>(device_data());
        if(!host_data_)
            host_data_ = std::unique_ptr<host_type[]>
                (new host_type[size()]);
        return host_data_.get();
    }

    device_type* device_data() {
        if(!dev_data_)
            dev_data_ = std::unique_ptr<device_type[]>
                (new device_type[size()]);
        return dev_data_.get();
    }

    device_type const* device_data() const {
        if(!dev_data_)
            dev_data_ = std::unique_ptr<device_type[]>
                (new device_type[size()]);
        return dev_data_.get();
    }

    void* erased_device_data() override { return device_data(); }
    const void* erased_device_data() const override { return device_data(); }

    size_t size() const override { return size_; }
    size_t size_in_bytes() const override { return sizeof(device_type) * size_; }
    host_type& operator[](size_t idx) {
        if(idx >= size()) throw comput_error("Access out of range");
        return host_data()[idx];
    }
    const host_type& operator[](size_t idx) const {
        if(idx >= size()) throw comput_error("Access out of range");
        return host_data()[idx];
    }

    buffer& operator=(buffer&& buf) {
        size_ = std::move(buf.size_);
        bt_ = std::move(buf.bt_);
        host_data_ = std::move(buf.host_data_);
        dev_data_ = std::move(buf.dev_data_);
        dev_buf_ = std::move(buf.dev_buf_);

        return *this;
    }

    buffer_type type() const override {
        return bt_;
    }

private:
    size_t size_;
    buffer_type bt_;
    mutable std::unique_ptr<host_type[]> host_data_;
    mutable std::unique_ptr<device_type[]> dev_data_;
    cl::Buffer dev_buf_;

public:
    void conv_dev_to_host() override {
        //if((void*)host_data() == (void*)device_data()) return;
        inv_convertor::assign(host_data(), device_data(), size());
    }

    void conv_host_to_dev() override {
        //if((void*)host_data() == (void*)device_data()) return;
        convertor::assign(device_data(), host_data(), size());
    }
};

static cl::Buffer nullptr_buf(NULL);

struct kernel : cl::Kernel {
    using cl::Kernel::Kernel;

    cl_int set_buffer(cl_uint index, abstract_buffer& b) {
        return setArg(index, b.buf());
    }

    cl_int set_null(cl_uint index) {
        return setArg(index, nullptr_buf);
    }

    void range(size_t r) { range_ = r; }
    size_t range() const { return range_; }

    const std::map<std::string, int>& indices() {
        if(index_.empty()) {
            size_t args = getInfo<CL_KERNEL_NUM_ARGS>();
            try {
                for(size_t arg_i = 0; arg_i < args; ++arg_i) {
                    std::string arg_name = getArgInfo<CL_KERNEL_ARG_NAME>(arg_i);
                    index_[arg_name] = arg_i;
                }
            } catch(cl::Error e) {
                if(e.err() == CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
                    throw comput_error("Program wasn't compiled with -cl-kernel-arg-info option");
                throw;
            }
        }

        return index_;
    }

    int get_index(const std::string& s) {
        auto& idx = indices();

        auto i = idx.find(s);
        if(i == idx.end())
            return -1;
        else return i->second;
    }


private:
    std::map<std::string, int> index_;
    size_t range_ = 1;
};

program compile(const std::string& s,
    const std::string& options = "")
{
    program prg(context::current(), s);

    try {
        prg.build(options.c_str());
    } catch(cl::Error e) {
        if(e.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string err_msg = "Compilation Error in ";
            err_msg += std::string(s.begin(), s.begin() + 50);
            err_msg += "...:\n";
            err_msg += prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    context::current().get_device());
            throw comput_error(err_msg);
        } else throw;
    }

    return prg;
}

program compile(std::istream& f, const std::string& options = "")
{
    return compile(std::string(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>()), options);
}

program compile(std::istream&& f, const std::string& options = "")
{
    return compile(f, options);
}

/* Pipeline Assembles Elements like Buffers and Kernels
 *
 * A pipeline is a manager of a series of kernels and buffers. You need to
 * register all buffers with their name bound, and all kernels with their
 * dependencies bound. The pipeline will automatically bind buffers to
 * processors arguments whose name matchs your bound name, and will resolve the
 * topology graph by launching kernels automatically.
 */

struct pipeline {
    void bind_buffer(const std::string& name, abstract_buffer& buf) {
        bufname_buf_index_[name] = &buf;

        auto i = bufname_krn_index_.find(name);
        if(i == bufname_krn_index_.end()) return;
        for(auto& ki : i->second) {
            kernel* k = ki.first;
            int index = ki.second;

            k->set_buffer(index, buf);
        }
    }

    void bind_kernel(const std::string& name, kernel& krn) {
        krnname_krn_index_[name] = &krn;

        for(auto& p : krn.indices()) {
            bufname_krn_index_[p.first].insert(
                    std::pair<kernel*, int>(&krn, p.second)
                );

            auto i = bufname_buf_index_.find(p.first);
            if(i != bufname_buf_index_.end())
                krn.set_buffer(p.second, *i->second);
        }
    }

    void bind_kernel_from_program(const program& prg) {
        std::vector<cl::Kernel> kernels;
        cl::Program(prg).createKernels(&kernels);

        for(cl::Kernel& clk : kernels) {
            kernel* k = new kernel(clk());
            kernels_.push_back(k);
            bind_kernel(k->getInfo<CL_KERNEL_FUNCTION_NAME>(), *k);
        }
    }

    void add_target(const std::string& t) {
        throw comput_error("Not implemented");
    }

    void add_dependency(const std::string& t, const std::string& d) {
        throw comput_error("Not implemented");
    }

    kernel* get_kernel(const std::string& n) {
        auto i = krnname_krn_index_.find(n);
        if(i == krnname_krn_index_.end()) return nullptr;
        return i->second;
    }

    ~pipeline() {
        for(kernel* k : kernels_) delete k;
    }

private:
    std::map<std::string, kernel*> krnname_krn_index_;
    std::map<std::string, abstract_buffer*> bufname_buf_index_;
    std::map<std::string, std::set<std::pair<kernel*, int>>> bufname_krn_index_;

    std::vector<kernel*> kernels_;
};

#define auto_bind_buffer(buf) bind_buffer(#buf, buf);
#define auto_bind_kernel(krn) bind_kernel(#krn, krn);

/*
 * Error handler for gcl test utilities
 */

template<typename Base>
struct comput_error_handler {
    static bool run_test(const std::function<void()>& f) {
        try {
            return Base::run_test(f);
        } catch(const cl::Error& e) {
            std::cout << "\033[1;33mCL Error\033[0m: ";
            std::cout << e.what() << " (" << e.err() << ')' << std::endl;
            return false;
        } catch(const comput_error& e) {
            std::cout << "\033[1;33mComput Error\033[0m: ";
            std::cout << e.what() << std::endl;
            return false;
        }
    }
};


}

#endif // COMPUT_H_INCLUDED

