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

class comput_error : public std::exception {
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

    static thread_local context* current_context_;
};

struct context_guard {
    context_guard(context& ctxt) : ctxt_(ctxt) { ctxt.set_current(); }
    ~context_guard() { ctxt_.unset_current(); }

private:
    context& ctxt_;
};

thread_local context* context::current_context_ = nullptr;

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


/**
 * buffer
 *
 * gcl::buffer wraps operations including transfer data to and from graphic
 * card, convert data between wrapped host data type and raw data.
 *
 * Lazy allocation policy for buffer, host data and device data, so always use
 * buf(), host_data() and dev_data() instead of accessing pointers directly.
 * Furthermore, when HostType and DeviceType are the same type, host_data() is
 * equivalent, and conversion functions do nothing.
 */

struct abstract_buffer {
    virtual cl::Buffer buf() = 0;
    virtual size_t size() const = 0;
    virtual size_t size_in_bytes() const = 0;
    virtual void conv_dev_to_host() = 0;
    virtual void conv_host_to_dev() = 0;

    virtual ~abstract_buffer() { }
};

enum buffer_type {
    host_map = CL_MEM_USE_HOST_PTR,
    no_access = CL_MEM_HOST_NO_ACCESS,
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

    buffer(const std::initializer_list<host_type>& l,
            buffer_type t = host_map) :
        size_(std::distance(l.begin(), l.end())),
        bt_(t)
    {
        std::copy(l.begin(), l.end(), host_data());
    }

    buffer(size_t count, buffer_type t = no_access) :
        size_(count), bt_(t) { }

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

    cl::Buffer buf() {
        if(dev_buf_() == NULL) {
            dev_buf_ = cl::Buffer(
                context::current(),
                //bt_ == host_map ? CL_MEM_USE_HOST_PTR : CL_MEM_READ_WRITE,
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

    size_t size() const { return size_; }
    size_t size_in_bytes() const { return sizeof(device_type) * size_; }
    host_type& operator[](size_t idx) { return host_data()[idx]; }
    const host_type& operator[](size_t idx) const { return host_data()[idx]; }

    buffer& operator=(buffer&& buf) {
        size_ = std::move(buf.size_);
        bt_ = std::move(buf.bt_);
        host_data_ = std::move(buf.host_data_);
        dev_data_ = std::move(buf.dev_data_);
        dev_buf_ = std::move(buf.dev_buf_);

        return *this;
    }

private:
    size_t size_;
    buffer_type bt_;
    mutable std::unique_ptr<host_type[]> host_data_;
    mutable std::unique_ptr<device_type[]> dev_data_;
    cl::Buffer dev_buf_;

public:
    void conv_dev_to_host() {
        //if((void*)host_data() == (void*)device_data()) return;
        inv_convertor::assign(host_data(), device_data(), size());
    }

    void conv_host_to_dev() {
        //if((void*)host_data() == (void*)device_data()) return;
        convertor::assign(device_data(), host_data(), size());
    }
};

static cl::Buffer nullptr_buf(NULL);

typedef std::vector<event> event_set;

/*
 * A promise is a manager of OpenCL command queue and events. You can
 * conveniently enqueue some operations through operator<<, and operations
 * would probably and should be non-block. All operations should inherit from
 * promise_runnable the base class.
 *
 * The semantic of operator<< is that, enqueue the operation and do not execute
 * it instantly, but wait for the completion of previous operation (Even if its
 * the initial promise, there is no guarantee for the instant execution) instead.
 * Many of time the behaviour is implementation-relevant, so it would possibly
 * be executed instantly and get a right answer -- now do not forget to test
 * on some other OpenCL impls, if you want your program cross-platform.
 *
 * Be careful of the difference between call-by-value and call-by-reference.
 * Most operations are call-by-value, but you can still transform them to
 * call-by-reference ones through wrapping them in a functor(lambda or something
 * like that), but it may cause some performance consumption.
 */

struct promise {
    typedef event (enqueue_func_type)(cl::CommandQueue, const event_set&);
    typedef promise (operation_func_type)(const promise& p);
    typedef void (procedure_type)();

    promise() : cmdq_(
            context::current(),
            context::current().get_device()) { }
    promise(const promise& other) :
        ev_(other.ev_), cmdq_(other.cmdq_) { }
    promise(std::initializer_list<promise> l)
    {
        for(const promise& cp : l) {
            ev_.insert(ev_.end(), cp.ev_.begin(), cp.ev_.end());
            if(cmdq_() == NULL) cmdq_ = cp.cmdq_;
            else if(cmdq_() != cp.cmdq_())
                throw comput_error("All promises have "
                        "to belong to the same queue.");
        }
    }

    promise operator <<(std::function<enqueue_func_type> f) const {
        event_set es{f(cmdq_, ev_)};
        if(es.front()())
            return promise(es, cmdq_);
        else
            return promise(event_set(), cmdq_);
    }

    promise operator <<(std::function<operation_func_type> f) const {
        return f(*this);
    }

    promise operator <<(std::function<procedure_type> f) const {
        cl_event bar_cle;
        std::vector<cl_event> es;
        for(const event& ce : ev_) es.push_back(ce());

        clEnqueueBarrierWithWaitList(cmdq_(), es.size(), es.data(), &bar_cle);
        cl::UserEvent uev(context::current());
        procedure_args* args = new procedure_args { uev, f };

        event e(bar_cle);
        e.setCallback(CL_COMPLETE, procedure_runner, args);
        return promise(event_set{ uev }, cmdq_);
    }

protected:
    const event_set& events() const { return ev_; }
    const cl::CommandQueue& command_queue() const { return cmdq_; }

    struct procedure_args {
        cl::UserEvent uev;
        std::function<procedure_type> f;
    };

    static void procedure_runner(cl_event, cl_int, void* vp_args) {
        procedure_args* args = static_cast<procedure_args*>(vp_args);

        args->f();
        args->uev.setStatus(CL_COMPLETE);
        delete args;
    }

    promise(const event_set& e, const cl::CommandQueue& c) :
        ev_(e), cmdq_(c) { }

private:
    event_set ev_;
    cl::CommandQueue cmdq_;

    friend class promise_runnable;
};

struct promise_runnable {
    typedef std::function<promise(const promise&)> listener_type;

    promise promise_run(const promise& p) const {
        return post_func_(run_body(pre_func_(p)));
    }
    void register_pre(listener_type f) { pre_func_ = f; }
    void register_post(listener_type f) { post_func_ = f; }

    virtual ~promise_runnable() { }

protected:
    virtual promise run_body(const promise& p) const {
        event bodye = run_body(p.command_queue(), p.events());
        return promise(event_set{bodye}, p.command_queue());
    }
    virtual event run_body(cl::CommandQueue cmdq, const event_set& ev) const {
        return event();
    }

    listener_type pre_func_ = does_nothing_;
    listener_type post_func_ = does_nothing_;

private:
    static promise does_nothing_(const promise& p) { return p; }
};

inline promise operator<<(const promise& p, const promise_runnable& r) {
    return r.promise_run(p);
}

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

struct mapop_ : promise_runnable {
    typedef abstract_buffer buf_type;
    typedef void (buf_type::*conv_func_type)();

    buf_type& buf_;

    cl_uint map_flag_;
    conv_func_type conv_func_;

    mapop_(buf_type& b, cl_uint mf, conv_func_type cf) :
        buf_(b), map_flag_(mf), conv_func_(cf) { }

protected:
    event run_body(cl::CommandQueue cmdq,
            const event_set& ev) const override {
        event e;
        void* mem = cmdq.enqueueMapBuffer(buf_.buf(), false, map_flag_, 0,
                buf_.size_in_bytes(), &ev, &e);

        cl::UserEvent uev(context::current());
        conv_args* args = new conv_args { uev, conv_func_, buf_ };
        e.setCallback(CL_COMPLETE, conv_native_kernel, args);

        event_set mapped_es{uev};
        cmdq.enqueueUnmapMemObject(buf_.buf(),
                mem, &mapped_es, &e);

        return e;
    }

private:
    struct conv_args {
        cl::UserEvent uev;
        conv_func_type conv_func;
        buf_type& buf;
    };

    static void conv_native_kernel(cl_event, cl_int, void* vp_args) {
        conv_args* args = static_cast<conv_args*>(vp_args);
        (args->buf.*(args->conv_func))();
        args->uev.setStatus(CL_COMPLETE);
        delete args;
    }
};

struct push : mapop_ {
    push(buf_type& b) :
        mapop_(b, CL_MAP_WRITE, &buf_type::conv_host_to_dev) { }
};

struct pull : mapop_ {
    pull(buf_type& b) :
        mapop_(b, CL_MAP_READ, &buf_type::conv_dev_to_host) { }
};

template<typename HostType, typename DevType>
struct fill_functor_ : promise_runnable {
    typedef buffer<HostType, DevType> buf_type;

    buf_type& buf_;
    typename buf_type::device_type pat_;

    fill_functor_(buf_type& b, const typename buf_type::host_type& pat) :
        buf_(b)
    {
        buf_type::convertor::assign(&pat_, &pat, 1);       
    }

protected:
    event run_body(cl::CommandQueue cmdq,
            const event_set& ev) const override {
        event e;
        cmdq.enqueueFillBuffer(buf_.buf(), pat_, 0, buf_.size_in_bytes(),
                &ev, &e);
        return e;
    }
};

template<typename HostType, typename DevType>
fill_functor_<HostType, DevType> fill(
        buffer<HostType, DevType>& b, const HostType& d) {
    return fill_functor_<HostType, DevType>(b, d);
}

struct run/*_kernel_functor*/ : promise_runnable {
public:
    run/*_kernel_functor*/(kernel& krn, size_t gp = 0) :
        krn_(krn), global_partition_(gp) { }

protected:
    event run_body(cl::CommandQueue cmdq,
            const event_set& ev) const override {
        event e;
        if(global_partition_)
            cmdq.enqueueNDRangeKernel(krn_, cl::NullRange,
                    cl::NDRange(global_partition_), cl::NullRange,
                    &ev, &e);
        else
            cmdq.enqueueNDRangeKernel(krn_, cl::NullRange,
                    cl::NDRange(krn_.range()), cl::NullRange,
                    &ev, &e);
        return e;
    }

private:
    kernel& krn_;
    size_t global_partition_;
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

event wait(cl::CommandQueue cmdq,
        const event_set& ev) {
    cl::WaitForEvents(ev);
    return event(NULL);
}

auto wait_until_done = wait;

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

/*
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

