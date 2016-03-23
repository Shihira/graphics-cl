#include "matrix.h"

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <ctime>
#include <cassert>

#include <memory>
#include <iterator>
#include <vector>
#include <functional>
#include <string>
#include <initializer_list>

namespace gcl {

typedef cl::Event event;
typedef cl::Program program;

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
            throw std::runtime_error("Recursive context is not allowed.");
        current_context_ = this;
    }

    void unset_current() {
        current_context_ = nullptr;
    }

    static context& current() {
        if(!current_context_)
            throw std::runtime_error("No context exists.");
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

template<typename HostType>
struct default_conversion_ {
    typedef HostType type;
};

template<>
struct default_conversion_<col4> {
    typedef cl_float4 type;
};


////////////////////////////////////////////////////////////////////////////////
// buffer

enum buffer_type {
    host_map = CL_MEM_USE_HOST_PTR,
    no_access = CL_MEM_HOST_NO_ACCESS,
};

template<
    typename HostType,
    typename DeviceType = typename default_conversion_<HostType>::type,
    // DeviceType must be a plain type, namely no pointers or vtable is allowed
    typename Convertor = type_convertor_<HostType, DeviceType>,
    typename InvConvertor = type_convertor_<DeviceType, HostType> >
struct buffer {
    typedef HostType host_type;
    typedef DeviceType device_type;

    typedef host_type* iterator;
    typedef host_type const* const_iterator;

    buffer() { }

    buffer(const std::initializer_list<host_type>& l,
            buffer_type t = host_map) :
        size_(std::distance(l.begin(), l.end())),
        bt_(t),
        host_data_(new host_type[size_])
    {
        std::copy(l.begin(), l.end(), host_data());
    }

    buffer(size_t count, buffer_type t = no_access) :
        size_(count), bt_(t) { }

    buffer(size_t count, host_type v, buffer_type t = host_map) :
        size_(count),
        bt_(t),
        host_data_(new host_type[size_])
    {
        std::fill_n(host_data_.get(), size_, v);
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
        if(!host_data_)
            host_data_ = std::unique_ptr<host_type[]>
                (new host_type[size()]);
        return host_data_.get();
    }

    host_type const* host_data() const {
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
    host_type& operator[](size_t idx) { return host_data()[idx]; }
    const host_type& operator[](size_t idx) const { return host_data()[idx]; }

    buffer& operator=(buffer&& buf) {
        size_ = buf.size_;
        bt_ = buf.bt_;
        host_data_ = std::move(buf.host_data_);
        dev_data_ = std::move(buf.dev_data_);
        dev_buf_ = std::move(dev_buf_);
        return *this;
    }

protected:
    size_t size_;
    buffer_type bt_;
    mutable std::unique_ptr<host_type[]> host_data_;
    mutable std::unique_ptr<device_type[]> dev_data_;
    cl::Buffer dev_buf_;

    template<typename T1, typename T2>
    friend struct pull_functor_;
    template<typename T1, typename T2>
    friend struct push_functor_;
    template<typename T1, typename T2>
    friend struct unpull_functor_;
    template<typename T1, typename T2>
    friend struct unpush_functor_;

    void conv_dev_to_host_() {
        InvConvertor::assign(host_data(), device_data(), size());
    }

    void conv_host_to_dev_() {
        Convertor::assign(device_data(), host_data(), size());
    }
};

static cl::Buffer nullptr_buf(NULL);

struct kernel : cl::Kernel {
    using cl::Kernel::Kernel;

    template<typename T1, typename T2, typename T3, typename T4>
    cl_int set_buffer(cl_uint index, buffer<T1, T2, T3, T4>& b) {
        return setArg(index, b.buf());
    }
};

typedef std::vector<event> event_set;

struct promise {
    typedef event (enqueue_func_type)(
            cl::CommandQueue, const event_set&);

    promise() : cmdq_(context::current(), context::current().get_device()) { }
    promise(const promise& other) :
        ev_(other.ev_), cmdq_(other.cmdq_) { }
    promise(std::initializer_list<promise> l)
    {
        for(const promise& cp : l) {
            ev_.insert(ev_.end(), cp.ev_.begin(), cp.ev_.end());
            if(cmdq_() == NULL) cmdq_ = cp.cmdq_;
            else if(cmdq_() != cp.cmdq_())
                throw std::runtime_error("All promises have "
                        "to belong to the same queue.");
        }
    }

    promise operator <<(std::function<enqueue_func_type> f) const {
        return promise(event_set{f(cmdq_, ev_)}, cmdq_);
    }

protected:
    promise(const event_set& e, const cl::CommandQueue& c) :
        ev_(e), cmdq_(c) { }

    event_set ev_;
    cl::CommandQueue cmdq_;
};

template<typename HostType, typename DevType>
struct pull_functor_ {
    buffer<HostType, DevType>& buf_;

    pull_functor_(buffer<HostType, DevType>& b) : buf_(b) { }

    event operator()(cl::CommandQueue cmdq,
            const event_set& ev) const {
        event e;
        void* mem = cmdq.enqueueMapBuffer(buf_.buf(), true, CL_MAP_READ, 0,
                sizeof(DevType) * buf_.size(), &ev, &e);
        if(mem != (void*)buf_.device_data())
            throw std::runtime_error("Failed to map buffer using host_ptr");
        return e;
    }
};


template<typename HostType, typename DevType>
pull_functor_<HostType, DevType> pull(buffer<HostType, DevType>& b) {
    return pull_functor_<HostType, DevType>(b);
}

template<typename HostType, typename DevType>
struct push_functor_ {
    buffer<HostType, DevType>& buf_;

    push_functor_(buffer<HostType, DevType>& b) : buf_(b) { }

    event operator()(cl::CommandQueue cmdq,
            const event_set& ev) const {
        event e;
        void* mem = cmdq.enqueueMapBuffer(buf_.buf(), true, CL_MAP_WRITE, 0,
                sizeof(DevType) * buf_.size(), &ev, &e);
        assert(mem == (void*)buf_.device_data());
        return e;
    }
};

template<typename HostType, typename DevType>
push_functor_<HostType, DevType> push(buffer<HostType, DevType>& b) {
    return push_functor_<HostType, DevType>(b);
}

template<typename HostType, typename DevType>
struct unpull_functor_ {
    buffer<HostType, DevType>& buf_;

    unpull_functor_(buffer<HostType, DevType>& b) : buf_(b) { }

    event operator()(cl::CommandQueue cmdq,
            const event_set& ev) const {
        event e;
        buf_.conv_dev_to_host_();
        cmdq.enqueueUnmapMemObject(buf_.buf(),
                buf_.device_data(), &ev, &e);
        return e;
    }
};


template<typename HostType, typename DevType>
unpull_functor_<HostType, DevType> unpull(buffer<HostType, DevType>& b) {
    return unpull_functor_<HostType, DevType>(b);
}

template<typename HostType, typename DevType>
struct unpush_functor_ {
    buffer<HostType, DevType>& buf_;

    unpush_functor_(buffer<HostType, DevType>& b) : buf_(b) { }

    event operator()(cl::CommandQueue cmdq,
            const event_set& ev) const {
        event e;
        buf_.conv_host_to_dev_();
        cmdq.enqueueUnmapMemObject(buf_.buf(),
                buf_.device_data(), &ev, &e);
        return e;
    }
};

template<typename HostType, typename DevType>
unpush_functor_<HostType, DevType> unpush(buffer<HostType, DevType>& b) {
    return unpush_functor_<HostType, DevType>(b);
}

struct run {
    const kernel& krn_;
    size_t global_partition_;

    event operator()(cl::CommandQueue cmdq,
            const event_set& ev) const {
        event e;
        cmdq.enqueueNDRangeKernel(krn_, cl::NullRange,
                cl::NDRange(global_partition_), cl::NullRange,
                &ev, &e);
        return e;
    }

    run(const kernel& krn, size_t global_partition) :
        krn_(krn), global_partition_(global_partition) { }
};

program compile(const std::string& s)
{
    program prg(context::current(), s);

    try {
        prg.build();
    } catch(cl::Error e) {
        if(e.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string err_msg = "Compilation Error in ";
            err_msg += std::string(s.begin(), s.begin() + 50);
            err_msg += "...:\n";
            err_msg += prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    context::current().get_device());
            throw std::runtime_error(err_msg);
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

program compile(std::istream& f)
{
    return compile(std::string(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>()));
}

program compile(std::istream&& f)
{
    return compile(f);
}

}

