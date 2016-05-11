#ifndef PROMISE_H_INCLUDED
#define PROMISE_H_INCLUDED

#include "comput.h"

namespace gcl {

typedef std::vector<event> event_set;

struct promise_runnable;

/* Promise Chaining for Asynchronous OpenCL Operation
 *
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
        }
    }

    /*
     * The `then` function in Other Promise Implementations
     */
    promise operator<<(const promise_runnable& r) const;

protected:
    const event_set& events() const { return ev_; }
    const cl::CommandQueue& command_queue() const { return cmdq_; }

    promise(const event_set& e, const cl::CommandQueue& c) :
            ev_(e), cmdq_(c) {
        // reject null events
        if(e[0]() == NULL) ev_.clear();
    }

private:
    event_set ev_;
    cl::CommandQueue cmdq_;

    friend class promise_runnable;
};

/* Base Class of Asynchronous Operations
 *
 * All operation to enqueue should be derived from promise_runnable. This class
 * allows you to customize an operation instance through registering pre and
 * post listener, through which you can enqueue some extra operation before
 * and after the certain operation is enqueued.
 *
 * Override one of <run_body> to implement your own operation.
 */
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

    listener_type pre_func_ = do_nothing_;
    listener_type post_func_ = do_nothing_;

private:
    static promise do_nothing_(const promise& p) { return p; }
};

inline promise promise::operator<<(const promise_runnable& r) const {
    return r.promise_run(*this);
}

// Abstracts Common Functionalities of `push` and `pop`
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
        conv_args_t* args = new conv_args_t { uev, conv_func_, buf_ };
        e.setCallback(CL_COMPLETE, conv_native_kernel, args);

        event_set mapped_es{uev};
        cmdq.enqueueUnmapMemObject(buf_.buf(),
                mem, &mapped_es, &e);

        return e;
    }

private:
    struct conv_args_t {
        cl::UserEvent uev;
        conv_func_type conv_func;
        buf_type& buf;
    };

    static void conv_native_kernel(cl_event, cl_int, void* vp_args) {
        conv_args_t* args = static_cast<conv_args_t*>(vp_args);
        (args->buf.*(args->conv_func))();
        args->uev.setStatus(CL_COMPLETE);
        delete args;
    }
};

// Convert Data and Push a Buffer
struct push : mapop_ {
    push(buf_type& b) :
        mapop_(b, CL_MAP_WRITE, &buf_type::conv_host_to_dev) { }
};

// Pop a Buffer and Convert Data
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

// Fill a Buffer with a Specified Pattern
template<typename HostType, typename DevType>
fill_functor_<HostType, DevType> fill(
        buffer<HostType, DevType>& b, const HostType& d) {
    return fill_functor_<HostType, DevType>(b, d);
}

// Run a Kernel
struct run_kernel : promise_runnable {
    run_kernel(kernel& krn, size_t gp = 0) :
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

template<typename FuncType>
struct run_procedure_functor_ : promise_runnable {
    typedef FuncType func_type;
    typedef void handler_type(std::function<func_type>, cl::UserEvent);

    run_procedure_functor_(const std::function<func_type>& f,
            const std::function<handler_type>& h) :
            functor_(f), handler_(h) { }

    event run_body(cl::CommandQueue cmdq,
            const event_set& ev) const override {
        event e;
        cmdq.enqueueBarrierWithWaitList(&ev, &e);

        cl::UserEvent uev(context::current());
        arg_t* args = new arg_t { uev, functor_, handler_ };
        e.setCallback(CL_COMPLETE, event_callback, args);

        return uev;
    }

private:
    std::function<func_type> functor_;
    std::function<handler_type> handler_;

    struct arg_t {
        cl::UserEvent uev;
        std::function<func_type> func;
        std::function<handler_type> hdlr;
    };

    static void event_callback(cl_event, cl_int, void* vp_args) {
        arg_t* args = static_cast<arg_t*>(vp_args);
        args->hdlr(args->func, args->uev);
        delete args;
    }
};

struct wait : promise_runnable {
    event run_body(cl::CommandQueue cmdq,
            const event_set& ev) const override {
        cl::WaitForEvents(ev);
        return event(NULL);
    }
};

typedef run_kernel run;
const wait wait_until_done = wait();

/* Run a Functor
 *
 * You can run two types of functors: `promise ()` and `void ()`. For the
 * former, it ensures operations following it not to run before operations
 * enqueued in the returned promise get cleared. Similarily, the latter make
 * sure those operations to run (NOTE: not to be enqueued) after this functor
 * has got called.
 *
 * You should have carefully considered before enqueuing an operation into an
 * existent not OOO promise asynchronously, especially when there are some
 * operations waiting in that queue. When you enqueue dependent operations,
 * they may produce deadlocks. A proper way is to create a new promise or use
 * a certainly empty promise.
 *
 * IMPLEMENTATION DETAILS:
 *
 * First, A barrier is created to gather events waited for by current promise,
 * or rather, combine multiple events to one. Then bind a callback to this
 * event, in which user functor is called. A user event is created and block
 * following operations on it then, and this event will be release (set to
 * CL_COMPLETE) right after user functor is called.
 * 
 * `promise ()` functor is somewhat more complicated, but actually the same
 * idea. When it got the returned promise from funtor, you cannot then set the
 * user event to CL_COMPLETE immediately. A straight idea would be enqueue
 * another functor used to set that user event :-).
 */

typedef run_procedure_functor_<void()> run_functor_type;
typedef run_procedure_functor_<promise()> run_functor_chain_type;

run_functor_type run_functor(const std::function<void()>& f)
{
    return run_procedure_functor_<void()>(f,
        [](std::function<void()> func, cl::UserEvent uev) {
            func();
            uev.setStatus(CL_COMPLETE);
        });
}

run_functor_chain_type run_functor_chain(
        const std::function<promise()>& f)
{
    return run_procedure_functor_<promise()>(f,
        [](std::function<promise()> func, cl::UserEvent uev) {
            func() <<
                run_functor([uev]() -> void {
                    clSetUserEventStatus(uev(), CL_COMPLETE);
                });
        });
}

// Alias for run_functor
run_functor_type call(const std::function<void()>& f)
{ return run_functor(f); }
// Alias for run_functor_chain
run_functor_chain_type callc(const std::function<promise()>& f)
{ return run_functor_chain(f); }


}

#endif // PROMISE_H_INCLUDED
