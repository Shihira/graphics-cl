#ifndef SINGLETON_H_INCLUDED
#define SINGLETON_H_INCLUDED

#include <memory>

namespace shrtool {

template<typename T>
class __generic_singleton_ptr {
public:
    T* instance = nullptr;

    void reset(T* ptr) {
        if(instance)
            delete instance;
        instance = ptr;
    }

    ~__generic_singleton_ptr() {
        if(instance)
            delete instance;
    }

    operator bool() const { return instance; }
    T& operator*() const { return *instance; }
};

template<typename T>
class generic_singleton {
public:
    ~generic_singleton() {}

    static T& inst() {
        if(!_instance)
            _instance.reset(new T());

        return *_instance;
    }
    
protected:
    generic_singleton() {};
    static __generic_singleton_ptr<T> _instance;

private:
    generic_singleton(const generic_singleton&) = delete;
    generic_singleton& operator=(const generic_singleton &) = delete;
};

template<typename T>
__generic_singleton_ptr<T> generic_singleton<T>::_instance;

}

#endif // SINGLETON_H_INCLUDED
