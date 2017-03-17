#ifndef GUI_H_INCLUDED
#define GUI_H_INCLUDED

#include <string>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <iostream>

#include <SDL2/SDL.h>

#define SDL_USEREVENT_REPAINT 0

namespace gcl {

class window {
protected:
    SDL_Window* _window = nullptr;
    SDL_Surface* _surface = nullptr;

    window() {
        static bool sdl_has_init = false;
        if(!sdl_has_init) {
            if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
                throw std::runtime_error("Failed to initialize SDL.");
            atexit(SDL_Quit);
        }

        _surface = SDL_GetWindowSurface(_window);
    }

public:
    window(const std::string& title, size_t w = 800, size_t h = 600) {
        window();

        _window = SDL_CreateWindow(title.c_str(),
                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                w, h, SDL_WINDOW_SHOWN);
        if(!_window) throw std::runtime_error("Failed to create window.");

        _surface = SDL_GetWindowSurface(_window);
    }

    ~window() {
        //if(_renderer) SDL_DestroyRenderer(_renderer);
        if(_window) SDL_DestroyWindow(_window);
    }

    int width() const {
        int w, h;
        SDL_GetWindowSize(_window, &w, &h);
        return w;
    }

    int height() const {
        int w, h;
        SDL_GetWindowSize(_window, &w, &h);
        return h;
    }

    SDL_Window* sdl_window() const { return _window; }
    SDL_Surface* sdl_surface() const { return _surface; }
};

class application {
public:
    enum mouse_button {
        none = 0,
        left_button = SDL_BUTTON_LMASK,
        right_button = SDL_BUTTON_RMASK,
        middle_button = SDL_BUTTON_MMASK,
    };

private:
    application() { }
    static application _app;

    std::function<void(void)> _on_paint;
    std::function<void(void)> _on_exit;
    std::function<void(int, int, mouse_button)> _on_mouse_down;
    std::function<void(int, int, mouse_button)> _on_mouse_up;
    std::function<void(int, int, uint32_t)> _on_mouse_move;

    uint32_t fps_ = 60;

    static uint32_t paint_timer_cb_(uint32_t interval, void* paint_cb) {
        SDL_Event e;
        e.type = SDL_USEREVENT;
        e.user.type = SDL_USEREVENT;
        e.user.code = SDL_USEREVENT_REPAINT;

        SDL_PushEvent(&e);

        return interval;
    }

public:

    void register_on_paint(std::function<void(void)> func) {
        _on_paint = func;
    }

    void register_on_exit(std::function<void(void)> func) {
        _on_exit = func;
    }

    void register_on_mouse_down(decltype(_on_mouse_down) func) {
        _on_mouse_down = func;
    }

    void register_on_mouse_up(decltype(_on_mouse_up) func) {
        _on_mouse_up = func;
    }

    void register_on_mouse_move(decltype(_on_mouse_move) func) {
        _on_mouse_move = func;
    }

    void set_fps(uint32_t fps) { fps_ = fps; }
    uint32_t fps() { return fps_; }

    void run() {
        SDL_Event e;
        framerate_controller fc(std::chrono::system_clock::now());

        while(true) {
            while(SDL_PollEvent(&e)) {
                if(e.type == SDL_QUIT) {
                    if(_on_exit) _on_exit();
                    return;
                }

                if(e.type == SDL_MOUSEBUTTONUP) {
                    if(_on_mouse_up) _on_mouse_up(e.button.x, e.button.y,
                        e.button.button == SDL_BUTTON_LEFT ? left_button :
                        e.button.button == SDL_BUTTON_RIGHT ? right_button :
                        e.button.button == SDL_BUTTON_MIDDLE ? middle_button :
                        none);
                }

                if(e.type == SDL_MOUSEBUTTONDOWN) {
                    if(_on_mouse_down) _on_mouse_down(e.button.x, e.button.y,
                        e.button.button == SDL_BUTTON_LEFT ? left_button :
                        e.button.button == SDL_BUTTON_RIGHT ? right_button :
                        e.button.button == SDL_BUTTON_MIDDLE ? middle_button :
                        none);
                }

                if(e.type == SDL_MOUSEMOTION) {
                    if(_on_mouse_move) _on_mouse_move(e.motion.x, e.motion.y,
                            e.motion.state);
                }
            }

            if(_on_paint)
                _on_paint();

            fc.input_recent_tick(std::chrono::system_clock::now());
            double delay = fc.get_sleep_time();
            if(delay > 0) {
                SDL_Delay(delay / 1000);
            }

            std::cerr << fc.get_frame_rate() << std::endl;
        }
    }

    struct framerate_controller {
        std::chrono::system_clock::time_point last_fps_count;
        std::chrono::system_clock::time_point last_tick;

        bool is_last_timeout = true;
        double accum = 0;
        double sleep_time = 0;

        size_t old_frames = 0;
        size_t frames = 0;

        framerate_controller(std::chrono::system_clock::time_point lt) :
            last_fps_count(lt), last_tick(lt) { }

        void input_recent_tick(std::chrono::system_clock::time_point ct) {
            using namespace std::chrono;

            auto dur = ct - last_tick;
            int cost = duration_cast<microseconds>(dur).count();
            last_tick = ct;

            frames += 1;
            if(duration_cast<milliseconds>
                    (ct - last_fps_count).count() > 1000) {
                old_frames = frames;
                frames = 0;
                last_fps_count = system_clock::now();
            }

            is_last_timeout = cost > 16666;

            if(accum == 0) {
                accum = cost;
                sleep_time = 16666 - cost;
            } else {
                accum = (accum * 5 + cost) / 6;
                std::cerr << "accum: " << accum << std::endl;
                if(accum > 16666)
                    sleep_time -= 1000;
                else
                    sleep_time += 1000;
            }
        }

        double get_sleep_time() {
            if(is_last_timeout)
                return 0;
            return sleep_time;
        }

        size_t get_frame_rate() {
            return old_frames;
        }
    };

    static application& instance() {
        return _app;
    }
};

application application::_app;

inline application& app() {
    return application::instance();
}

}

#endif // GUI_H_INCLUDED
