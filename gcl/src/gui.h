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

        int timer = SDL_AddTimer(1000 / fps_, paint_timer_cb_, NULL);
        uint32_t tick = SDL_GetTicks();

        uint32_t fps_counter = 0;
        auto last_fc_time = std::chrono::system_clock::now();

        while(true) {
            int has_event = SDL_WaitEvent(&e);
            if(!has_event) break;

            if(e.type == SDL_QUIT) {
                if(_on_exit) _on_exit();
                break;
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

            if(e.type == SDL_USEREVENT) {
                if(e.user.code == SDL_USEREVENT_REPAINT) {
                    if(_on_paint) {
                        uint32_t d_tick = (SDL_GetTicks() - tick) / 16;
                        _on_paint();
                        tick += d_tick * 16;

                        // FPS counter
                        fps_counter += 1;
                        auto dur = std::chrono::system_clock::now() -
                            last_fc_time;
                        if(std::chrono::duration_cast<
                                std::chrono::seconds>(dur).count() >= 1) {
                            //std::cerr << fps_counter << std::endl;
                            fps_counter = 0;
                            last_fc_time = std::chrono::system_clock::now();
                        }

                    }
                }
            }
        }

        SDL_RemoveTimer(timer);
    }

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
