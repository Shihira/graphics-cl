#ifndef GUI_H_INCLUDED
#define GUI_H_INCLUDED

#include <string>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <iostream>

#define SDL_MAIN_HANDLED

#include <SDL2/SDL.h>

#include "common/singleton.h"

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

class application : public shrtool::generic_singleton<application> {
public:
    enum mouse_button {
        none = 0,
        left_button = SDL_BUTTON_LMASK,
        right_button = SDL_BUTTON_RMASK,
        middle_button = SDL_BUTTON_MMASK,
    };

private:
    std::function<void(void)> _on_paint;
    std::function<void(void)> _on_exit;
    std::function<void(int, int, mouse_button)> _on_mouse_down;
    std::function<void(int, int, mouse_button)> _on_mouse_up;
    std::function<void(int, int, uint32_t)> _on_mouse_move;
    std::function<void(int, int)> _on_mouse_wheel;

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

    void register_on_mouse_wheel(decltype(_on_mouse_wheel) func) {
        _on_mouse_wheel = func;
    }

    void run() {
        SDL_Event e;

        uint32_t fps_counter = 0;
        auto last_fc_time = std::chrono::system_clock::now();

        while(true) {
            int has_event = SDL_PollEvent(&e);

            if(_on_paint && !has_event) {
                _on_paint();

                // FPS counter
                fps_counter += 1;
                auto dur = std::chrono::system_clock::now() -
                    last_fc_time;
                if(std::chrono::duration_cast<
                        std::chrono::seconds>(dur).count() >= 1) {
                    std::cout << "FPS: " << fps_counter << std::endl;
                    fps_counter = 0;
                    last_fc_time = std::chrono::system_clock::now();
                }

                continue;
            }

            if(has_event && e.type == SDL_QUIT) {
                if(_on_exit) _on_exit();
                break;
            }

            if(has_event && e.type == SDL_MOUSEBUTTONUP) {
                if(_on_mouse_up) _on_mouse_up(e.button.x, e.button.y,
                    e.button.button == SDL_BUTTON_LEFT ? left_button :
                    e.button.button == SDL_BUTTON_RIGHT ? right_button :
                    e.button.button == SDL_BUTTON_MIDDLE ? middle_button :
                    none);
            }

            if(has_event && e.type == SDL_MOUSEBUTTONDOWN) {
                if(_on_mouse_down) _on_mouse_down(e.button.x, e.button.y,
                    e.button.button == SDL_BUTTON_LEFT ? left_button :
                    e.button.button == SDL_BUTTON_RIGHT ? right_button :
                    e.button.button == SDL_BUTTON_MIDDLE ? middle_button :
                    none);
            }

            if(has_event && e.type == SDL_MOUSEMOTION) {
                if(_on_mouse_move) _on_mouse_move(e.motion.x, e.motion.y,
                        e.motion.state);
            }

            if(has_event && e.type == SDL_MOUSEWHEEL) {
                if(_on_mouse_wheel) _on_mouse_wheel(
                        e.wheel.x, e.wheel.y);
            }
        }
    }
};

}

#endif // GUI_H_INCLUDED
