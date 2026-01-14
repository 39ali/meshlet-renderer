#pragma once
#include <GLFW/glfw3.h>
#include <array>
class InputManager {
public:
  static constexpr int KEY_COUNT = 348; // GLFW_KEY_LAST + 1
  static constexpr int MOUSE_BUTTON_COUNT = 3;

  InputManager();

  void init(GLFWwindow *window);

  bool isKeyPressedOnce(int key) const;
  bool isKeyPressed(int key) const;
  bool isKeyReleased(int key) const;

  bool isMouseButtonPressed(int button) const;
  bool isMouseButtonClicked(int button) const;
  void getMouseDelta(double &dx, double &dy);
  float getScrollX() const { return scrollX; }
  float getScrollY() const { return scrollY; }

  void update() {
    scrollX = 0.0f;
    scrollY = 0.0f;
    deltaX = 0.0;
    deltaY = 0.0;

    for (uint32_t i = 0; i < instance->mouseButtons.size(); i++) {
      instance->mouseButtons[i].clicked = false;
    }

    keysOnce.fill(State::None);
  }

private:
  enum State {
    Pressed,
    Released,
    None,
  };
  std::array<bool, KEY_COUNT> keys;
  std::array<State, KEY_COUNT> keysOnce;
  // std::array<uint8_t, KEY_COUNT> keysPressedOnce;

  struct MouseState {
    bool clicked;
    bool pressed;
  };
  std::array<MouseState, MOUSE_BUTTON_COUNT> mouseButtons;

  double mouseX;
  double mouseY;
  double lastMouseX;
  double lastMouseY;
  double deltaX;
  double deltaY;
  float scrollX;
  float scrollY;

  static InputManager *instance;

  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);
  static void mouseButtonCallback(GLFWwindow *window, int button, int action,
                                  int mods);
  static void scrollCallback(GLFWwindow *window, double xoffset,
                             double yoffset);
  static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos);
};
