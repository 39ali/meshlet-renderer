#include "inputManager.h"
#include <cassert>
#include <iostream>
InputManager *InputManager::instance = nullptr;

InputManager::InputManager() : scrollX(0.0f), scrollY(0.0f) {
  keys.fill(0);
  keysOnce.fill(State::None);
  mouseButtons.fill({false, false});
}

void InputManager::init(GLFWwindow *window) {
  if (!instance) {
    instance = this;
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);

    glfwGetCursorPos(window, &mouseX, &mouseY);
    lastMouseX = mouseX;
    lastMouseY = mouseY;
  }
}

bool InputManager::isKeyPressed(int key) const {
  assert(key < KEY_COUNT);
  return keys[key];
}

bool InputManager::isKeyReleased(int key) const {
  assert(key < KEY_COUNT);
  return keysOnce[key] == State::Released;
}

bool InputManager::isKeyPressedOnce(int key) const {
  assert(key < KEY_COUNT);
  return keysOnce[key] == State::Pressed;
}

bool InputManager::isMouseButtonPressed(int button) const {
  assert(button < MOUSE_BUTTON_COUNT);
  return mouseButtons[button].pressed;
}

bool InputManager::isMouseButtonClicked(int button) const {
  assert(button < MOUSE_BUTTON_COUNT);
  return mouseButtons[button].clicked;
}

void InputManager::getMouseDelta(double &dx, double &dy) {
  dx = deltaX;
  dy = deltaY;
}

void InputManager::keyCallback(GLFWwindow *window, int key, int scancode,
                               int action, int mods) {
  if (!instance)
    return;
  if (key < 0 || key >= KEY_COUNT)
    return;

  instance->keys[key] = action != GLFW_RELEASE;

  State s = State::None;
  if (action == GLFW_PRESS)
    s = State::Pressed;
  if (action == GLFW_RELEASE)
    s = State::Released;

  instance->keysOnce[key] = s;
}

void InputManager::mouseButtonCallback(GLFWwindow *window, int button,
                                       int action, int mods) {
  if (!instance)
    return;

  auto &state = instance->mouseButtons[button];

  if (action == GLFW_PRESS) {
    state.pressed = true;
  } else if (action == GLFW_RELEASE) {
    if (state.pressed) {
      state.clicked = true;
    }
    state.pressed = false;
  }
}

void InputManager::scrollCallback(GLFWwindow *window, double xoffset,
                                  double yoffset) {
  if (!instance)
    return;

  instance->scrollX += static_cast<float>(xoffset);
  instance->scrollY += static_cast<float>(yoffset);
}

void InputManager::cursorPosCallback(GLFWwindow *window, double xpos,
                                     double ypos) {
  if (!instance)
    return;

  instance->mouseX = xpos;
  instance->mouseY = ypos;

  instance->deltaX += xpos - instance->lastMouseX;
  instance->deltaY += instance->lastMouseY - ypos;
  instance->lastMouseX = xpos;
  instance->lastMouseY = ypos;
}