#include "../renderer.h"
#include "flyCamera.h"
#include "inputManager.h"
#include "timer.h"
#include "webGPUContext.h"
#include <GLFW/glfw3.h>
#include <iostream>

#ifdef EMSCRIPTEN
#include <emscripten/emscripten.h>
#endif

uint32_t width = 1280;
uint32_t height = 720;
Timer timer{};
GLFWwindow *window;
InputManager input;
FlyCamera camera(45.0f, width / (float)height, 0.01f, 1000.0f);
Renderer *renderer;
bool isCamMode = false;
float totatlTime = 0;
void loop() {

  double dt = timer.tick();
  totatlTime += dt;

  if (input.isKeyPressedOnce(GLFW_KEY_LEFT_SHIFT)) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    isCamMode = true;
  } else if (input.isKeyReleased(GLFW_KEY_LEFT_SHIFT)) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    isCamMode = false;
  }

  if (isCamMode)
    camera.update(input, dt);

  renderer->render(camera, totatlTime);

  if (input.isKeyPressed(GLFW_KEY_ESCAPE)) {
    glfwSetWindowShouldClose(window, true);
  }

  input.update();

#ifdef __EMSCRIPTEN__
  // we need this for webgpu async events to be triggered
  emscripten_sleep(0);
#endif
}

int main() {
  if (!glfwInit()) {
    return 0;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  window = glfwCreateWindow(width, height, "WebGPU Renderer", nullptr, nullptr);
  float xscale, yscale;
  glfwGetWindowContentScale(window, &xscale, &yscale);

  input.init(window);
  renderer = new Renderer{window, width, height, xscale, yscale};

  camera.position.z = 80;
  camera.position.x = 20;

#ifdef EMSCRIPTEN
  emscripten_set_main_loop(loop, 0, false);
#else
  while (!glfwWindowShouldClose(window)) {
    loop();
    glfwPollEvents();
  }
  glfwTerminate();
#endif

  return 0;
}
