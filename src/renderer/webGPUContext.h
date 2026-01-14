#pragma once
#include <webgpu/webgpu_cpp.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

class WebGPUContext {
public:
  WebGPUContext(GLFWwindow *window, uint32_t width, uint32_t height,
                float xscale, float yscale);
  void beginFrame();
  void endFrame();

  // wgpu::Device device() const { return m_device; }
  // wgpu::Queue queue() const { return m_queue; }
  // wgpu::TextureView currentTextureView() const { return m_currentView; }

public:
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Queue queue;

  wgpu::Surface surface;
  wgpu::TextureView currentView;
  wgpu::TextureFormat backbufferFormat;

  uint32_t width;
  uint32_t height;

  void createSurface(void *window);
};
