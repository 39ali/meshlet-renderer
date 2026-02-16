#include "webGPUContext.h"
#include <cassert>

// #ifdef EMSCRIPTEN
// #include <emscripten.h>
// #else
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
// #include <dawn/native/DawnNative.h>
// #endif

#include <webgpu/webgpu_glfw.h>
//
#include <dawn/webgpu_cpp_print.h>
#include <iostream>

WebGPUContext::WebGPUContext(GLFWwindow *windowHandle, uint32_t width,
                             uint32_t height, float xscale, float yscale) {

  static constexpr auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
  wgpu::InstanceDescriptor instanceDescriptor{
      .requiredFeatureCount = 1, .requiredFeatures = &kTimedWaitAny};

  instance = wgpu::CreateInstance(&instanceDescriptor);
  if (instance == nullptr) {
    std::cerr << "Instance creation failed!\n";
    // return EXIT_FAILURE;
  }

  wgpu::Future f1 = instance.RequestAdapter(
      nullptr, wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestAdapterStatus status, wgpu::Adapter a,
             wgpu::StringView message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
          std::cout << "RequestAdapter: " << message << "\n";
          exit(0);
        }
        adapter = std::move(a);
      });
  instance.WaitAny(f1, UINT64_MAX);

  wgpu::DeviceDescriptor desc{};
  desc.SetUncapturedErrorCallback([](const wgpu::Device &,
                                     wgpu::ErrorType errorType,
                                     wgpu::StringView message) {
    std::cout << "Error: " << errorType << " - message: " << message << "\n";
  });

#ifndef EMSCRIPTEN
  const char *const enabledToggles[] = {
      "use_user_defined_labels_in_backend",
  };
  wgpu::DawnTogglesDescriptor deviceTogglesDesc;
  deviceTogglesDesc.enabledToggles = enabledToggles;
  deviceTogglesDesc.enabledToggleCount = 1;
  desc.nextInChain = &deviceTogglesDesc;
#endif

  wgpu::Limits limits{};
  adapter.GetLimits(&limits);
  desc.requiredLimits = &limits;

  wgpu::Future f2 = adapter.RequestDevice(
      &desc, wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestDeviceStatus status, wgpu::Device d,
             wgpu::StringView message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          std::cout << "RequestDevice: " << message << "\n";
          exit(0);
        }
        device = std::move(d);
      });
  instance.WaitAny(f2, UINT64_MAX);

  queue = device.GetQueue();

  surface = wgpu::Surface::Acquire(
      wgpuGlfwCreateSurfaceForWindow(instance.Get(), windowHandle));

  wgpu::SurfaceCapabilities capabilities;
  surface.GetCapabilities(adapter, &capabilities);
  backbufferFormat = capabilities.formats[0];

  wgpu::SurfaceConfiguration config{.device = device,
                                    .format = backbufferFormat,
                                    .width = width * int(xscale),
                                    .height = height * int(yscale)};

  std::cout << "surface:" << config.width << "," << config.height << std::endl;
  surface.Configure(&config);

  this->width = config.width;
  this->height = config.height;
}

void WebGPUContext::beginFrame() {
  // Get current texture from surface
  //   WGPUTexture backbuffer = wgpuSurfaceGetCurrentTexture(m_surface);
  //   m_currentView = wgpuTextureCreateView(backbuffer, nullptr);

  //   // Command encoder
  //   WGPUCommandEncoder encoder =
  //       wgpuDeviceCreateCommandEncoder(m_device, nullptr);

  //   // Render pass
  //   WGPURenderPassColorAttachment color{};
  //   color.view = m_currentView;
  //   color.loadOp = WGPULoadOp_Clear;
  //   color.storeOp = WGPUStoreOp_Store;
  //   color.clearValue = {0.1f, 0.2f, 0.3f, 1.0f};

  //   WGPURenderPassDescriptor pass{};
  //   pass.colorAttachmentCount = 1;
  //   pass.colorAttachments = &color;

  //   WGPURenderPassEncoder rp = wgpuCommandEncoderBeginRenderPass(encoder,
  //   &pass); wgpuRenderPassEncoderEnd(rp);

  //   // Submit
  //   WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
  //   wgpuQueueSubmit(m_queue, 1, &cmd);

  //   // Release backbuffer (modern API replaces swapchain)
  //   wgpuTextureRelease(backbuffer);
  //   wgpuCommandBufferRelease(cmd);
  //   wgpuCommandEncoderRelease(encoder);
}

void WebGPUContext::endFrame() {
  // Nothing needed; presenting happens automatically in modern Dawn/WebGPU
  // Optionally, you could flush the queue here if desired
}
