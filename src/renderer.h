#pragma once
#define GLFW_INCLUDE_NONE
#include "renderer/flyCamera.h"
#include "renderer/webGPUContext.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Renderer {
public:
  Renderer(GLFWwindow *window, uint32_t width, uint32_t height, float xscale,
           float yscale);
  void render(const FlyCamera &camera, float time);

  void initGui(GLFWwindow *window);
  void renderGui(wgpu::CommandEncoder &encoder, wgpu::TextureView &view);

  void initCullPipeline();
  void initRenderPipline();

private:
private:
  WebGPUContext ctx;

  wgpu::Sampler basicSampler;
  wgpu::Texture depthTexture;
  wgpu::TextureView depthView;

  wgpu::Buffer sceneBuffer;

  wgpu::RenderPipeline renderPipeline;
  wgpu::BindGroup renderBindGroup;

  wgpu::ComputePipeline cullingPipeline;
  wgpu::BindGroup cullingBindGroup;
  wgpu::Buffer visibleIndexBuffer;
  wgpu::Buffer indirectBuffer;
  wgpu::Buffer depthBuffer;

  // radix sort onesweep
  wgpu::ComputePipeline histogramPipeline;
  wgpu::BindGroup histogramBindGroup1;
  wgpu::BindGroup histogramBindGroup2;
  wgpu::Buffer histogramBuffer;
  wgpu::Buffer radixUniformBuffer;
  //
  wgpu::ComputePipeline prefixPipeline;
  wgpu::BindGroup prefixBindGroup;
  wgpu::Buffer globalOffsetBuffer;
  wgpu::Buffer tileOffsetBuffer;
  //
  wgpu::ComputePipeline scatterPipeline;
  wgpu::BindGroup scatterBindGroup1;
  wgpu::BindGroup scatterBindGroup2;
  wgpu::Buffer keysOutBuffer;
  wgpu::Buffer valsOutBuffer;
  //

  uint32_t gaussianCount;
};