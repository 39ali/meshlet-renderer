#pragma once
#define GLFW_INCLUDE_NONE
#include "renderer/flyCamera.h"
#include "renderer/webGPUContext.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

struct GaussianGPU {
  glm::vec3 mean;
  float opacity;

  float cov3d[6]; // 24 bytes: [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
  float _pad1[2]; // 8 bytes padding

  glm::vec3 color; // 12 bytes
  float _pad2;     // 4 bytes
};

class Renderer {
public:
  Renderer(GLFWwindow *window, uint32_t width, uint32_t height, float xscale,
           float yscale);
  void render(const FlyCamera &camera, float time);

  void initGui(GLFWwindow *window);
  void renderGui(wgpu::CommandEncoder &encoder, wgpu::TextureView &view);

  void initRenderPipline();
  void initCullPipeline();

private:
private:
  WebGPUContext ctx;

  wgpu::Sampler basicSampler;
  wgpu::Texture depthTexture;
  wgpu::TextureView depthView;

  wgpu::Buffer sceneBuffer;

  std::vector<GaussianGPU> gpuGaussian;
  wgpu::Buffer gaussianBuffer;
  wgpu::Buffer sortedIndicesBuffer;

  wgpu::Buffer visibleIndexBuffer;
  wgpu::Buffer indirectBuffer;

  wgpu::RenderPipeline renderPipeline;
  wgpu::BindGroup renderBindGroup;

  wgpu::ComputePipeline cullingPipeline;
  wgpu::BindGroup cullingBindGroup;

  uint32_t gaussianCount;
};