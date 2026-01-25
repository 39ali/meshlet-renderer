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

  wgpu::Buffer vertexBuffer;
  wgpu::Buffer meshletBuffer;
  wgpu::Buffer meshletVertexBuffer;
  wgpu::Buffer meshletTriangleBuffer;

  wgpu::Buffer indirectBuffer;

  uint32_t meshletCount;
  uint32_t trianglesCount;
  uint32_t maxMeshletTriangleCount;
  //
};