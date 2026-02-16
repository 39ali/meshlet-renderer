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

  void initDepthCopyPipeline();
  void initHizMipsPipeline();
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
  wgpu::RenderPipeline zpassPipeline;
  wgpu::BindGroup renderBindGroup;

  wgpu::ComputePipeline cullingPipeline;
  wgpu::BindGroup cullingBindGroup;

  wgpu::Buffer vertexBuffer;
  wgpu::Buffer meshletBuffer;
  wgpu::Buffer meshletVertexBuffer;
  wgpu::Buffer meshletTriangleBuffer;

  wgpu::Buffer meshletInstanceBuffer;
  wgpu::Buffer meshInstanceBuffer;

  wgpu::Buffer indirectBuffer;
  wgpu::Buffer readbackBuffer;
  bool readbackPending = false;
  uint32_t renderedMeshletsCount = 0;
  wgpu::Buffer visibleMeshletBuffer;
  wgpu::Buffer prevVisibleMeshletBuffer;

  wgpu::Texture hizTexture;
  wgpu::TextureView hizBaseView;
  std::vector<wgpu::TextureView> hizViews;

  wgpu::ComputePipeline depthCopyPipeline;
  wgpu::BindGroup depthCopyBindGroup;

  wgpu::ComputePipeline hizGenPipeline;
  std::vector<wgpu::BindGroup> hizGenBindGroups;
  bool hasPrevDepth = false;

  uint32_t meshletCount;
  uint32_t trianglesCount;
  uint32_t maxMeshletTriangleCount;
  //
};