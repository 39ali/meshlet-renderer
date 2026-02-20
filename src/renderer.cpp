
#include "renderer.h"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include "meshoptimizer.h"

#include <fstream>
#include <iostream>

struct SceneData {
  glm::mat4 proj;
  glm::mat4 view;
  glm::vec3 camPos;
  float _pad0;
  float aspect;
  float fov;
  glm::vec2 nearFar;
  float screenHeight;
  uint32_t hizMipCount;
  uint32_t hasPrevDepth;
  uint32_t late;
};

struct Vertex {
  glm::vec3 position;
  float padding;
};

struct Meshlet {
  uint32_t vertexOffset;
  uint32_t triangleOffset;
  uint32_t vertexCount;
  uint32_t triangleCount;
  glm::vec2 boundingSphere0;
  glm::vec2 boundingSphere1;
  uint32_t boundingCone;
  uint32_t color;
};

struct MeshInstance {
  glm::vec3 position;
  float scale;
  glm::vec4 orientation;
};

struct MeshletInstance {
  uint32_t meshletID;
  uint32_t instanceID;
};

#include <random>
glm::mat4 createRandomTranslation(float minRange, float maxRange) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(minRange, maxRange);

  glm::vec3 randomPos(dis(gen), dis(gen), dis(gen));

  return glm::translate(glm::mat4(1.0f), randomPos);
}

uint32_t generateRandomColor() {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  float r = dist(gen);
  float g = dist(gen);
  float b = dist(gen);

  uint32_t R = static_cast<uint32_t>(r * 255.0f) & 0xFF;
  uint32_t G = static_cast<uint32_t>(g * 255.0f) & 0xFF;
  uint32_t B = static_cast<uint32_t>(b * 255.0f) & 0xFF;
  uint32_t A = 255;

  return (A << 24) | (B << 16) | (G << 8) | R;
}

void loadGLTF(const char *path, std::vector<Vertex> &vertices,
              std::vector<uint32_t> &indices) {
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;

  if (!loader.LoadBinaryFromFile(&model, &err, &warn, path)) {
    throw std::runtime_error("Failed to load glTF");
  }

  const auto &prim = model.meshes[0].primitives[0];

  // positions
  const auto &posAcc = model.accessors.at(prim.attributes.at("POSITION"));
  const auto &posView = model.bufferViews[posAcc.bufferView];
  const auto &posBuf = model.buffers[posView.buffer];

  const float *pos = reinterpret_cast<const float *>(
      &posBuf.data[posView.byteOffset + posAcc.byteOffset]);

  vertices.resize(posAcc.count);
  for (size_t i = 0; i < posAcc.count; i++) {
    vertices[i].position =
        glm::vec3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]);
  }

  // indices
  const auto &idxAcc = model.accessors.at(prim.indices);
  const auto &idxView = model.bufferViews[idxAcc.bufferView];
  const auto &idxBuf = model.buffers[idxView.buffer];

  indices.resize(idxAcc.count);

  if (idxAcc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(
        &idxBuf.data[idxView.byteOffset + idxAcc.byteOffset]);
    for (size_t i = 0; i < indices.size(); i++)
      indices[i] = src[i];
  } else {
    memcpy(indices.data(), &idxBuf.data[idxView.byteOffset + idxAcc.byteOffset],
           indices.size() * sizeof(uint32_t));
  }
}

void buildMeshlets(const std::vector<Vertex> &vertices,
                   const std::vector<uint32_t> &indices,
                   std::vector<Meshlet> &meshletsOut,
                   std::vector<uint32_t> &meshletVertices,
                   std::vector<uint32_t> &meshletTrianglesOut) {
  constexpr uint32_t MaxVerts = 64;
  constexpr uint32_t MaxTris = 126;

  size_t maxMeshlets =
      meshopt_buildMeshletsBound(indices.size(), MaxVerts, MaxTris);

  std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
  meshletVertices.resize(indices.size());

  std::vector<uint8_t> meshletTriangles{};
  meshletTriangles.resize(indices.size());

  size_t count = meshopt_buildMeshlets(
      meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
      indices.data(), indices.size(), &vertices[0].position.x, vertices.size(),
      sizeof(Vertex), MaxVerts, MaxTris, 0.0f);

  meshlets.resize(count);
  meshletsOut.resize(count);

  meshletVertices.resize(meshlets.back().vertex_offset +
                         meshlets.back().vertex_count);

  for (size_t i = 0; i < count; i++) {
    meshletsOut[i] = {
        meshlets[i].vertex_offset,
        meshlets[i].triangle_offset,
        meshlets[i].vertex_count,
        meshlets[i].triangle_count,
    };
  }

  for (size_t i = 0; i < count; i++) {
    const meshopt_Meshlet &meshlet = meshlets[i];
    meshopt_Bounds b = meshopt_computeMeshletBounds(
        &meshletVertices[meshlet.vertex_offset],
        &meshletTriangles[meshlet.triangle_offset], meshlet.triangle_count,
        &vertices[0].position.x, vertices.size(), sizeof(Vertex));

    // todo: make 16bit per element
    meshletsOut[i].boundingSphere0 = glm::vec2(b.center[0], b.center[1]);
    meshletsOut[i].boundingSphere1 = glm::vec2(b.center[2], b.radius);

    meshletsOut[i].boundingCone =
        (uint32_t(uint8_t(b.cone_axis_s8[0]))) |       // x -> byte 0
        (uint32_t(uint8_t(b.cone_axis_s8[1])) << 8) |  // y -> byte 1
        (uint32_t(uint8_t(b.cone_axis_s8[2])) << 16) | // z -> byte 2
        (uint32_t(uint8_t(b.cone_cutoff_s8)) << 24);   // w -> byte 3

    meshletsOut[i].color = generateRandomColor();
  }

  // todo: use u8
  meshletTriangles.resize(meshlets.back().triangle_offset +
                          meshlets.back().triangle_count * 3);
  meshletTrianglesOut.reserve(meshletTriangles.size());
  for (auto v : meshletTriangles) {
    meshletTrianglesOut.push_back(static_cast<uint32_t>(v));
  }
}

Renderer::Renderer(GLFWwindow *window, uint32_t width, uint32_t height,
                   float xscale, float yscale)
    : ctx(WebGPUContext(window, width, height, xscale, yscale)) {

  initGui(window);

  wgpu::SamplerDescriptor samplerDesc = {};
  samplerDesc.magFilter = wgpu::FilterMode::Linear;
  samplerDesc.minFilter = wgpu::FilterMode::Linear;
  samplerDesc.addressModeU = wgpu::AddressMode::Repeat;
  samplerDesc.addressModeV = wgpu::AddressMode::Repeat;
  samplerDesc.addressModeW = wgpu::AddressMode::Repeat;
  basicSampler = ctx.device.CreateSampler(&samplerDesc);

  {
    auto desc = wgpu::TextureDescriptor{
        .label = "depth texture",
        .usage = wgpu::TextureUsage::RenderAttachment |
                 wgpu::TextureUsage::TextureBinding,
        .size = {ctx.width, ctx.height, 1},
        .format = wgpu::TextureFormat::Depth24Plus,
    };
    depthTexture = ctx.device.CreateTexture(&desc);
    depthView = depthTexture.CreateView();
  }

  {
    uint32_t mipCount = floor(log2(std::max(width, height))) + 1;

    auto desc =
        wgpu::TextureDescriptor{.usage = wgpu::TextureUsage::TextureBinding |
                                         wgpu::TextureUsage::StorageBinding |
                                         wgpu::TextureUsage::CopyDst,
                                .size = {ctx.width, ctx.height, 1},
                                .format = wgpu::TextureFormat::R32Float,
                                .mipLevelCount = mipCount};

    hizTexture = ctx.device.CreateTexture(&desc);

    hizViews.reserve(mipCount);
    for (uint32_t i = 0; i < mipCount; i++) {
      wgpu::TextureViewDescriptor viewDesc{};
      viewDesc.baseMipLevel = i;
      viewDesc.mipLevelCount = 1;
      viewDesc.dimension = wgpu::TextureViewDimension::e2D;
      viewDesc.format = wgpu::TextureFormat::R32Float;

      hizViews.push_back(hizTexture.CreateView(&viewDesc));
    }

    std::cout << "hizTexture mip count:" << hizTexture.GetMipLevelCount()
              << std::endl;

    wgpu::TextureViewDescriptor viewDesc{};
    viewDesc.baseMipLevel = 0;
    viewDesc.mipLevelCount = mipCount;
    viewDesc.dimension = wgpu::TextureViewDimension::e2D;
    viewDesc.format = wgpu::TextureFormat::R32Float;

    hizBaseView = hizTexture.CreateView(&viewDesc);
  }

  std::vector<Vertex> vertices{};
  std::vector<uint32_t> indices{};
  loadGLTF("assets/Duck.glb", vertices, indices);

  std::vector<Meshlet> meshlets{};
  std::vector<uint32_t> meshletVertices{};
  std::vector<uint32_t> meshletTriangles{};
  buildMeshlets(vertices, indices, meshlets, meshletVertices, meshletTriangles);

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "scene buffer";
    desc.size = sizeof(SceneData);
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    sceneBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    wgpu::BufferDescriptor indirectDesc{
        .label = "Indirect Draw Buffer",
        .usage = wgpu::BufferUsage::Indirect | wgpu::BufferUsage::Storage |
                 wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc,
        .size = sizeof(uint32_t) * 4};

    indirectBuffer = ctx.device.CreateBuffer(&indirectDesc);
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "vertexBuffer";
    desc.size = sizeof(Vertex) * vertices.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    vertexBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(vertexBuffer, 0, vertices.data(),
                          vertexBuffer.GetSize());
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "meshletBuffer";
    desc.size = sizeof(Meshlet) * meshlets.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meshletBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(meshletBuffer, 0, meshlets.data(),
                          meshletBuffer.GetSize());
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "meshletVertexBuffer";
    desc.size = sizeof(uint32_t) * meshletVertices.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meshletVertexBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(meshletVertexBuffer, 0, meshletVertices.data(),
                          meshletVertexBuffer.GetSize());
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "meshletTriangleBuffer";
    desc.size = sizeof(uint32_t) * meshletTriangles.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meshletTriangleBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(meshletTriangleBuffer, 0, meshletTriangles.data(),
                          meshletTriangleBuffer.GetSize());
  }

  //
  std::vector<MeshInstance> meshInstances;
  {
    glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));
    uint32_t instanceCount = 5000;
    float spacing = 3.0f;
    int gridSize =
        static_cast<int>(std::ceil(std::pow(instanceCount, 1.0 / 3.0)));

    for (int x = 0; x < gridSize; x++) {
      for (int y = 0; y < gridSize; y++) {
        for (int z = 0; z < gridSize; z++) {
          glm::vec3 position = glm::vec3(x * spacing, y * spacing, z * spacing);

          // glm::mat4 model =
          //     glm::translate(glm::mat4(1.0f), position) * scaleMat;
          // x, y, z, w)
          meshInstances.push_back(
              {.position = position,
               .scale = 0.02f,
               .orientation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)});

          if (meshInstances.size() >= instanceCount)
            break;
        }
        if (meshInstances.size() >= instanceCount)
          break;
      }
      if (meshInstances.size() >= instanceCount)
        break;
    }

    wgpu::BufferDescriptor desc{};
    desc.label = "meshInstanceBuffer";
    desc.size = sizeof(MeshInstance) * meshInstances.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meshInstanceBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(meshInstanceBuffer, 0, meshInstances.data(),
                          meshInstanceBuffer.GetSize());
  }

  {
    std::vector<MeshletInstance> meshletInstances;

    for (uint32_t j = 0; j < meshInstances.size(); j++) {
      for (uint32_t i = 0; i < meshlets.size(); i++) {
        meshletInstances.push_back({.meshletID = i, .instanceID = j});
      }
    }

    wgpu::BufferDescriptor desc{};
    desc.label = "meshletInstanceBuffer";
    desc.size = sizeof(MeshletInstance) * meshletInstances.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meshletInstanceBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(meshletInstanceBuffer, 0, meshletInstances.data(),
                          meshletInstanceBuffer.GetSize());

    meshletCount = meshletInstances.size();
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "visibleMeshletBuffer";
    desc.size = meshletInstanceBuffer.GetSize();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    visibleMeshletBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "prevVisibleMeshletBuffer";
    desc.size = meshletInstanceBuffer.GetSize();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    prevVisibleMeshletBuffer = ctx.device.CreateBuffer(&desc);
  }

  maxMeshletTriangleCount = 0;
  for (auto &meshlet : meshlets) {
    maxMeshletTriangleCount =
        std::max(meshlet.triangleCount, maxMeshletTriangleCount);
    trianglesCount += meshlet.triangleCount;
  }

  std::cout << "avgMeshletTriangleCount: " << trianglesCount / meshlets.size()
            << std::endl;
  std::cout << "maxMeshletTriangleCount: " << maxMeshletTriangleCount
            << std::endl;
  std::cout << "meshletCount :" << meshletCount << std::endl;
  std::cout << "vertices count :" << vertices.size() << std::endl;
  std::cout << "triangles count :" << trianglesCount * meshInstances.size()
            << std::endl;
  std::cout << "rendered triangles count :"
            << maxMeshletTriangleCount * meshInstances.size() << std::endl;

  initCullPipeline();
  initDepthCopyPipeline();
  initHizMipsPipeline();
  initRenderPipline();

  {
    wgpu::BufferDescriptor desc{};
    desc.size = indirectBuffer.GetSize();
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

    readbackBuffer = ctx.device.CreateBuffer(&desc);
  }
}

void Renderer::initDepthCopyPipeline() {

  wgpu::BindGroupLayoutEntry entries[2] = {};

  // srcDepth
  entries[0].binding = 0;
  entries[0].visibility = wgpu::ShaderStage::Compute;
  entries[0].texture.sampleType = wgpu::TextureSampleType::Depth;

  // dstMip0
  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Compute;
  entries[1].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
  entries[1].storageTexture.format = wgpu::TextureFormat::R32Float;

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 2, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  wgpu::BindGroupEntry bgEntries[2] = {};
  bgEntries[0].binding = 0;
  bgEntries[0].textureView = depthView;

  //
  bgEntries[1].binding = 1;
  bgEntries[1].textureView = hizViews[0];
  //

  auto bgd = wgpu::BindGroupDescriptor{.label = "copy depth bindgroup",
                                       .layout = bgl,
                                       .entryCount = 2,
                                       .entries = bgEntries};
  depthCopyBindGroup = ctx.device.CreateBindGroup(&bgd);

  const char *shaderWGSL = R"(
@group(0) @binding(0)
var srcDepth : texture_depth_2d;

@group(0) @binding(1)
var dstMip0 : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {

    let coord = vec2<i32>(id.xy);

    let depth = textureLoad(srcDepth, coord, 0);

    textureStore(dstMip0, coord, vec4<f32>(depth));
}

  )";

  wgpu::ShaderSourceWGSL wgslDesc{};
  wgslDesc.code = shaderWGSL;
  wgpu::ShaderModuleDescriptor shaderDesc = wgpu::ShaderModuleDescriptor{
      .nextInChain = &wgslDesc, .label = "depth copy shader"};
  wgpu::ShaderModule shaderModule = ctx.device.CreateShaderModule(&shaderDesc);

  wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {
      .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl};

  wgpu::PipelineLayout pipelineLayout =
      ctx.device.CreatePipelineLayout(&pipelineLayoutDesc);

  wgpu::ComputePipelineDescriptor cpDesc = {

      .label = "depth copy pipeline",
      .layout = pipelineLayout,
      .compute =
          {
              .module = shaderModule,
              .entryPoint = "main",
          },

  };

  depthCopyPipeline = ctx.device.CreateComputePipeline(&cpDesc);
}

void Renderer::initHizMipsPipeline() {

  wgpu::BindGroupLayoutEntry entries[2] = {};

  // srcDepth
  entries[0].binding = 0;
  entries[0].visibility = wgpu::ShaderStage::Compute;
  entries[0].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;

  // dstMip0
  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Compute;
  entries[1].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
  entries[1].storageTexture.format = wgpu::TextureFormat::R32Float;

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 2, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  hizGenBindGroups.reserve(hizViews.size() - 1);
  for (uint32_t i = 1; i < hizViews.size(); i++) {

    wgpu::BindGroupEntry bgEntries[2] = {};
    bgEntries[0].binding = 0;
    bgEntries[0].textureView = hizViews[i - 1];

    //
    bgEntries[1].binding = 1;
    bgEntries[1].textureView = hizViews[i];
    //

    auto bgd = wgpu::BindGroupDescriptor{.label = "hiz mips gen bindgroup",
                                         .layout = bgl,
                                         .entryCount = 2,
                                         .entries = bgEntries};
    hizGenBindGroups.push_back(ctx.device.CreateBindGroup(&bgd));
  }

  const char *shaderWGSL = R"(
@group(0) @binding(0)
var srcMip : texture_2d<f32>;

@group(0) @binding(1)
var dstMip : texture_storage_2d<r32float, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {

    let dst = id.xy;
    let src = dst * 2u;

    let d0 = textureLoad(srcMip, vec2<i32>(src), 0).x;
    let d1 = textureLoad(srcMip, vec2<i32>(src + vec2<u32>(1,0)), 0).x;
    let d2 = textureLoad(srcMip, vec2<i32>(src + vec2<u32>(0,1)), 0).x;
    let d3 = textureLoad(srcMip, vec2<i32>(src + vec2<u32>(1,1)), 0).x;

    let maxDepth = max(max(d0,d1), max(d2,d3));

    textureStore(dstMip, vec2<i32>(dst), vec4<f32>(maxDepth));
}

  )";

  wgpu::ShaderSourceWGSL wgslDesc{};
  wgslDesc.code = shaderWGSL;
  wgpu::ShaderModuleDescriptor shaderDesc = wgpu::ShaderModuleDescriptor{
      .nextInChain = &wgslDesc, .label = "hiz gen shader"};
  wgpu::ShaderModule shaderModule = ctx.device.CreateShaderModule(&shaderDesc);

  wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {
      .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl};

  wgpu::PipelineLayout pipelineLayout =
      ctx.device.CreatePipelineLayout(&pipelineLayoutDesc);

  wgpu::ComputePipelineDescriptor cpDesc = {

      .label = "hiz gen pipeline",
      .layout = pipelineLayout,
      .compute =
          {
              .module = shaderModule,
              .entryPoint = "main",
          },

  };

  hizGenPipeline = ctx.device.CreateComputePipeline(&cpDesc);
}

void Renderer::initRenderPipline() {

  wgpu::BindGroupLayoutEntry entries[8] = {};

  // scene
  entries[0].binding = 0;
  entries[0].visibility =
      wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
  entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
  entries[0].buffer.minBindingSize = sizeof(SceneData);

  // vertex
  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Vertex;
  entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[1].buffer.minBindingSize = vertexBuffer.GetSize();

  // meshlet
  entries[2].binding = 2;
  entries[2].visibility = wgpu::ShaderStage::Vertex;
  entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[2].buffer.minBindingSize = meshletBuffer.GetSize();

  // meshletvertex
  entries[3].binding = 3;
  entries[3].visibility = wgpu::ShaderStage::Vertex;
  entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[3].buffer.minBindingSize = meshletVertexBuffer.GetSize();

  // meshlettriangle
  entries[4].binding = 4;
  entries[4].visibility = wgpu::ShaderStage::Vertex;
  entries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[4].buffer.minBindingSize = meshletTriangleBuffer.GetSize();

  // meshletInstances
  entries[5].binding = 5;
  entries[5].visibility = wgpu::ShaderStage::Vertex;
  entries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[5].buffer.minBindingSize = meshletInstanceBuffer.GetSize();

  // meshInstances
  entries[6].binding = 6;
  entries[6].visibility = wgpu::ShaderStage::Vertex;
  entries[6].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[6].buffer.minBindingSize = meshInstanceBuffer.GetSize();

  // visibleMeshlets
  entries[7].binding = 7;
  entries[7].visibility = wgpu::ShaderStage::Vertex;
  entries[7].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[7].buffer.minBindingSize = visibleMeshletBuffer.GetSize();

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 8, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  wgpu::BindGroupEntry bgEntries[8] = {};
  bgEntries[0].binding = 0;
  bgEntries[0].buffer = sceneBuffer;
  bgEntries[0].offset = 0;
  bgEntries[0].size = sizeof(SceneData);
  //
  bgEntries[1].binding = 1;
  bgEntries[1].buffer = vertexBuffer;
  bgEntries[1].offset = 0;
  bgEntries[1].size = vertexBuffer.GetSize();
  //
  bgEntries[2].binding = 2;
  bgEntries[2].buffer = meshletBuffer;
  bgEntries[2].offset = 0;
  bgEntries[2].size = meshletBuffer.GetSize();

  bgEntries[3].binding = 3;
  bgEntries[3].buffer = meshletVertexBuffer;
  bgEntries[3].offset = 0;
  bgEntries[3].size = meshletVertexBuffer.GetSize();

  bgEntries[4].binding = 4;
  bgEntries[4].buffer = meshletTriangleBuffer;
  bgEntries[4].offset = 0;
  bgEntries[4].size = meshletTriangleBuffer.GetSize();

  bgEntries[5].binding = 5;
  bgEntries[5].buffer = meshletInstanceBuffer;
  bgEntries[5].offset = 0;
  bgEntries[5].size = meshletInstanceBuffer.GetSize();

  bgEntries[6].binding = 6;
  bgEntries[6].buffer = meshInstanceBuffer;
  bgEntries[6].offset = 0;
  bgEntries[6].size = meshInstanceBuffer.GetSize();

  bgEntries[7].binding = 7;
  bgEntries[7].buffer = visibleMeshletBuffer;
  bgEntries[7].offset = 0;
  bgEntries[7].size = visibleMeshletBuffer.GetSize();

  auto bgd = wgpu::BindGroupDescriptor{
      .layout = bgl, .entryCount = 8, .entries = bgEntries};
  renderBindGroup = ctx.device.CreateBindGroup(&bgd);

  /////

  auto shaderWGSL = R"(

struct SceneData {
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  cameraPos: vec3<f32>, 
  _pad0: f32,
  aspect: f32,
  fov: f32,// glm::tan(glm::radians(camera.fov * 0.5));
  nearFar: vec2<f32>,
  screenHeight: f32,
  hizMipCount: u32,
  hasPrevDepth:u32,
  _pad1:u32,  
};


struct Vertex {
    pos : vec3<f32>,
    pad:f32
};

struct Meshlet {
    vertexOffset   : u32,
    triangleOffset : u32,
    vertexCount    : u32,
    triangleCount  : u32,
    boundingSphere0 : vec2f,
    boundingSphere1 : vec2f,
    boundingCone   : u32,
    color:u32, 
};

struct MeshInstance {
  position:vec3f,
  scale:f32,
  orientation:vec4f
};

struct MeshletInstance {
    meshletID: u32,
    meshInstanceID: u32,
};

@group(0) @binding(0) var<uniform> scene : SceneData;

@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var<storage, read> meshlets : array<Meshlet>;
@group(0) @binding(3) var<storage, read> meshletVertices : array<u32>;
@group(0) @binding(4) var<storage, read> meshletTriangles : array<u32>;

@group(0) @binding(5) var<storage, read> meshletInstances: array<MeshletInstance>;
@group(0) @binding(6) var<storage, read> meshInstances: array<MeshInstance>;
@group(0) @binding(7) var<storage, read> visibleMeshlets: array<u32>;



struct VSOut {
    @builtin(position) pos : vec4<f32>,
     @location(0) color : vec3<f32>,
};

fn hash32(x: u32) -> u32 {
    var v = x;
    v = (v ^ (v >> 16u)) * 0x7feb352du;
    v = (v ^ (v >> 15u)) * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn rand3(meshletID: u32) -> vec3<f32> {
    let h = hash32(meshletID);
    let r = f32((h & 0xFFu)) / 255.0;
    let g = f32((h >> 8u) & 0xFFu) / 255.0;
    let b = f32((h >> 16u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b);
}

fn rotateQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vtx : u32,
    @builtin(instance_index) meshletID : u32
) -> VSOut {

  let visibleId = visibleMeshlets[meshletID]; 
  let meshletInstance= meshletInstances[visibleId];

  let meshlet= meshlets[meshletInstance.meshletID];
  let mesh= meshInstances[meshletInstance.meshInstanceID];


  
  let maxVtx = meshlet.triangleCount * 3u; 
  
   if (vtx >= maxVtx) { var o : VSOut; o.pos = vec4<f32>(0.0, 0.0, 0.0, 0.0); return o; } 
  
  let triangleVertexIndex : u32 = meshletTriangles[meshlet.triangleOffset + vtx]; 
  
  let vertexIndex : u32 = meshletVertices[meshlet.vertexOffset + triangleVertexIndex]; 
  
  let v = vertices[vertexIndex];

    var out : VSOut;
    out.pos = scene.proj* scene.view * vec4<f32>( rotateQuat(v.pos,mesh.orientation)*mesh.scale + mesh.position, 1.0);
    
    out.color =vec3f( f32(meshlet.color&0xff)/255.0 , f32((meshlet.color>>8)&0xff)/255. ,f32((meshlet.color>>16)&0xff)/255.0);
    return out;
}

@fragment
fn fs_main(@location(0) color : vec3<f32>) -> @location(0) vec4<f32> {
 return vec4<f32>(color, 1.0);
}


  )";
  wgpu::ShaderSourceWGSL wgslDesc = {};
  wgslDesc.code = shaderWGSL;
  // shader modules
  auto desc = wgpu::ShaderModuleDescriptor{.nextInChain = &wgslDesc};
  auto shaderModule = ctx.device.CreateShaderModule(&desc);

  // pipeline layout

  wgpu::PipelineLayoutDescriptor plDesc = {
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &bgl,
  };
  wgpu::PipelineLayout pipelineLayout =
      ctx.device.CreatePipelineLayout(&plDesc);

  // render pipeline

  wgpu::ColorTargetState colorTarget = {};
  colorTarget.format = ctx.backbufferFormat;

  wgpu::BlendState blend = {
      .color{.operation = wgpu::BlendOperation::Add,
             .srcFactor = wgpu::BlendFactor::SrcAlpha,
             .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},

      .alpha = wgpu::BlendComponent{
          .operation = wgpu::BlendOperation::Add,
          .srcFactor = wgpu::BlendFactor::One,
          .dstFactor = wgpu::BlendFactor::Zero,
      }};

  colorTarget.blend = &blend;

  wgpu::FragmentState fragment = {};
  fragment.module = shaderModule;
  fragment.entryPoint = "fs_main";
  fragment.targetCount = 1;
  fragment.targets = &colorTarget;

  wgpu::RenderPipelineDescriptor rpDesc = {.label = "meshletRender"};
  rpDesc.layout = pipelineLayout;
  rpDesc.vertex = {
      .module = shaderModule, .entryPoint = "vs_main", .buffers = {}};
  rpDesc.fragment = &fragment;

  wgpu::DepthStencilState depthState{
      .format = wgpu::TextureFormat::Depth24Plus,
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::LessEqual,
  };
  rpDesc.depthStencil = &depthState;
  rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  rpDesc.primitive.frontFace = wgpu::FrontFace::CCW;
  rpDesc.primitive.cullMode = wgpu::CullMode::Back;

  renderPipeline = ctx.device.CreateRenderPipeline(&rpDesc);

  //
  rpDesc.label = "zpass";
  rpDesc.fragment = nullptr;

  zpassPipeline = ctx.device.CreateRenderPipeline(&rpDesc);
}

void Renderer::initCullPipeline() {

  wgpu::BindGroupLayoutEntry entries[11] = {};

  // scene
  entries[0].binding = 0;
  entries[0].visibility = wgpu::ShaderStage::Compute;
  entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
  entries[0].buffer.minBindingSize = sizeof(SceneData);

  // vertex
  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Compute;
  entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[1].buffer.minBindingSize = vertexBuffer.GetSize();

  // meshlet
  entries[2].binding = 2;
  entries[2].visibility = wgpu::ShaderStage::Compute;
  entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[2].buffer.minBindingSize = meshletBuffer.GetSize();

  // meshletvertex
  entries[3].binding = 3;
  entries[3].visibility = wgpu::ShaderStage::Compute;
  entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[3].buffer.minBindingSize = meshletVertexBuffer.GetSize();

  // meshlettriangle
  entries[4].binding = 4;
  entries[4].visibility = wgpu::ShaderStage::Compute;
  entries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[4].buffer.minBindingSize = meshletTriangleBuffer.GetSize();

  // meshletInstances
  entries[5].binding = 5;
  entries[5].visibility = wgpu::ShaderStage::Compute;
  entries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[5].buffer.minBindingSize = meshletInstanceBuffer.GetSize();

  // meshInstances
  entries[6].binding = 6;
  entries[6].visibility = wgpu::ShaderStage::Compute;
  entries[6].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[6].buffer.minBindingSize = meshInstanceBuffer.GetSize();

  // visibleMeshlets
  entries[7].binding = 7;
  entries[7].visibility = wgpu::ShaderStage::Compute;
  entries[7].buffer.type = wgpu::BufferBindingType::Storage;
  entries[7].buffer.minBindingSize = visibleMeshletBuffer.GetSize();

  // drawArgs
  entries[8].binding = 8;
  entries[8].visibility = wgpu::ShaderStage::Compute;
  entries[8].buffer.type = wgpu::BufferBindingType::Storage;
  entries[8].buffer.minBindingSize = indirectBuffer.GetSize();

  // hizTex
  entries[9].binding = 9;
  entries[9].visibility = wgpu::ShaderStage::Compute;
  entries[9].texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;

  // prevVisibleMeshlet
  entries[10].binding = 10;
  entries[10].visibility = wgpu::ShaderStage::Compute;
  entries[10].buffer.type = wgpu::BufferBindingType::Storage;
  entries[10].buffer.minBindingSize = visibleMeshletBuffer.GetSize();

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 11, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  wgpu::BindGroupEntry bgEntries[11] = {};
  bgEntries[0].binding = 0;
  bgEntries[0].buffer = sceneBuffer;
  bgEntries[0].offset = 0;
  bgEntries[0].size = sizeof(SceneData);
  //
  bgEntries[1].binding = 1;
  bgEntries[1].buffer = vertexBuffer;
  bgEntries[1].offset = 0;
  bgEntries[1].size = vertexBuffer.GetSize();
  //
  bgEntries[2].binding = 2;
  bgEntries[2].buffer = meshletBuffer;
  bgEntries[2].offset = 0;
  bgEntries[2].size = meshletBuffer.GetSize();

  bgEntries[3].binding = 3;
  bgEntries[3].buffer = meshletVertexBuffer;
  bgEntries[3].offset = 0;
  bgEntries[3].size = meshletVertexBuffer.GetSize();

  bgEntries[4].binding = 4;
  bgEntries[4].buffer = meshletTriangleBuffer;
  bgEntries[4].offset = 0;
  bgEntries[4].size = meshletTriangleBuffer.GetSize();

  bgEntries[5].binding = 5;
  bgEntries[5].buffer = meshletInstanceBuffer;
  bgEntries[5].offset = 0;
  bgEntries[5].size = meshletInstanceBuffer.GetSize();

  bgEntries[6].binding = 6;
  bgEntries[6].buffer = meshInstanceBuffer;
  bgEntries[6].offset = 0;
  bgEntries[6].size = meshInstanceBuffer.GetSize();

  bgEntries[7].binding = 7;
  bgEntries[7].buffer = visibleMeshletBuffer;
  bgEntries[7].offset = 0;
  bgEntries[7].size = visibleMeshletBuffer.GetSize();

  bgEntries[8].binding = 8;
  bgEntries[8].buffer = indirectBuffer;
  bgEntries[8].offset = 0;
  bgEntries[8].size = indirectBuffer.GetSize();

  bgEntries[9].binding = 9;
  bgEntries[9].textureView = hizBaseView;

  bgEntries[10].binding = 10;
  bgEntries[10].buffer = prevVisibleMeshletBuffer;
  bgEntries[10].offset = 0;
  bgEntries[10].size = prevVisibleMeshletBuffer.GetSize();

  auto bgd = wgpu::BindGroupDescriptor{.label = "culling bindgroup",
                                       .layout = bgl,
                                       .entryCount = 11,
                                       .entries = bgEntries};
  cullingBindGroup = ctx.device.CreateBindGroup(&bgd);

  const char *shaderWGSL = R"(

struct SceneData {
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  cameraPos: vec3<f32>, 
  _pad0: f32,
  aspect: f32,
  fov: f32,// glm::tan(glm::radians(camera.fov * 0.5));
  nearFar: vec2<f32>,
  screenHeight: f32,
  hizMipCount: u32,
  hasPrevDepth : u32,
  late: u32,  
};


struct Vertex {
    pos : vec3<f32>,
    pad:f32
};


struct Meshlet {
    vertexOffset   : u32,
    triangleOffset : u32,
    vertexCount    : u32,
    triangleCount  : u32,
    boundingSphere0 : vec2f,
    boundingSphere1 : vec2f,
    boundingCone   : u32,
    color:u32, 
};

struct MeshInstance {
  position:vec3f,
  scale:f32,
  orientation:vec4f
};

struct MeshletInstance {
    meshletID: u32,
    meshInstanceID: u32,
};

  struct DrawIndirectArgs {
      vertexCount   : u32,
      instanceCount : atomic<u32>,
      firstVertex   : u32,
      firstInstance : u32,
  };

@group(0) @binding(0) var<uniform> scene : SceneData;

@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var<storage, read> meshlets : array<Meshlet>;
@group(0) @binding(3) var<storage, read> meshletVertices : array<u32>;
@group(0) @binding(4) var<storage, read> meshletTriangles : array<u32>;

@group(0) @binding(5) var<storage, read> meshletInstances: array<MeshletInstance>;
@group(0) @binding(6) var<storage, read> meshInstances: array<MeshInstance>;
@group(0) @binding(7) var<storage, read_write> visibleMeshlets: array<u32>;
@group(0) @binding(8) var<storage, read_write> drawArgs: DrawIndirectArgs;

@group(0) @binding(9) var hizTex: texture_2d<f32>;
@group(0) @binding(10) var<storage, read_write> prevVisibleMeshlet: array<u32>;



fn sphereInViewFrustum(center: vec3<f32>, radius: f32) -> bool {

    let z =-center.z;
    //(near/far)
    if (z + radius < scene.nearFar.x) { return false; }
    if (z - radius > scene.nearFar.y)  { return false; }

    // (top/bottom)
    let y_limit = z * scene.fov;
    if (center.y - radius > y_limit) { return false; }
    if (center.y + radius < -y_limit){ return false; }

    //(left/right)
    let x_limit = y_limit * scene.aspect;
    if (center.x - radius > x_limit) { return false; }
    if (center.x + radius < -x_limit){ return false; }

    return true; 
}

//https://zeux.io/2023/01/12/approximate-projected-bounds/
fn projectSphereView(
    c : vec3f,
    r : f32,
    znear : f32,
    P00 : f32,
    P11 : f32,
    aabbOut:  ptr<function,vec4f>
) -> bool
{
    if (c.z < r + znear) {
        return false;
    }

    let cr = c * r;
    let czr2 = c.z * c.z - r * r;

    let vx = sqrt(c.x * c.x + czr2);
    let minx = (vx * c.x - cr.z) / (vx * c.z + cr.x);
    let maxx = (vx * c.x + cr.z) / (vx * c.z - cr.x);

    let vy = sqrt(c.y * c.y + czr2);
    let miny = (vy * c.y - cr.z) / (vy * c.z + cr.y);
    let maxy = (vy * c.y + cr.z) / (vy * c.z - cr.y);

    var aabb = vec4f(
        minx * P00,
        miny * P11,
        maxx * P00,
        maxy * P11
    );

    // clip space -> uv space
    *aabbOut = aabb.xwzy * vec4f(0.5, -0.5, 0.5, -0.5) + vec4f(0.5);

    return true; 
}


fn coneCull(center: vec3f,
            radius: f32,
            axis: vec3f,
            cutoff: f32,
            cameraPos: vec3f) -> bool
{
    let v =  center-cameraPos;
    let dist = length(v);

    return dot(v, axis) >= cutoff * dist -radius ;
}

fn rotateQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}


@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgID : vec3<u32>,
    @builtin(local_invocation_id) localID : vec3<u32>,
    @builtin(num_workgroups) numWG : vec3<u32>) {

    let groupIndex =
        wgID.y * numWG.x + wgID.x;

    let meshletIndex =
        groupIndex * 64u + localID.x;

    if (meshletIndex >= arrayLength(&meshletInstances)) {
        return;
    }

    // first pass processes only previously visible meshlets
    if (scene.hasPrevDepth==1&& 
        scene.late == 0u &&
        prevVisibleMeshlet[meshletIndex] == 0u)
    {
        return;
    }

    var visible = true;

    if (scene.hasPrevDepth==1){

      let instance = meshletInstances[meshletIndex];

      let m = meshlets[instance.meshletID];
      let mesh =meshInstances[instance.meshInstanceID];

      var center = vec3f(m.boundingSphere0,m.boundingSphere1.x);
    
      let center_world = rotateQuat(center,mesh.orientation)* mesh.scale + mesh.position;
      let center_view = scene.view * vec4f(center_world,1.0);
      
      let radius = m.boundingSphere1.y*mesh.scale;

      var axis_unpacked: vec4<f32> = unpack4x8snorm(m.boundingCone);
      var axis = rotateQuat(axis_unpacked.xyz,mesh.orientation);
      axis = (scene.view * vec4(axis, 0.0)).xyz;
      
      let cutoff =axis_unpacked.w;

      //backface culling
     visible = !coneCull(center_view.xyz,
                               radius,
                               axis,
                               cutoff,
                               vec3f(0.0));
        

      // frustum culling
      if (!sphereInViewFrustum(center_view.xyz, radius)) {
        visible = false;
      }

      // occlusion culling
      var aabb : vec4<f32>;
      if(projectSphereView(vec3f(center_view.x,center_view.y,-center_view.z),radius,scene.nearFar.x,scene.proj[0][0],scene.proj[1][1],
      &aabb)){

        let baseSize = textureDimensions(hizTex, 0);
        let width  = (aabb.z - aabb.x) * f32(baseSize.x);
        let height = (aabb.w - aabb.y) * f32(baseSize.y);

        var mip = u32(floor(log2(max(width, height))));
        mip = min(mip, u32(scene.hizMipCount - 1));

        let mipSize = textureDimensions(hizTex, i32(mip));

        let minCoord = vec2u(aabb.xy * vec2f(mipSize));
        let maxCoord = vec2u(aabb.zw * vec2f(mipSize));

        //sample another 4 corners
        var depthMax: f32 =  textureLoad(hizTex, vec2i(vec2f(minCoord + maxCoord)*0.5), i32(mip)).x;
        depthMax = max(depthMax,textureLoad(hizTex, vec2i(minCoord), i32(mip)).x);
        depthMax = max(depthMax, textureLoad(hizTex, vec2i(vec2u(maxCoord.x, minCoord.y)), i32(mip)).x);
        depthMax = max(depthMax, textureLoad(hizTex, vec2i(vec2u(minCoord.x, maxCoord.y)), i32(mip)).x);
        depthMax = max(depthMax, textureLoad(hizTex, vec2i(maxCoord), i32(mip)).x);

        let dir = normalize(center_world.xyz-scene.cameraPos);
        let sphereFront = center_world.xyz + dir * radius;

        let clip = scene.proj* scene.view * vec4<f32>(sphereFront, 1.0);
        let sphereDepth = clip.z / clip.w;

        if (sphereDepth > depthMax + 0.0009){
          visible=false; 
        }

      }
    
    }

    var append = false;
    if (scene.hasPrevDepth == 0u) {
        append = true;
    }
    else {
        
        // first pass : reused meshlets
        if (scene.late == 0u) {
            append = visible;
        }

        // 2nd pass : newly visible meshlets
        if (scene.late == 1u &&
            prevVisibleMeshlet[meshletIndex] == 0u)
        {
            append = visible;
        }
    }

    if (append) {
        let instanceIndex =
            atomicAdd(&drawArgs.instanceCount, 1u);

        visibleMeshlets[instanceIndex]=meshletIndex;
    }

    if (scene.late == 1u)
    {
        prevVisibleMeshlet[meshletIndex] =
            select(0u, 1u, visible);
    }
 
}

  )";

  wgpu::ShaderSourceWGSL wgslDesc{};
  wgslDesc.code = shaderWGSL;
  wgpu::ShaderModuleDescriptor shaderDesc = wgpu::ShaderModuleDescriptor{
      .nextInChain = &wgslDesc, .label = "culling"};
  wgpu::ShaderModule shaderModule = ctx.device.CreateShaderModule(&shaderDesc);

  wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {
      .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl};

  wgpu::PipelineLayout pipelineLayout =
      ctx.device.CreatePipelineLayout(&pipelineLayoutDesc);

  wgpu::ComputePipelineDescriptor cpDesc = {

      .label = "cullingpipeline",
      .layout = pipelineLayout,
      .compute =
          {
              .module = shaderModule,
              .entryPoint = "main",
          },

  };

  cullingPipeline = ctx.device.CreateComputePipeline(&cpDesc);
}

void Renderer::render(const FlyCamera &camera, float time) {
  // Update data
  SceneData scene;
  scene.proj = camera.getProjectionMatrix();

  // glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));
  scene.view = camera.getViewMatrix();
  scene.camPos = camera.position;
  scene.aspect = camera.aspect;
  scene.fov = glm::tan(glm::radians(camera.fov * 0.5));
  scene.nearFar = glm::vec2(camera.nearPlane, camera.farPlane);
  scene.screenHeight = ctx.height;
  scene.hizMipCount = hizTexture.GetMipLevelCount();
  scene.hasPrevDepth = hasPrevDepth;
  scene.late = 0;

  ctx.queue.WriteBuffer(sceneBuffer, 0, &scene, sizeof(SceneData));

  {
    uint32_t zero[] = {maxMeshletTriangleCount * 3, 0, 0, 0};
    ctx.queue.WriteBuffer(indirectBuffer, 0, zero, 4 * sizeof(uint32_t));
  }

  wgpu::SurfaceTexture surfaceTexture;
  ctx.surface.GetCurrentTexture(&surfaceTexture);
  wgpu::TextureView view = surfaceTexture.texture.CreateView();

  wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();

  // cull using prev hiz
  {
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(cullingPipeline);
    pass.SetBindGroup(0, cullingBindGroup);
    uint32_t totalGroups = (meshletCount + 63) / 64;
    uint32_t groupsX = std::min(totalGroups, 65535u);
    uint32_t groupsY = (totalGroups + 65534u) / 65535u;

    pass.DispatchWorkgroups(groupsX, groupsY, 1);
    pass.End();
  }

  // zpass
  {
    wgpu::RenderPassDepthStencilAttachment depthAttachment{
        .view = depthView,
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 1.0f};

    wgpu::RenderPassDescriptor passDesc{.colorAttachmentCount = 0,
                                        .colorAttachments = nullptr,
                                        .depthStencilAttachment =
                                            &depthAttachment

    };
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(zpassPipeline);
    pass.SetBindGroup(0, renderBindGroup);
    pass.DrawIndirect(indirectBuffer, 0);
    pass.End();
  }

  // generate hiz mips
  {

    // copy depth to hiz 0
    {
      wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
      pass.SetPipeline(depthCopyPipeline);
      pass.SetBindGroup(0, depthCopyBindGroup);

      uint32_t w = (ctx.width + 7) / 8;
      uint32_t h = (ctx.height + 7) / 8;
      pass.DispatchWorkgroups(w, h, 1);
      pass.End();
    }
    for (uint32_t mip = 1; mip < hizViews.size(); mip++) {
      wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
      uint32_t mipWidth = std::max(1u, ctx.width >> mip);
      uint32_t mipHeight = std::max(1u, ctx.height >> mip);

      uint32_t groupsX = (mipWidth + 7) / 8;
      uint32_t groupsY = (mipHeight + 7) / 8;

      pass.SetPipeline(hizGenPipeline);

      pass.SetBindGroup(0, hizGenBindGroups[mip - 1]);

      pass.DispatchWorkgroups(groupsX, groupsY);

      pass.End();
    }
  }

  wgpu::CommandBuffer cmd = encoder.Finish();
  ctx.queue.Submit(1, &cmd);

  wgpu::CommandEncoder encoder2 = ctx.device.CreateCommandEncoder();
  // cull using the new hiz
  {
    scene.late = true;
    ctx.queue.WriteBuffer(sceneBuffer, 0, &scene, sizeof(SceneData));

    wgpu::ComputePassEncoder pass = encoder2.BeginComputePass();
    pass.SetPipeline(cullingPipeline);
    pass.SetBindGroup(0, cullingBindGroup);
    uint32_t totalGroups = (meshletCount + 63) / 64;
    uint32_t groupsX = std::min(totalGroups, 65535u);
    uint32_t groupsY = (totalGroups + 65534u) / 65535u;

    pass.DispatchWorkgroups(groupsX, groupsY, 1);
    pass.End();
  }

  // final render
  {
    wgpu::RenderPassColorAttachment colorAttachment{
        .view = view,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = wgpu::Color{0.1, 0.2, 0.4, 0.0}};

    wgpu::RenderPassDepthStencilAttachment depthAttachment{
        .view = depthView,
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 1.0f};

    wgpu::RenderPassDescriptor passDesc{.colorAttachmentCount = 1,
                                        .colorAttachments = &colorAttachment,
                                        .depthStencilAttachment =
                                            &depthAttachment

    };
    wgpu::RenderPassEncoder pass = encoder2.BeginRenderPass(&passDesc);
    pass.SetPipeline(renderPipeline);
    pass.SetBindGroup(0, renderBindGroup);
    pass.DrawIndirect(indirectBuffer, 0);
    pass.End();
  }

  hasPrevDepth = true;

  //
  if (!readbackPending) {
    encoder2.CopyBufferToBuffer(indirectBuffer, 0, readbackBuffer, 0,
                                indirectBuffer.GetSize());
  }

  // gui pass
  renderGui(encoder2, view);

  wgpu::CommandBuffer cmd1 = encoder2.Finish();
  ctx.queue.Submit(1, &cmd1);

  if (!readbackPending) {
    readbackPending = true;

    readbackBuffer.MapAsync(
        wgpu::MapMode::Read, 0, indirectBuffer.GetSize(),
        wgpu::CallbackMode::AllowSpontaneous,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView message) {
          if (status != wgpu::MapAsyncStatus::Success)
            return;

          const void *data =
              readbackBuffer.GetConstMappedRange(0, indirectBuffer.GetSize());
          const uint32_t *indirect = static_cast<const uint32_t *>(data);

          renderedMeshletsCount = indirect[1];
          readbackBuffer.Unmap();
          readbackPending = false;
        });
  }

#ifndef EMSCRIPTEN
  ctx.surface.Present();
  ctx.instance.ProcessEvents();
#endif
}

void Renderer::initGui(GLFWwindow *window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOther(window, true);

  // Setup Renderer Backend (WebGPU)
  ImGui_ImplWGPU_InitInfo init_info;
  init_info.Device = ctx.device.Get();
  init_info.NumFramesInFlight = 3;
  init_info.RenderTargetFormat =
      static_cast<WGPUTextureFormat>(ctx.backbufferFormat);
  init_info.DepthStencilFormat = WGPUTextureFormat_Undefined;

  ImGui_ImplWGPU_Init(&init_info);

  ImGuiIO &io = ImGui::GetIO();
  io.DisplaySize = ImVec2(ctx.width, ctx.height);
}
void Renderer::renderGui(wgpu::CommandEncoder &encoder,
                         wgpu::TextureView &view) {

  wgpu::RenderPassColorAttachment colorAttachment{.view = view,
                                                  .loadOp = wgpu::LoadOp::Load,
                                                  .storeOp =
                                                      wgpu::StoreOp::Store,
                                                  .clearValue = wgpu::Color{
                                                      0.0, // R
                                                      1.0, // G
                                                      0.0, // B
                                                      1.0  // A
                                                  }};
  wgpu::RenderPassDescriptor passDesc{.colorAttachmentCount = 1,
                                      .colorAttachments = &colorAttachment};
  wgpu::RenderPassEncoder guiPasss = encoder.BeginRenderPass(&passDesc);
  // Start the ImGui frame
  ImGui_ImplWGPU_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowSize(ImVec2(300, 100));
  ImGui::Begin("Settings");
  ImGui::Text("Application average %.3f ms/frame",
              1000.0f / ImGui::GetIO().Framerate);

  ImGui::Separator();
  ImGui::Text("rendered meshlets: %d/%d", renderedMeshletsCount, meshletCount);
  ImGui::Text("rendered triangles: %d/%d",
              maxMeshletTriangleCount * renderedMeshletsCount,
              maxMeshletTriangleCount * meshletCount);

  ImGui::End();

  // Render ImGui
  ImGui::Render();

  ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), guiPasss.Get());

  guiPasss.End();
}
