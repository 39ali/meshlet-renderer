
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
  uint32_t totalMeshlets;
  float padding[3];
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
};

struct VisibleMeshlet {
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

  meshletsOut.resize(count);
  for (size_t i = 0; i < count; i++) {
    meshletsOut[i] = {meshlets[i].vertex_offset, meshlets[i].triangle_offset,
                      meshlets[i].vertex_count, meshlets[i].triangle_count};
  }

  meshletVertices.resize(meshletsOut.back().vertexOffset +
                         meshletsOut.back().vertexCount);

  // pack triangles into 32bit
  // We store 3 uint32s for every 4 triangles
  meshletTriangles.resize(meshletsOut.back().triangleOffset +
                          meshletsOut.back().triangleCount * 3);
  meshletTrianglesOut.reserve(meshletTriangles.size() / 4);

  for (auto &m : meshletsOut) {

    const uint8_t *tris = meshletTriangles.data() + m.triangleOffset;

    m.triangleOffset = uint32_t(meshletTrianglesOut.size());

    for (uint32_t i = 0; i < m.triangleCount; i += 4) {
      uint32_t v0 = 0u;
      uint32_t v1 = 0u;
      uint32_t v2 = 0u;

      for (uint32_t j = 0; j < 4; ++j) {
        if (i + j < m.triangleCount) {
          const uint8_t *t = tris + (i + j) * 3;

          v0 |= uint32_t(t[0]) << (j * 8);
          v1 |= uint32_t(t[1]) << (j * 8);
          v2 |= uint32_t(t[2]) << (j * 8);
        }
      }

      meshletTrianglesOut.push_back(v0);
      meshletTrianglesOut.push_back(v1);
      meshletTrianglesOut.push_back(v2);
    }
  }

  // ///////////////////////debug
  // for (size_t mIdx = 0; mIdx < meshletsOut.size(); ++mIdx) {
  //   const auto &m = meshletsOut[mIdx];
  //   std::cout << "Meshlet " << mIdx << " (" << m.triangleCount
  //             << " triangles):" << std::endl;

  //   // Each block of 3 uint32s contains 4 triangles
  //   uint32_t numBlocks = (m.triangleCount + 3) / 4;

  //   for (uint32_t b = 0; b < numBlocks; ++b) {
  //     uint32_t base = m.triangleOffset + (b * 3);

  //     uint32_t pV0 = meshletTrianglesOut[base + 0];
  //     uint32_t pV1 = meshletTrianglesOut[base + 1];
  //     uint32_t pV2 = meshletTrianglesOut[base + 2];

  //     for (uint32_t t = 0; t < 4; ++t) {
  //       uint32_t triIdx = (b * 4) + t;
  //       if (triIdx >= m.triangleCount)
  //         break;

  //       // Extract 8-bit local indices
  //       uint32_t v0 = (pV0 >> (t * 8)) & 0xFFu;
  //       uint32_t v1 = (pV1 >> (t * 8)) & 0xFFu;
  //       uint32_t v2 = (pV2 >> (t * 8)) & 0xFFu;

  //       std::cout << "  Tri " << triIdx << ": [" << v0 << ", " << v1 << ", "
  //                 << v2 << "]" << std::endl;
  //     }
  //   }
  //   std::cout << "--------------------------" << std::endl;
  // }

  // for (uint32_t i = 0; i < meshlets.size(); i++) {

  //   for (uint32_t j = 0; j < meshlets[i].triangle_count; j++) {

  //     std::cout
  //         << "  Tri " << j << ": ["
  //         << uint32_t(meshletTriangles[meshlets[i].triangle_offset + j * 3 +
  //         0])
  //         << ", "
  //         << uint32_t(meshletTriangles[meshlets[i].triangle_offset + j * 3 +
  //         1])
  //         << ", "
  //         << uint32_t(meshletTriangles[meshlets[i].triangle_offset + j * 3 +
  //         2])
  //         << "]" << std::endl;
  //   }
  // }

  //   for (auto t : meshletTrianglesOut) {
  //     for (){
  //  std::cout << <<(t) << std::endl;
  //     }

  //   }

  int kk = 0;
}

std::vector<uint32_t> generateStaticSwizzleBuffer(uint32_t maxVisibleMeshlets) {
  std::vector<uint32_t> swizzle;
  swizzle.reserve(maxVisibleMeshlets * 384);

  for (uint32_t m = 0; m < maxVisibleMeshlets; ++m) {
    uint32_t meshletBase = m * 384;
    for (uint32_t i = 0; i < 128; i += 4) {
      uint32_t blockStart = meshletBase + (i / 4) * 12;
      for (uint32_t j = 0; j < 4; ++j) {
        swizzle.push_back(blockStart + 0 + j); // V0s
        swizzle.push_back(blockStart + 4 + j); // V1s
        swizzle.push_back(blockStart + 8 + j); // V2s
      }
    }
  }
  return swizzle;
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

  auto desc = wgpu::TextureDescriptor{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .size = {ctx.width, ctx.height, 1},
      .format = wgpu::TextureFormat::Depth24Plus,
  };
  depthTexture = ctx.device.CreateTexture(&desc);
  depthView = depthTexture.CreateView();

  std::vector<Vertex> vertices{};
  std::vector<uint32_t> indices{};
  loadGLTF("assets/Duck.glb", vertices, indices);

  std::vector<Meshlet> meshlets{};
  std::vector<uint32_t> meshletVertices{};
  std::vector<uint32_t> meshletTriangles{};
  buildMeshlets(vertices, indices, meshlets, meshletVertices, meshletTriangles);
  // generateStaticIndexBuffer(meshlets.size());

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "scene buffer";
    desc.size = sizeof(SceneData);
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    sceneBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    wgpu::BufferDescriptor indirectDesc{.label = "Indirect Draw Buffer",
                                        .usage = wgpu::BufferUsage::Indirect |
                                                 wgpu::BufferUsage::Storage |
                                                 wgpu::BufferUsage::CopyDst,
                                        .size = sizeof(uint32_t) * 5};

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

  std::vector<MeshInstance> meshInstances;
  glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));

  for (uint32_t i = 0; i < 10000; i++) {
    glm::mat4 randomModelMatrix = createRandomTranslation(-30.0f, 30.0f);
    // glm::mat4 randomModelMatrix = createRandomTranslation(-1.0f, 1.0f);
    meshInstances.push_back({.modelMatrix = randomModelMatrix * scaleMat});
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "meshInstanceBuffer";
    desc.size = sizeof(MeshInstance) * meshInstances.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    meshInstanceBuffer = ctx.device.CreateBuffer(&desc);

    ctx.queue.WriteBuffer(meshInstanceBuffer, 0, meshInstances.data(),
                          meshInstanceBuffer.GetSize());
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "visibleMeshletBuffer";
    desc.size = sizeof(VisibleMeshlet) * meshlets.size() * meshInstances.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    visibleMeshletBuffer = ctx.device.CreateBuffer(&desc);

    std::vector<VisibleMeshlet> visibleMeshlets;

    for (uint32_t j = 0; j < meshInstances.size(); j++) {
      for (uint32_t i = 0; i < meshlets.size(); i++) {
        visibleMeshlets.push_back({.meshletID = i, .instanceID = j});
      }
    }

    ctx.queue.WriteBuffer(visibleMeshletBuffer, 0, visibleMeshlets.data(),
                          visibleMeshletBuffer.GetSize());
  }

  meshletCount = meshlets.size() * meshInstances.size();
  trianglesCount = meshletTriangles.size() * 4 / 3.;

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "Global Output Index Buffer";
    // Calculate worst-case size: every potential meshlet being visible
    desc.size = meshletCount * (3 * 128) * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::Storage;
    globalIndexBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    auto indices = generateStaticSwizzleBuffer(meshletCount);
    wgpu::BufferDescriptor desc{};
    desc.label = "index buffer";
    // Calculate worst-case size: every potential meshlet being visible
    desc.size = indices.size() * sizeof(uint32_t);
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index;
    indexBuffer = ctx.device.CreateBuffer(&desc);
    ctx.queue.WriteBuffer(indexBuffer, 0, indices.data(),
                          indexBuffer.GetSize());
  }

  maxMeshletTriangleCount = 0;
  for (auto &meshlet : meshlets) {
    maxMeshletTriangleCount =
        std::max(meshlet.triangleCount, maxMeshletTriangleCount);
  }

  std::cout << "maxMeshletTriangleCount: " << maxMeshletTriangleCount
            << std::endl;
  std::cout << "meshletCount :" << meshletCount << std::endl;
  std::cout << "triangles count :" << trianglesCount * meshInstances.size()
            << std::endl;
  std::cout << "vertices count :" << vertices.size() << std::endl;

  initCullPipeline();
  initRenderPipline();
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

  // visibleMeshlets
  entries[5].binding = 5;
  entries[5].visibility = wgpu::ShaderStage::Vertex;
  entries[5].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[5].buffer.minBindingSize = visibleMeshletBuffer.GetSize();

  // instances
  entries[6].binding = 6;
  entries[6].visibility = wgpu::ShaderStage::Vertex;
  entries[6].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[6].buffer.minBindingSize = meshInstanceBuffer.GetSize();

  // globalIndexBuffer
  entries[7].binding = 7;
  entries[7].visibility = wgpu::ShaderStage::Vertex;
  entries[7].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[7].buffer.minBindingSize = globalIndexBuffer.GetSize();

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
  bgEntries[5].buffer = visibleMeshletBuffer;
  bgEntries[5].offset = 0;
  bgEntries[5].size = visibleMeshletBuffer.GetSize();

  bgEntries[6].binding = 6;
  bgEntries[6].buffer = meshInstanceBuffer;
  bgEntries[6].offset = 0;
  bgEntries[6].size = meshInstanceBuffer.GetSize();

  bgEntries[7].binding = 7;
  bgEntries[7].buffer = globalIndexBuffer;
  bgEntries[7].offset = 0;
  bgEntries[7].size = globalIndexBuffer.GetSize();

  auto bgd = wgpu::BindGroupDescriptor{
      .layout = bgl, .entryCount = 8, .entries = bgEntries};
  renderBindGroup = ctx.device.CreateBindGroup(&bgd);

  /////

  auto shaderWGSL = R"(

struct SceneData {
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  totalMeshlets:u32,
  padding:array<u32,3>,
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
};

struct VisibleMeshlet {
    meshletID: u32,
    instanceID: u32,
};

struct MeshInstance {
  modelMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> scene : SceneData;

@group(0) @binding(1) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(2) var<storage, read> meshlets : array<Meshlet>;
@group(0) @binding(3) var<storage, read> meshletVertices : array<u32>;
@group(0) @binding(4) var<storage, read> meshletTriangles : array<u32>;

@group(0) @binding(5) var<storage, read> visibleMeshlets: array<VisibleMeshlet>;
@group(0) @binding(6) var<storage, read> instances: array<MeshInstance>;
@group(0) @binding(7) var<storage, read> outputIndices: array<u32>;



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

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VSOut {

    let packed = outputIndices[vid];
    let localID = packed & 0xFFu;
    let meshletID = packed >> 8u;

    if (localID == 0xff) {
        var o : VSOut;
        o.pos=  vec4<f32>(0.0, 0.0, 2.0, 1.0);
       
        return o;
    }

    let visibleData = visibleMeshlets[meshletID];
    let m           = meshlets[visibleData.meshletID];
    let instance    = instances[visibleData.instanceID];

    // Look up actual vertex position
    let vertexIndex = meshletVertices[m.vertexOffset + localID];
    let v = vertices[vertexIndex];

    var out : VSOut;
    out.pos = scene.proj * scene.view * instance.modelMatrix * vec4<f32>(v.pos, 1.0);
    out.color = rand3(meshletID);
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

  wgpu::RenderPipelineDescriptor rpDesc = {};
  rpDesc.layout = pipelineLayout;

  rpDesc.vertex = {
      .module = shaderModule, .entryPoint = "vs_main", .buffers = {}};

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

  wgpu::DepthStencilState depthState{
      .format = wgpu::TextureFormat::Depth24Plus,
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::LessEqual,
  };

  wgpu::FragmentState fragment = {};
  fragment.module = shaderModule;
  fragment.entryPoint = "fs_main";
  fragment.targetCount = 1;
  fragment.targets = &colorTarget;
  rpDesc.fragment = &fragment;
  rpDesc.depthStencil = &depthState;

  rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
  rpDesc.primitive.frontFace = wgpu::FrontFace::CCW;
  rpDesc.primitive.cullMode = wgpu::CullMode::None;

  renderPipeline = ctx.device.CreateRenderPipeline(&rpDesc);
}

void Renderer::initCullPipeline() {

  wgpu::BindGroupLayoutEntry entries[6] = {};

  // scene
  entries[0].binding = 0;
  entries[0].visibility = wgpu::ShaderStage::Compute;
  entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
  entries[0].buffer.minBindingSize = sizeof(SceneData);

  // meshlet
  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Compute;
  entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[1].buffer.minBindingSize = meshletBuffer.GetSize();

  // meshlettriangle
  entries[2].binding = 2;
  entries[2].visibility = wgpu::ShaderStage::Compute;
  entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[2].buffer.minBindingSize = meshletTriangleBuffer.GetSize();

  // visibleMeshlets
  entries[3].binding = 3;
  entries[3].visibility = wgpu::ShaderStage::Compute;
  entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[3].buffer.minBindingSize = visibleMeshletBuffer.GetSize();

  // indirectbuffer
  entries[4].binding = 4;
  entries[4].visibility = wgpu::ShaderStage::Compute;
  entries[4].buffer.type = wgpu::BufferBindingType::Storage;
  entries[4].buffer.minBindingSize = indirectBuffer.GetSize();

  // globalIndexBuffer
  entries[5].binding = 5;
  entries[5].visibility = wgpu::ShaderStage::Compute;
  entries[5].buffer.type = wgpu::BufferBindingType::Storage;
  entries[5].buffer.minBindingSize = globalIndexBuffer.GetSize();

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 6, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  wgpu::BindGroupEntry bgEntries[6] = {};
  bgEntries[0].binding = 0;
  bgEntries[0].buffer = sceneBuffer;
  bgEntries[0].offset = 0;
  bgEntries[0].size = sizeof(SceneData);
  //

  bgEntries[1].binding = 1;
  bgEntries[1].buffer = meshletBuffer;
  bgEntries[1].offset = 0;
  bgEntries[1].size = meshletBuffer.GetSize();

  bgEntries[2].binding = 2;
  bgEntries[2].buffer = meshletTriangleBuffer;
  bgEntries[2].offset = 0;
  bgEntries[2].size = meshletTriangleBuffer.GetSize();

  bgEntries[3].binding = 3;
  bgEntries[3].buffer = visibleMeshletBuffer;
  bgEntries[3].offset = 0;
  bgEntries[3].size = visibleMeshletBuffer.GetSize();

  bgEntries[4].binding = 4;
  bgEntries[4].buffer = indirectBuffer;
  bgEntries[4].offset = 0;
  bgEntries[4].size = indirectBuffer.GetSize();

  bgEntries[5].binding = 5;
  bgEntries[5].buffer = globalIndexBuffer;
  bgEntries[5].offset = 0;
  bgEntries[5].size = globalIndexBuffer.GetSize();

  wgpu::BindGroupDescriptor bgDesc = {
      .label = "cull bindinggroup",
      .layout = bgl,
      .entryCount = 6,
      .entries = bgEntries,
  };
  cullingBindGroup = ctx.device.CreateBindGroup(&bgDesc);

  const char *shaderWGSL = R"(
struct SceneData {
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
  totalMeshlets:u32,
  padding:array<u32,3>,
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
};

struct VisibleMeshlet {
    meshletID: u32,
    instanceID: u32,
};

struct MeshInstance {
  modelMatrix: mat4x4<f32>,
};

struct DrawIndirectArgs {
vertexCount   : atomic<u32>,
    instanceCount : u32,
    firstVertex   : u32,
    firstInstance : u32,
};

@group(0) @binding(0) var<uniform> scene : SceneData;

@group(0) @binding(1) var<storage, read> meshlets : array<Meshlet>;
@group(0) @binding(2) var<storage, read> meshletTriangles : array<u32>;

@group(0) @binding(3) var<storage, read> visibleMeshlets: array<VisibleMeshlet>;
@group(0) @binding(4) var<storage, read_write> drawArgs : DrawIndirectArgs;
@group(0) @binding(5) var<storage, read_write> outputIndices :array<vec4<u32>> ;


var<workgroup> num_primitives : u32;
var<workgroup> base_index     : u32;

const GROUP_SIZE : u32 = 32;
@compute @workgroup_size(GROUP_SIZE)
fn main(
    @builtin(workgroup_id) wg : vec3<u32>,
    @builtin(local_invocation_index) local_id : u32
) {

    let group_id = wg.x + (wg.y * 256u); 

    // Guard against the remainder of the grid
    if (group_id >= scene.totalMeshlets) {
        return;
    }

    if (group_id >= scene.totalMeshlets) {
        return;
    }

    // lane 0 reads meshlet header (like Tellusim)
    if (local_id == 0u) {
        let vis = visibleMeshlets[group_id];
        let meshlet = meshlets[vis.meshletID];

        num_primitives = meshlet.triangleCount;
        base_index     = meshlet.triangleOffset;
    }

    workgroupBarrier();

   // ---- 1 invocation = 4 triangles ----
    var indices_0 : u32 = 0xffffffff;
    var indices_1 : u32 = 0xffffffff;
    var indices_2 : u32 = 0xffffffff;

    // only valid if this invocation maps to at least one triangle
    if (local_id * 4u < num_primitives) {
        let address = base_index + local_id * 3u;

        indices_0 = meshletTriangles[address + 0u];
        indices_1 = meshletTriangles[address + 1u]; 
        indices_2 = meshletTriangles[address + 2u]; 
    }

    let group_index = group_id << 8u;

    // 3 vec4 per invocation (one per triangle vertex)
    let index = (GROUP_SIZE * group_id + local_id) * 3u;

    outputIndices[index + 0u] =
        (vec4<u32>(
            indices_0,
            indices_0 >> 8u,
            indices_0 >> 16u,
            indices_0 >> 24u
        ) & vec4<u32>(0xFFu)) | vec4<u32>(group_index);

    outputIndices[index + 1u] =
        (vec4<u32>(
            indices_1,
            indices_1 >> 8u,
            indices_1 >> 16u,
            indices_1 >> 24u
        ) & vec4<u32>(0xFFu)) | vec4<u32>(group_index);

    outputIndices[index + 2u] =
        (vec4<u32>(
            indices_2,
            indices_2 >> 8u,
            indices_2 >> 16u,
            indices_2 >> 24u
        ) & vec4<u32>(0xFFu)) | vec4<u32>(group_index);
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
  scene.totalMeshlets = meshletCount;

  glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));
  scene.view = camera.getViewMatrix();
  ctx.queue.WriteBuffer(sceneBuffer, 0, &scene, sizeof(SceneData));

  {
    // struct DrawIndirectArgs {
    // vertexCount   : atomic<u32>,
    //     instanceCount : u32,
    //     firstVertex   : u32,
    //     firstInstance : u32,
    // };
    uint32_t vertexCount = meshletCount * 128 * 3;

    uint32_t zero[] = {vertexCount, 1, 0, 0, 0};
    ctx.queue.WriteBuffer(indirectBuffer, 0, zero, 5 * sizeof(uint32_t));
  }

  wgpu::SurfaceTexture surfaceTexture;
  ctx.surface.GetCurrentTexture(&surfaceTexture);
  wgpu::TextureView view = surfaceTexture.texture.CreateView();

  wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();

  // cull and write depth buffer
  {

    uint32_t totalTasks = meshletCount;
    uint32_t gridX = 256; // A safe width well under 65535
    uint32_t gridY = (totalTasks + gridX - 1) / gridX; // Ceiling division

    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(cullingPipeline);
    pass.SetBindGroup(0, cullingBindGroup);
    pass.DispatchWorkgroups(gridX, gridY, 1);
    pass.End();
  }

  // render
  {
    wgpu::RenderPassColorAttachment colorAttachment{
        .view = view,
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = wgpu::Color{0.1, 0.2, 0.4, 0.0}};

    wgpu::RenderPassDepthStencilAttachment depthAttachment{
        .view = depthView,
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Discard,
        .depthClearValue = 1.0f};

    wgpu::RenderPassDescriptor passDesc{.colorAttachmentCount = 1,
                                        .colorAttachments = &colorAttachment,
                                        .depthStencilAttachment =
                                            &depthAttachment

    };
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(renderPipeline);
    pass.SetBindGroup(0, renderBindGroup);
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32);
    pass.DrawIndexedIndirect(indirectBuffer, 0);
    pass.End();
  }

  // gui pass
  renderGui(encoder, view);

  wgpu::CommandBuffer cmd = encoder.Finish();

  ctx.queue.Submit(1, &cmd);

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

  // Your UI code
  ImGui::Begin("Settings");
  ImGui::Text("Application average %.3f ms/frame",
              1000.0f / ImGui::GetIO().Framerate);
  ImGui::End();

  // Render ImGui
  ImGui::Render();

  ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), guiPasss.Get());

  guiPasss.End();
}
