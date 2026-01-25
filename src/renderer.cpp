
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

  meshlets.resize(count);
  meshletVertices.resize(meshlets.back().vertex_offset +
                         meshlets.back().vertex_count);

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

  std::vector<MeshInstance> meshInstances;
  glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));

  for (uint32_t i = 0; i < 10000; i++) {
    glm::mat4 randomModelMatrix = createRandomTranslation(-30.0f, 30.0f);
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
  trianglesCount = meshletTriangles.size() / 3.;

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

  wgpu::BindGroupLayoutEntry entries[7] = {};

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

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 7, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  wgpu::BindGroupEntry bgEntries[7] = {};
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

  auto bgd = wgpu::BindGroupDescriptor{
      .layout = bgl, .entryCount = 7, .entries = bgEntries};
  renderBindGroup = ctx.device.CreateBindGroup(&bgd);

  /////

  auto shaderWGSL = R"(

struct SceneData {
  proj: mat4x4<f32>,
  view: mat4x4<f32>,
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
fn vs_main(
    @builtin(vertex_index) vtx : u32,
    @builtin(instance_index) meshletID : u32
) -> VSOut {

  let visibleData = visibleMeshlets[meshletID];
  let meshlet = meshlets[visibleData.meshletID];
  let instance = instances[visibleData.instanceID];

  
  let maxVtx = meshlet.triangleCount * 3u; 
  
   if (vtx >= maxVtx) { var o : VSOut; o.pos = vec4<f32>(0.0, 0.0, 0.0, 0.0); return o; } 
  
  let triangleVertexIndex : u32 = meshletTriangles[meshlet.triangleOffset + vtx]; 
  
  let vertexIndex : u32 = meshletVertices[meshlet.vertexOffset + triangleVertexIndex]; 
  
  let v = vertices[vertexIndex];

    var out : VSOut;
    out.pos = scene.proj* scene.view * instance.modelMatrix * vec4<f32>(v.pos, 1.0);
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

  //   {
  //     wgpu::BufferDescriptor desc{};
  //     desc.label = "depth buffer";
  //     desc.size = sizeof(uint32_t) * gaussianCount;
  //     desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  //     depthBuffer = ctx.device.CreateBuffer(&desc);
  //   }

  //   wgpu::BindGroupLayoutEntry entries[] = {
  //       {// Scene uniform
  //        .binding = 0,
  //        .visibility = wgpu::ShaderStage::Compute,
  //        .buffer = {.type = wgpu::BufferBindingType::Uniform}},
  //       {// Gaussians
  //        .binding = 1,
  //        .visibility = wgpu::ShaderStage::Compute,
  //        .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
  //       {// Visible indices
  //        .binding = 2,
  //        .visibility = wgpu::ShaderStage::Compute,
  //        .buffer = {.type = wgpu::BufferBindingType::Storage}},
  //       {// indirectBuffer
  //        .binding = 3,
  //        .visibility = wgpu::ShaderStage::Compute,
  //        .buffer = {.type = wgpu::BufferBindingType::Storage}},
  //       {// depth Buffer
  //        .binding = 4,
  //        .visibility = wgpu::ShaderStage::Compute,
  //        .buffer = {.type = wgpu::BufferBindingType::Storage}}};

  //   auto ldesc = wgpu::BindGroupLayoutDescriptor{
  //       .entryCount = 5,
  //       .entries = entries,
  //   };

  //   auto layout = ctx.device.CreateBindGroupLayout(&ldesc);

  //   std::array<wgpu::BindGroupEntry, 5> bgEntries = {};
  //   {

  //     bgEntries[0].binding = 0;
  //     bgEntries[0].buffer = sceneBuffer;
  //     bgEntries[0].offset = 0;
  //     bgEntries[0].size = sizeof(SceneData);
  //   }

  //   {
  //     bgEntries[1].binding = 1;
  //     bgEntries[1].buffer = gaussianBuffer;
  //     bgEntries[1].offset = 0;
  //     bgEntries[1].size = gaussianBuffer.GetSize();
  //   }

  //   {
  //     bgEntries[2].binding = 2;
  //     bgEntries[2].buffer = visibleIndexBuffer;
  //     bgEntries[2].offset = 0;
  //     bgEntries[2].size = visibleIndexBuffer.GetSize();
  //   }

  //   {
  //     bgEntries[3].binding = 3;
  //     bgEntries[3].buffer = indirectBuffer;
  //     bgEntries[3].offset = 0;
  //     bgEntries[3].size = indirectBuffer.GetSize();
  //   }

  //   {
  //     bgEntries[4].binding = 4;
  //     bgEntries[4].buffer = depthBuffer;
  //     bgEntries[4].offset = 0;
  //     bgEntries[4].size = depthBuffer.GetSize();
  //   }

  //   wgpu::BindGroupDescriptor bgDesc = {
  //       .label = "cull bindinggroup",
  //       .layout = layout,
  //       .entryCount = bgEntries.size(),
  //       .entries = bgEntries.data(),
  //   };
  //   cullingBindGroup = ctx.device.CreateBindGroup(&bgDesc);

  //   const char *shaderWGSL = R"(

  //    struct SceneData {
  //     proj: mat4x4<f32>,
  //     view: mat4x4<f32>,
  //     viewport: vec4<f32>,
  // };

  // struct Gaussian {
  //     meanxy: vec2<f32>,
  //     meanz_color : vec2<f32>,
  //     cov3d: array<f32, 6>,
  // }; // 40bytes

  // struct DrawIndirectArgs {
  //     vertexCount   : u32,
  //     instanceCount : atomic<u32>,
  //     firstVertex   : u32,
  //     firstInstance : u32,
  // };

  // @group(0) @binding(0) var<uniform> scene : SceneData;
  // @group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
  // @group(0) @binding(2) var<storage, read_write> visibleIndices : array<u32>;
  // @group(0) @binding(3) var<storage, read_write> drawArgs : DrawIndirectArgs;
  // @group(0) @binding(4) var<storage, read_write> depths : array<u32>;

  // fn floatToSortableUint(f: f32) -> u32 {
  //     let f_bits = bitcast<u32>(f);

  //     // If negative: flip all bits (0xFFFFFFFF)
  //     // If positive: flip only the sign bit (0x80000000)
  //     let mask = select(0x80000000u, 0xFFFFFFFFu, f < 0.0);

  //     return f_bits ^ mask;
  // }

  // @compute @workgroup_size(256)
  // fn main(@builtin(global_invocation_id) id : vec3<u32>) {

  //     let i = id.x;
  //     if (i >= arrayLength(&gaussians)) {
  //         return;
  //     }

  //     let g = gaussians[i];
  //     let pos = vec3f(g.meanxy,g.meanz_color.x);
  //     let worldPos = vec4<f32>(pos, 1.0);
  //     let viewPos  = scene.view * worldPos;
  //     let clipPos  = scene.proj * viewPos;

  //     let clip = 1.3 * clipPos.w;
  //     if (clipPos.z < 0.0 || clipPos.z > clipPos.w ||
  //         clipPos.x < -clip || clipPos.x > clip ||
  //         clipPos.y < -clip || clipPos.y > clip) {
  //         return ;
  //     }

  //     let writeIndex =atomicAdd(&drawArgs.instanceCount, 1u);
  //     visibleIndices[writeIndex] = i;

  //     // write depth
  //     let sortableDepth = floatToSortableUint(viewPos.z);
  //     depths[writeIndex] = sortableDepth;
  // }

  // )";

  //   wgpu::ShaderSourceWGSL wgslDesc{};
  //   wgslDesc.code = shaderWGSL;
  //   wgpu::ShaderModuleDescriptor shaderDesc = wgpu::ShaderModuleDescriptor{
  //       .nextInChain = &wgslDesc, .label = "culling"};
  //   wgpu::ShaderModule shaderModule =
  //   ctx.device.CreateShaderModule(&shaderDesc);

  //   wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {
  //       .bindGroupLayoutCount = 1, .bindGroupLayouts = &layout};

  //   wgpu::PipelineLayout pipelineLayout =
  //       ctx.device.CreatePipelineLayout(&pipelineLayoutDesc);

  //   wgpu::ComputePipelineDescriptor cpDesc = {

  //       .label = "cullingpipeline",
  //       .layout = pipelineLayout,
  //       .compute =
  //           {
  //               .module = shaderModule,
  //               .entryPoint = "main",
  //           },

  //   };

  //   cullingPipeline = ctx.device.CreateComputePipeline(&cpDesc);
}

void Renderer::render(const FlyCamera &camera, float time) {
  // Update data
  SceneData scene;
  scene.proj = camera.getProjectionMatrix();

  glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));
  scene.view = camera.getViewMatrix();
  ctx.queue.WriteBuffer(sceneBuffer, 0, &scene, sizeof(SceneData));

  {
    uint32_t zero[] = {maxMeshletTriangleCount * 3, meshletCount, 0, 0};
    ctx.queue.WriteBuffer(indirectBuffer, 0, zero, 4 * sizeof(uint32_t));
  }

  wgpu::SurfaceTexture surfaceTexture;
  ctx.surface.GetCurrentTexture(&surfaceTexture);
  wgpu::TextureView view = surfaceTexture.texture.CreateView();

  // cull and write depth buffer
  {
    // wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    // wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    // pass.SetPipeline(cullingPipeline);
    // pass.SetBindGroup(0, cullingBindGroup);
    // uint32_t groups = (gaussianCount + 255) / 256;
    // pass.DispatchWorkgroups(groups);
    // pass.End();

    // wgpu::CommandBuffer cmd = encoder.Finish();
    // ctx.queue.Submit(1, &cmd);
  }

  // render
  wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
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
    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&passDesc);
    pass.SetPipeline(renderPipeline);
    pass.SetBindGroup(0, renderBindGroup);
    pass.DrawIndirect(indirectBuffer, 0);
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
