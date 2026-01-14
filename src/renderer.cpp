
// https://virtualhumans.mpi-inf.mpg.de/3DVision25/slides25/pdf/Lecture_08_GaussianSplatting.pdf#:~:text=The%20covariance%20matrix%20%CE%A3%20of%20a%203D,meaning%20if%20its%20a%20positive%2Dsemi%20definite%20matrix.
#include "renderer.h"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <chrono>

struct GaussianCPU {
  glm::vec3 mean;
  float _pad0;

  glm::vec3 scale;
  float _pad1;

  glm::vec4 rotation;

  glm::vec3 color;
  float opacity;
};

struct SortEntity {
  float depth;
  uint32_t index;

  // We want Back-to-Front, so further objects (larger depth) come first
  bool operator>(const SortEntity &other) const { return depth > other.depth; }

  bool operator<(const SortEntity &other) const { return depth < other.depth; }
};

std::vector<uint32_t> sortGaussians(const std::vector<GaussianGPU> &gaussians,
                                    const glm::mat4 &viewMatrix) {
  size_t count = gaussians.size();
  std::vector<SortEntity> depthList(count);

  for (uint32_t i = 0; i < count; ++i) {
    // Transform the 3D mean into camera space
    // We only care about the Z component
    // ViewMatrix * vec4(pos, 1.0)
    glm::vec4 viewPos = viewMatrix * glm::vec4(gaussians[i].mean, 1.0f);
    depthList[i] = {viewPos.z, i};
  }

  // Sort Back-to-Front
  // Use std::stable_sort if you want to avoid flickering for identical depths
  std::sort(depthList.begin(), depthList.end(),
            [](const SortEntity &a, const SortEntity &b) {
              return a.depth < b.depth; // back → front
            });

  // Extract just the indices
  std::vector<uint32_t> sortedIndices(count);
  for (size_t i = 0; i < count; ++i) {
    sortedIndices[i] = depthList[i].index;
  }

  return sortedIndices;
}

struct SceneData {
  glm::mat4 proj;
  glm::mat4 view;
  glm::vec4 viewport;
};

struct PropertyInfo {
  std::string name;
  std::string type;
};

struct PlyHeader {
  bool binary = false;
  size_t vertexCount = 0;
  std::vector<PropertyInfo> properties;
};

PlyHeader parsePlyHeader(std::ifstream &file) {
  PlyHeader header;
  std::string line;
  while (std::getline(file, line)) {
    if (line == "end_header")
      break;
    std::istringstream iss(line);
    std::string token;
    iss >> token;
    if (token == "format") {
      std::string fmt;
      iss >> fmt;
      header.binary = (fmt.find("binary") != std::string::npos);
    } else if (token == "element") {
      std::string elem;
      iss >> elem;
      if (elem == "vertex") {
        iss >> header.vertexCount;
      }
    } else if (token == "property") {
      std::string type, name;
      iss >> type >> name;
      header.properties.push_back({name, type});
    }
  }
  return header;
}

// Helper to convert SH coefficients to RGB
inline float SH_C0 = 0.28209479177387814f;

glm::vec3 shToRGB(float sh0, float sh1, float sh2) {
  // Convert from SH to RGB (assuming DC component only)
  float r = 0.5f + SH_C0 * sh0;
  float g = 0.5f + SH_C0 * sh1;
  float b = 0.5f + SH_C0 * sh2;
  return glm::vec3(glm::clamp(r, 0.0f, 1.0f), glm::clamp(g, 0.0f, 1.0f),
                   glm::clamp(b, 0.0f, 1.0f));
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<GaussianCPU> loadGaussianPLY(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Failed to open PLY");

  PlyHeader header = parsePlyHeader(file);
  if (!header.binary)
    throw std::runtime_error("ASCII PLY not supported yet");

  std::vector<GaussianCPU> gaussians(header.vertexCount);

  for (size_t i = 0; i < header.vertexCount; i++) {
    GaussianCPU g{};
    float sh_dc[3] = {0, 0, 0};

    for (const auto &prop : header.properties) {
      float v;

      // Read based on type
      if (prop.type == "float") {
        file.read(reinterpret_cast<char *>(&v), sizeof(float));
      } else if (prop.type == "double") {
        double d;
        file.read(reinterpret_cast<char *>(&d), sizeof(double));
        v = static_cast<float>(d);
      } else if (prop.type == "uchar") {
        unsigned char uc;
        file.read(reinterpret_cast<char *>(&uc), sizeof(unsigned char));
        v = static_cast<float>(uc);
      } else {
        // Default to float for unknown types
        file.read(reinterpret_cast<char *>(&v), sizeof(float));
      }

      // Parse property
      if (prop.name == "x")
        g.mean.x = v;
      else if (prop.name == "y")
        g.mean.y = v;
      else if (prop.name == "z")
        g.mean.z = v;
      else if (prop.name == "f_dc_0")
        sh_dc[0] = v;
      else if (prop.name == "f_dc_1")
        sh_dc[1] = v;
      else if (prop.name == "f_dc_2")
        sh_dc[2] = v;
      else if (prop.name == "opacity")
        g.opacity = sigmoid(v); // Convert from logit space
      else if (prop.name == "scale_0")
        g.scale.x = std::exp(v); // Convert from log space
      else if (prop.name == "scale_1")
        g.scale.y = std::exp(v);
      else if (prop.name == "scale_2")
        g.scale.z = std::exp(v);
      else if (prop.name == "rot_0")
        g.rotation.w = v;
      else if (prop.name == "rot_1")
        g.rotation.x = v;
      else if (prop.name == "rot_2")
        g.rotation.y = v;
      else if (prop.name == "rot_3")
        g.rotation.z = v;
    }

    // Convert SH to RGB
    g.color = shToRGB(sh_dc[0], sh_dc[1], sh_dc[2]);

    gaussians[i] = g;
  }

  return gaussians;
}

std::vector<GaussianGPU>
preprocessGaussians(const std::vector<GaussianCPU> &cpuGaussians) {
  std::vector<GaussianGPU> gpuGaussians;
  gpuGaussians.reserve(cpuGaussians.size());

  for (const auto &g : cpuGaussians) {
    GaussianGPU gpu;
    gpu.mean = g.mean;
    gpu.opacity = g.opacity;

    if (g.opacity >= 1.0) {
      std::cout << g.opacity << std::endl;
    }
    gpu.color = g.color;

    // Build rotation matrix from quaternion
    glm::quat q = glm::normalize(
        glm::quat(g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z));
    glm::mat3 R = glm::mat3_cast(q);

    // Scale matrix
    glm::mat3 S = glm::mat3(g.scale.x, 0.0f, 0.0f, 0.0f, g.scale.y, 0.0f, 0.0f,
                            0.0f, g.scale.z);

    // Compute M = R * S
    glm::mat3 M = R * S;

    // Compute 3D covariance: Σ = M * M^T
    glm::mat3 Sigma = M * glm::transpose(M);

    // Store upper triangle (symmetric matrix)
    gpu.cov3d[0] = Sigma[0][0]; // Σ00
    gpu.cov3d[1] = Sigma[1][0]; // Σ01 (was Sigma[0][1])
    gpu.cov3d[2] = Sigma[2][0]; // Σ02 (was Sigma[0][2])
    gpu.cov3d[3] = Sigma[1][1]; // Σ11
    gpu.cov3d[4] = Sigma[2][1]; // Σ12 (was Sigma[1][2])
    gpu.cov3d[5] = Sigma[2][2];

    gpu._pad1[0] = 0.0f;
    gpu._pad1[1] = 0.0f;
    gpu._pad2 = 0.0f;

    gpuGaussians.push_back(gpu);
  }

  return gpuGaussians;
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

  auto gaussians = loadGaussianPLY("train_iteration_7000.ply");

  gpuGaussian = preprocessGaussians(gaussians);

  gaussianCount = gpuGaussian.size();
  std::cout << "gaussianCount: " << gaussianCount << " " << sizeof(GaussianGPU)
            << std::endl;

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "scene buffer";
    desc.size = sizeof(SceneData);
    desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    sceneBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "gaussian buffer";
    desc.size = sizeof(GaussianGPU) * gpuGaussian.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    gaussianBuffer = ctx.device.CreateBuffer(&desc);
    ctx.queue.WriteBuffer(gaussianBuffer, 0, gpuGaussian.data(),
                          sizeof(GaussianGPU) * gpuGaussian.size());
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "sorted indices buffer";
    desc.size = sizeof(uint32_t) * gpuGaussian.size();
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    sortedIndicesBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    wgpu::BufferDescriptor desc{};
    desc.label = "visibleIndexBuffer";
    desc.size = sizeof(uint32_t) * gaussianCount;
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    visibleIndexBuffer = ctx.device.CreateBuffer(&desc);
  }

  {
    wgpu::BufferDescriptor indirectDesc{.label = "Indirect Draw Buffer",
                                        .usage = wgpu::BufferUsage::Indirect |
                                                 wgpu::BufferUsage::Storage |
                                                 wgpu::BufferUsage::CopyDst,
                                        .size = sizeof(uint32_t) * 4};

    indirectBuffer = ctx.device.CreateBuffer(&indirectDesc);
  }

  initRenderPipline();
  initCullPipeline();
}

void Renderer::initRenderPipline() {

  wgpu::BindGroupLayoutEntry entries[3] = {};
  entries[0].binding = 0;
  entries[0].visibility =
      wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
  entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
  entries[0].buffer.minBindingSize = sizeof(SceneData);

  entries[1].binding = 1;
  entries[1].visibility = wgpu::ShaderStage::Vertex;
  entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[1].buffer.minBindingSize = gaussianBuffer.GetSize();

  entries[2].binding = 2;
  entries[2].visibility = wgpu::ShaderStage::Vertex;
  entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  entries[2].buffer.minBindingSize = sortedIndicesBuffer.GetSize();

  wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 3, .entries = entries};
  wgpu::BindGroupLayout bgl = ctx.device.CreateBindGroupLayout(&bglDesc);

  wgpu::BindGroupEntry bgEntries[3] = {};
  bgEntries[0].binding = 0;
  bgEntries[0].buffer = sceneBuffer;
  bgEntries[0].offset = 0;
  bgEntries[0].size = sizeof(SceneData);
  //
  bgEntries[1].binding = 1;
  bgEntries[1].buffer = gaussianBuffer;
  bgEntries[1].offset = 0;
  bgEntries[1].size = gaussianBuffer.GetSize(); // sizeof(SceneData);
                                                //
  bgEntries[2].binding = 2;
  bgEntries[2].buffer = sortedIndicesBuffer;
  bgEntries[2].offset = 0;
  bgEntries[2].size = sortedIndicesBuffer.GetSize();

  auto bgd = wgpu::BindGroupDescriptor{
      .layout = bgl, .entryCount = 3, .entries = bgEntries};
  renderBindGroup = ctx.device.CreateBindGroup(&bgd);

  /////

  auto shaderWGSL = R"(
struct SceneData {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    viewport: vec4<f32>, // (width, height, _, _)
};

struct Gaussian {
    mean: vec3<f32>,
    opacity: f32,
    cov3d: array<f32, 6>, // upper triangle
    _pad1: vec2<f32>,
    color: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> scene: SceneData;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VSOut {
    let idx = sorted_indices[iid];
    let g = gaussians[idx];
    
    let cam = scene.view * vec4<f32>(g.mean, 1.0);
    let clipPos = scene.proj * cam;
    
    let clip = 1.2 * clipPos.w;
    if (clipPos.z < 0.0 || clipPos.z > clipPos.w ||
        clipPos.x < -clip || clipPos.x > clip ||
        clipPos.y < -clip || clipPos.y > clip) {
        return VSOut(vec4<f32>(0.0, 0.0, 2.0, 1.0),
                     vec4<f32>(0.0),
                     vec2<f32>(0.0));
    }
    
    let ndc = clipPos.xy / clipPos.w;
    
    let Vrk = mat3x3<f32>(
        vec3<f32>(g.cov3d[0], g.cov3d[1], g.cov3d[2]),
        vec3<f32>(g.cov3d[1], g.cov3d[3], g.cov3d[4]),
        vec3<f32>(g.cov3d[2], g.cov3d[4], g.cov3d[5])
    );
    
    // Extract focal lengths - perspectiveRH_ZO format
    let fx = scene.proj[0][0] * scene.viewport.x * 0.5;
    let fy = scene.proj[1][1] * scene.viewport.y * 0.5;  // Keep positive
    
    // For RH system, cam.z is negative when in front of camera
    let z = cam.z;  // Don't negate!
    let z2 = z * z;
    
    let J = mat3x3<f32>(
        vec3<f32>( fx / z, 0.0,- (fx * cam.x) / z2 ),
        vec3<f32>( 0.0, fy / z, -(fy * cam.y) / z2 ),  // Both positive
        vec3<f32>( 0.0, 0.0, 0.0 )
    );
    
    let view3 = mat3x3<f32>(
        scene.view[0].xyz,
        scene.view[1].xyz,
        scene.view[2].xyz
    );
    
    let T = transpose(view3) * J;
    let cov2d = transpose(T) * Vrk * T;
    
    let mid = (cov2d[0][0] + cov2d[1][1]) * 0.5;
    let radius = length(vec2<f32>((cov2d[0][0] - cov2d[1][1]) * 0.5, cov2d[0][1]));
    let lambda1 = mid + radius;
    let lambda2 = mid - radius;
    
    if (lambda2 <= 0.0) {
        return VSOut(vec4<f32>(0.0, 0.0, 2.0, 1.0),
                     vec4<f32>(0.0),
                     vec2<f32>(0.0));
    }
    
    let diag = normalize(vec2<f32>(cov2d[0][1], lambda1 - cov2d[0][0]));
    let majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diag;
    let minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2<f32>(diag.y, -diag.x);

    
    
let quad = array<vec2<f32>, 4>(
    vec2<f32>(-2.0, -2.0),
    vec2<f32>( 2.0, -2.0),
    vec2<f32>(-2.0,  2.0),
    vec2<f32>( 2.0,  2.0)
);
    let q = quad[vid];
    
    let finalNDC = vec2<f32>(ndc.x, ndc.y) + q.x * 2.0 * majorAxis / scene.viewport.xy
                                          + q.y * 2.0 * minorAxis / scene.viewport.xy;
    
    // For ZO depth range [0,1], adjust depth factor
    let depthFactor = clipPos.z / clipPos.w;
    let color = vec4<f32>(g.color, g.opacity) * clamp(depthFactor + 1.0, 0.0, 1.0);
    
    let depth = clipPos.z / clipPos.w;  // Keep in [0,1] for WebGPU
   
    
    return VSOut(vec4<f32>(finalNDC.x, finalNDC.y, 0.0, 1.0), color, q);

}


@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let A = -dot(in.uv, in.uv);

    
    if (A < -4.0) {
        discard;
    }

    let B = exp(A) * in.color.a;
    
   
    return vec4<f32>(in.color.rgb ,B);
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
      .module = shaderModule,
      .entryPoint = "vs_main",
      .buffers = {} // No vertex buffers; procedural generation
  };

  wgpu::ColorTargetState colorTarget = {};
  colorTarget.format = ctx.backbufferFormat;
  // wgpu::BlendState blend = {
  //     .color{.operation = wgpu::BlendOperation::Add,
  //            .srcFactor = wgpu::BlendFactor::OneMinusDstAlpha,
  //            .dstFactor = wgpu::BlendFactor::One},

  //     .alpha = wgpu::BlendComponent{
  //         .operation = wgpu::BlendOperation::Add,
  //         .srcFactor = wgpu::BlendFactor::OneMinusDstAlpha,
  //         .dstFactor = wgpu::BlendFactor::One,
  //     }};

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
      .depthWriteEnabled = false,
      .depthCompare = wgpu::CompareFunction::LessEqual,
  };

  wgpu::FragmentState fragment = {};
  fragment.module = shaderModule;
  fragment.entryPoint = "fs_main";
  fragment.targetCount = 1;
  fragment.targets = &colorTarget;
  rpDesc.fragment = &fragment;
  rpDesc.depthStencil = &depthState;

  rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
  rpDesc.primitive.frontFace = wgpu::FrontFace::CW; // Counter-Clockwise
  rpDesc.primitive.cullMode = wgpu::CullMode::None; // Hide the back

  renderPipeline = ctx.device.CreateRenderPipeline(&rpDesc);
}

void Renderer::initCullPipeline() {

  wgpu::BindGroupLayoutEntry entries[] = {
      {// Scene uniform
       .binding = 0,
       .visibility = wgpu::ShaderStage::Compute,
       .buffer = {.type = wgpu::BufferBindingType::Uniform}},
      {// Gaussians
       .binding = 1,
       .visibility = wgpu::ShaderStage::Compute,
       .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}},
      {// Visible indices
       .binding = 2,
       .visibility = wgpu::ShaderStage::Compute,
       .buffer = {.type = wgpu::BufferBindingType::Storage}},
      {// indirectBuffer
       .binding = 3,
       .visibility = wgpu::ShaderStage::Compute,
       .buffer = {.type = wgpu::BufferBindingType::Storage}}};

  auto ldesc = wgpu::BindGroupLayoutDescriptor{
      .entryCount = 4,
      .entries = entries,
  };

  auto layout = ctx.device.CreateBindGroupLayout(&ldesc);

  std::array<wgpu::BindGroupEntry, 4> bgEntries = {};
  {

    bgEntries[0].binding = 0;
    bgEntries[0].buffer = sceneBuffer;
    bgEntries[0].offset = 0;
    bgEntries[0].size = sizeof(SceneData);
  }

  {
    bgEntries[1].binding = 1;
    bgEntries[1].buffer = gaussianBuffer;
    bgEntries[1].offset = 0;
    bgEntries[1].size = gaussianBuffer.GetSize();
  }

  {
    bgEntries[2].binding = 2;
    bgEntries[2].buffer = visibleIndexBuffer;
    bgEntries[2].offset = 0;
    bgEntries[2].size = visibleIndexBuffer.GetSize();
  }

  {
    bgEntries[3].binding = 3;
    bgEntries[3].buffer = indirectBuffer;
    bgEntries[3].offset = 0;
    bgEntries[3].size = indirectBuffer.GetSize();
  }

  wgpu::BindGroupDescriptor bgDesc = {
      .label = "cull bindinggroup",
      .layout = layout,
      .entryCount = bgEntries.size(),
      .entries = bgEntries.data(),
  };
  cullingBindGroup = ctx.device.CreateBindGroup(&bgDesc);

  const char *shaderWGSL = R"(

   struct SceneData {
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverseViewProj: mat4x4<f32>,
    cameraPos: vec3<f32>,
    time: f32,
};

struct Gaussian {
   mean: vec3<f32>,
    _pad0: f32,

    scale: vec3<f32>,
    _pad1: f32,

    rotation: vec4<f32>,

    color: vec3<f32>,
    opacity: f32,
};

struct DrawIndirectArgs {
    vertexCount   : u32,
    instanceCount : atomic<u32>,
    firstVertex   : u32,
    firstInstance : u32,
};


@group(0) @binding(0) var<uniform> scene : SceneData;
@group(0) @binding(1) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2) var<storage, read_write> visibleIndices : array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgs : DrawIndirectArgs;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {

   

    let i = id.x;
    if (i >= arrayLength(&gaussians)) {
        return;
    }

    // let g = gaussians[i];

    // // World → view → clip
    // let worldPos = vec4<f32>(g.mean, 1.0);
    // let viewPos  = scene.view * worldPos;
    // let clipPos  = scene.proj * viewPos;

    // // Behind camera
    // if (clipPos.w <= 0.0) {
    //     return;
    // }

    // let ndc = clipPos.xyz / clipPos.w;

    // // Frustum culling (slightly relaxed)
    // if (ndc.x < -1.2 || ndc.x > 1.2 ||
    //     ndc.y < -1.2 || ndc.y > 1.2 ||
    //     ndc.z <  0.0 || ndc.z > 1.0) {
    //     return;
    // }

    // // Screen-size culling (cheap heuristic)
    // let maxScale = max(g.scale.x, max(g.scale.y, g.scale.z));
    // let approxRadius = maxScale / clipPos.w;

    // if (approxRadius < 0.5) {
    //     return;
    // }

    // Survived → append index
    let writeIndex =atomicAdd(&drawArgs.instanceCount, 1u);
    visibleIndices[writeIndex] = i;
}

)";

  wgpu::ShaderSourceWGSL wgslDesc{};
  wgslDesc.code = shaderWGSL;
  wgpu::ShaderModuleDescriptor shaderDesc = wgpu::ShaderModuleDescriptor{
      .nextInChain = &wgslDesc, .label = "culling"};
  wgpu::ShaderModule shaderModule = ctx.device.CreateShaderModule(&shaderDesc);

  wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {
      .bindGroupLayoutCount = 1, .bindGroupLayouts = &layout};

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

  // cpDesc.compute.entryPoint = "details_main";
  // detailNoisePipeline = ctx.device.CreateComputePipeline(&cpDesc);
}

void Renderer::render(const FlyCamera &camera, float time) {

  // Update data
  SceneData scene;
  scene.proj = camera.getProjectionMatrix();
  glm::mat4 flipScale =
      glm::scale(glm::mat4(1.0f), glm::vec3(-1.0f, -1.0f, 1.0f));

  scene.view = camera.getViewMatrix() * flipScale;
  scene.viewport = glm::vec4(ctx.width, ctx.height, 0.0, 0.0);
  // Send to GPU
  ctx.queue.WriteBuffer(sceneBuffer, 0, &scene, sizeof(SceneData));

  // {
  //   std::vector<uint32_t> zeros(gaussianCount, 0);
  //   ctx.queue.WriteBuffer(visibleIndexBuffer, 0, zeros.data(),
  //                         gaussianCount * sizeof(uint32_t));
  // }

  // {
  //   uint32_t zero[] = {4, 0, 0, 0};
  //   ctx.queue.WriteBuffer(indirectBuffer, 0, zero, 4 * sizeof(uint32_t));
  // }

  {
    auto start = std::chrono::high_resolution_clock::now();
    auto sortedIndices = sortGaussians(gpuGaussian, scene.view);
    ctx.queue.WriteBuffer(sortedIndicesBuffer, 0, sortedIndices.data(),
                          sizeof(uint32_t) * sortedIndices.size());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Average time: " << duration.count() << " microseconds\n";
  }

  wgpu::SurfaceTexture surfaceTexture;
  ctx.surface.GetCurrentTexture(&surfaceTexture);
  wgpu::TextureView view = surfaceTexture.texture.CreateView();

  wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
  // {

  //   wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  //   pass.SetPipeline(cullingPipeline);
  //   pass.SetBindGroup(0, cullingBindGroup);
  //   uint32_t groups = (gaussianCount + 255) / 256;
  //   pass.DispatchWorkgroups(groups);
  //   pass.End();
  // }
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
    // pass.DrawIndirect(indirectBuffer, 0);
    pass.Draw(4, gaussianCount);
    pass.End();
  }

  // gui pass
  // renderGui(encoder, view);

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
