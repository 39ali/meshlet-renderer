
#include "renderer.h"

#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_wgpu.h"

#include <fstream>
#include <iostream>

struct SceneData {
  glm::mat4 proj;
  glm::mat4 view;
  glm::vec4 viewport;
};

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

  // gpuGaussian = loadGaussianGPU("assets/train_iteration_30000.splat");

  // gaussianCount = gpuGaussian.size();
  // std::cout << "gaussianCount: " << gaussianCount << " " <<
  // sizeof(GaussianGPU)
  //           << std::endl;

  // {
  //   wgpu::BufferDescriptor desc{};
  //   desc.label = "scene buffer";
  //   desc.size = sizeof(SceneData);
  //   desc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  //   sceneBuffer = ctx.device.CreateBuffer(&desc);
  // }

  // {
  //   wgpu::BufferDescriptor desc{};
  //   desc.label = "gaussian buffer";
  //   desc.size = sizeof(GaussianGPU) * gpuGaussian.size();
  //   desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  //   gaussianBuffer = ctx.device.CreateBuffer(&desc);
  //   ctx.queue.WriteBuffer(gaussianBuffer, 0, gpuGaussian.data(),
  //                         sizeof(GaussianGPU) * gpuGaussian.size());
  // }

  // {
  //   wgpu::BufferDescriptor desc{};
  //   desc.label = "visibleIndexBuffer";
  //   desc.size = sizeof(uint32_t) * gaussianCount;
  //   desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
  //   visibleIndexBuffer = ctx.device.CreateBuffer(&desc);
  // }

  // {
  //   wgpu::BufferDescriptor indirectDesc{.label = "Indirect Draw Buffer",
  //                                       .usage = wgpu::BufferUsage::Indirect
  //                                       |
  //                                                wgpu::BufferUsage::Storage |
  //                                                wgpu::BufferUsage::CopyDst,
  //                                       .size = sizeof(uint32_t) * 4};

  //   indirectBuffer = ctx.device.CreateBuffer(&indirectDesc);
  // }

  initCullPipeline();
  initRenderPipline();
}

void Renderer::initRenderPipline() {

  //   wgpu::BindGroupLayoutEntry entries[3] = {};
  //   entries[0].binding = 0;
  //   entries[0].visibility =
  //       wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
  //   entries[0].buffer.type = wgpu::BufferBindingType::Uniform;
  //   entries[0].buffer.minBindingSize = sizeof(SceneData);

  //   entries[1].binding = 1;
  //   entries[1].visibility = wgpu::ShaderStage::Vertex;
  //   entries[1].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  //   entries[1].buffer.minBindingSize = gaussianBuffer.GetSize();

  //   entries[2].binding = 2;
  //   entries[2].visibility = wgpu::ShaderStage::Vertex;
  //   entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
  //   entries[2].buffer.minBindingSize = visibleIndexBuffer.GetSize();

  //   wgpu::BindGroupLayoutDescriptor bglDesc{.entryCount = 3, .entries =
  //   entries}; wgpu::BindGroupLayout bgl =
  //   ctx.device.CreateBindGroupLayout(&bglDesc);

  //   wgpu::BindGroupEntry bgEntries[3] = {};
  //   bgEntries[0].binding = 0;
  //   bgEntries[0].buffer = sceneBuffer;
  //   bgEntries[0].offset = 0;
  //   bgEntries[0].size = sizeof(SceneData);
  //   //
  //   bgEntries[1].binding = 1;
  //   bgEntries[1].buffer = gaussianBuffer;
  //   bgEntries[1].offset = 0;
  //   bgEntries[1].size = gaussianBuffer.GetSize(); // sizeof(SceneData);
  //                                                 //
  //   bgEntries[2].binding = 2;
  //   bgEntries[2].buffer = visibleIndexBuffer;
  //   bgEntries[2].offset = 0;
  //   bgEntries[2].size = visibleIndexBuffer.GetSize();

  //   auto bgd = wgpu::BindGroupDescriptor{
  //       .layout = bgl, .entryCount = 3, .entries = bgEntries};
  //   renderBindGroup = ctx.device.CreateBindGroup(&bgd);

  //   /////

  //   auto shaderWGSL = R"(
  // struct SceneData {
  //     proj: mat4x4<f32>,
  //     view: mat4x4<f32>,
  //     viewport: vec4<f32>, // (width, height, _, _)
  // };

  // struct Gaussian {
  //     meanxy: vec2<f32>,
  //     meanz_color : vec2<f32>,
  //     cov3d: array<f32, 6>,
  // }; // 40bytes

  // @group(0) @binding(0) var<uniform> scene: SceneData;
  // @group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
  // @group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;

  // struct VSOut {
  //     @builtin(position) position: vec4<f32>,
  //     @location(0) color: vec4<f32>,
  //     @location(1) uv: vec2<f32>,
  // };

  // fn unpackRGBA(packed: u32) -> vec4f {
  //     let r = packed & 0xFFu;
  //     let g = (packed >> 8) & 0xFFu;
  //     let b = (packed >> 16) & 0xFFu;
  //     let a = (packed >> 24) & 0xFFu;
  //     var c = vec4f(f32(r)/255., f32(g)/255., f32(b)/255., f32(a)/255.);
  //     return c;
  // }

  // @vertex
  // fn vs_main(
  //     @builtin(vertex_index) vid: u32,
  //     @builtin(instance_index) iid: u32
  // ) -> VSOut {
  //     let idx = sorted_indices[iid];
  //     let g = gaussians[idx];

  //     let pos = vec3f(g.meanxy,g.meanz_color.x);
  //     let cam = scene.view * vec4<f32>(pos, 1.0);
  //     let clipPos = scene.proj * cam;

  //     let ndc = clipPos.xy / clipPos.w;

  //     let Vrk = mat3x3<f32>(
  //         vec3<f32>(g.cov3d[0], g.cov3d[1], g.cov3d[2]),
  //         vec3<f32>(g.cov3d[1], g.cov3d[3], g.cov3d[4]),
  //         vec3<f32>(g.cov3d[2], g.cov3d[4], g.cov3d[5])
  //     );

  //     // Extract focal lengths - perspectiveRH_ZO format
  //     let fx = scene.proj[0][0] * scene.viewport.x * 0.5;
  //     let fy = scene.proj[1][1] * scene.viewport.y * 0.5;  // Keep positive

  //     let z2 = cam.z * cam.z;

  //     let J = mat3x3<f32>(
  //         vec3<f32>( fx / cam.z, 0.0,- (fx * cam.x) / z2 ),
  //         vec3<f32>( 0.0, fy / cam.z, -(fy * cam.y) / z2 ),  // Both positive
  //         vec3<f32>( 0.0, 0.0, 0.0 )
  //     );

  //     let view3 = mat3x3<f32>(
  //         scene.view[0].xyz,
  //         scene.view[1].xyz,
  //         scene.view[2].xyz
  //     );

  //     let T = transpose(view3) * J;
  //     let cov2d = transpose(T) * Vrk * T;

  //     let mid = (cov2d[0][0] + cov2d[1][1]) * 0.5;
  //     let radius = length(vec2<f32>((cov2d[0][0] - cov2d[1][1]) * 0.5,
  //     cov2d[0][1])); let lambda1 = mid + radius; let lambda2 = mid - radius;

  //     if (lambda2 <= 0.0) {
  //         return VSOut(vec4<f32>(0.0, 0.0, 2.0, 1.0),
  //                      vec4<f32>(0.0),
  //                      vec2<f32>(0.0));
  //     }

  //     let diag = normalize(vec2<f32>(cov2d[0][1], lambda1 - cov2d[0][0]));
  //     let majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diag;
  //     let minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2<f32>(diag.y,
  //     -diag.x);

  // let quad = array<vec2<f32>, 4>(
  //     vec2<f32>(-2.0, -2.0),
  //     vec2<f32>( 2.0, -2.0),
  //     vec2<f32>(-2.0,  2.0),
  //     vec2<f32>( 2.0,  2.0)
  // );
  //     let q = quad[vid];

  //     let finalNDC = vec2<f32>(ndc.x, ndc.y) + q.x * 2.0 * majorAxis /
  //     scene.viewport.xy
  //                                           + q.y * 2.0 * minorAxis /
  //                                           scene.viewport.xy;

  //     // For ZO depth range [0,1], adjust depth factor
  //     let depthFactor = clipPos.z / clipPos.w;

  //     let rgba = unpackRGBA(bitcast<u32>(g.meanz_color.y));
  //     let color= vec4<f32>(rgba.rgb, rgba.a) * clamp(depthFactor + 1.0,
  //     0.0, 1.0);

  //     let depth = clipPos.z / clipPos.w;  // Keep in [0,1] for WebGPU

  //     return VSOut(vec4<f32>(finalNDC.x, finalNDC.y, 0.0, 1.0), color, q);

  // }

  // @fragment
  // fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  //     let A = -dot(in.uv, in.uv);

  //     if (A < -4.0) {
  //         discard;
  //     }

  //     let B = exp(A) * in.color.a;

  //     return vec4<f32>(in.color.rgb ,B);
  // }

  // )";
  //   wgpu::ShaderSourceWGSL wgslDesc = {};
  //   wgslDesc.code = shaderWGSL;
  //   // shader modules
  //   auto desc = wgpu::ShaderModuleDescriptor{.nextInChain = &wgslDesc};
  //   auto shaderModule = ctx.device.CreateShaderModule(&desc);

  //   // pipeline layout

  //   wgpu::PipelineLayoutDescriptor plDesc = {
  //       .bindGroupLayoutCount = 1,
  //       .bindGroupLayouts = &bgl,
  //   };
  //   wgpu::PipelineLayout pipelineLayout =
  //       ctx.device.CreatePipelineLayout(&plDesc);

  //   // render pipeline

  //   wgpu::RenderPipelineDescriptor rpDesc = {};
  //   rpDesc.layout = pipelineLayout;

  //   rpDesc.vertex = {
  //       .module = shaderModule,
  //       .entryPoint = "vs_main",
  //       .buffers = {} // No vertex buffers; procedural generation
  //   };

  //   wgpu::ColorTargetState colorTarget = {};
  //   colorTarget.format = ctx.backbufferFormat;

  //   wgpu::BlendState blend = {
  //       .color{.operation = wgpu::BlendOperation::Add,
  //              .srcFactor = wgpu::BlendFactor::SrcAlpha,
  //              .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},

  //       .alpha = wgpu::BlendComponent{
  //           .operation = wgpu::BlendOperation::Add,
  //           .srcFactor = wgpu::BlendFactor::One,
  //           .dstFactor = wgpu::BlendFactor::Zero,
  //       }};

  //   colorTarget.blend = &blend;

  //   wgpu::DepthStencilState depthState{
  //       .format = wgpu::TextureFormat::Depth24Plus,
  //       .depthWriteEnabled = false,
  //       .depthCompare = wgpu::CompareFunction::LessEqual,
  //   };

  //   wgpu::FragmentState fragment = {};
  //   fragment.module = shaderModule;
  //   fragment.entryPoint = "fs_main";
  //   fragment.targetCount = 1;
  //   fragment.targets = &colorTarget;
  //   rpDesc.fragment = &fragment;
  //   rpDesc.depthStencil = &depthState;

  //   rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleStrip;
  //   rpDesc.primitive.frontFace = wgpu::FrontFace::CW;
  //   rpDesc.primitive.cullMode = wgpu::CullMode::None;

  //   renderPipeline = ctx.device.CreateRenderPipeline(&rpDesc);
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
  return;
  // Update data
  SceneData scene;
  scene.proj = camera.getProjectionMatrix();
  glm::mat4 flipScale =
      glm::scale(glm::mat4(1.0f), glm::vec3(-1.0f, -1.0f, 1.0f));
  scene.view = camera.getViewMatrix() * flipScale;
  scene.viewport = glm::vec4(ctx.width, ctx.height, 0.0, 0.0);
  ctx.queue.WriteBuffer(sceneBuffer, 0, &scene, sizeof(SceneData));

  {
    std::vector<uint32_t> zeros(gaussianCount, 0);
    ctx.queue.WriteBuffer(visibleIndexBuffer, 0, zeros.data(),
                          gaussianCount * sizeof(uint32_t));
  }

  {
    uint32_t zero[] = {4, 0, 0, 0};
    ctx.queue.WriteBuffer(indirectBuffer, 0, zero, 4 * sizeof(uint32_t));
  }

  // {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   auto sortedIndices = sortGaussians(gpuGaussian, scene.view);
  //   ctx.queue.WriteBuffer(sortedIndicesBuffer, 0, sortedIndices.data(),
  //                         sizeof(uint32_t) * sortedIndices.size());

  //   auto end = std::chrono::high_resolution_clock::now();
  //   auto duration =
  //       std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  //   std::cout << "Average time: " << duration.count() << " microseconds\n";
  // }

  wgpu::SurfaceTexture surfaceTexture;
  ctx.surface.GetCurrentTexture(&surfaceTexture);
  wgpu::TextureView view = surfaceTexture.texture.CreateView();

  // cull and write depth buffer
  {
    wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(cullingPipeline);
    pass.SetBindGroup(0, cullingBindGroup);
    uint32_t groups = (gaussianCount + 255) / 256;
    pass.DispatchWorkgroups(groups);
    pass.End();

    wgpu::CommandBuffer cmd = encoder.Finish();
    ctx.queue.Submit(1, &cmd);
  }

  // sort
  {
    const uint32_t tiles = (gaussianCount + 255) / 256;

    for (uint32_t pass = 0; pass < 4; ++pass) {
      const uint32_t digitOffset = pass * 8;

      // update uniforms

      struct RadixParams {
        uint32_t digitOffset;
        uint32_t numWorkgroups;
      } histParams = {digitOffset, tiles};

      ctx.queue.WriteBuffer(radixUniformBuffer, 0, &histParams,
                            sizeof(histParams));

      // encode commands

      wgpu::CommandEncoder encoder = ctx.device.CreateCommandEncoder();
      encoder.ClearBuffer(histogramBuffer, 0, histogramBuffer.GetSize());
      encoder.ClearBuffer(globalOffsetBuffer, 0, globalOffsetBuffer.GetSize());
      encoder.ClearBuffer(tileOffsetBuffer, 0, tileOffsetBuffer.GetSize());

      {
        wgpu::ComputePassDescriptor passDesc{};
        wgpu::ComputePassEncoder cpass = encoder.BeginComputePass(&passDesc);

        // ----------  histogram ----------
        cpass.SetPipeline(histogramPipeline);
        if (pass % 2 == 0) {
          cpass.SetBindGroup(0, histogramBindGroup1);
        } else {
          cpass.SetBindGroup(0, histogramBindGroup2);
        }
        cpass.DispatchWorkgroups(tiles, 1, 1);

        // ----------  prefix ----------
        cpass.SetPipeline(prefixPipeline);
        cpass.SetBindGroup(0, prefixBindGroup);
        cpass.DispatchWorkgroups(1, 1, 1);

        // ---------- scatter ----------
        cpass.SetPipeline(scatterPipeline);
        if (pass % 2 == 0) {
          cpass.SetBindGroup(0, scatterBindGroup1);
        } else {
          cpass.SetBindGroup(0, scatterBindGroup2);
        }
        cpass.DispatchWorkgroups(tiles, 1, 1);

        cpass.End();
      }

      wgpu::CommandBuffer cmd = encoder.Finish();
      ctx.queue.Submit(1, &cmd);
    }
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
