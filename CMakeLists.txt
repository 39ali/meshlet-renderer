cmake_minimum_required(VERSION 3.20)
project(app LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "EMSCRIPTEN = ${EMSCRIPTEN}")


# --------------------------------------------------
# Sources
# --------------------------------------------------
file(GLOB_RECURSE SRC
    src/*.cpp
)

# --------------------------------------------------
# Executable targets
# --------------------------------------------------
 add_executable(app_native ${SRC})


 if (NOT EMSCRIPTEN)
    # ---------- GLFW ----------
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

    add_subdirectory(lib/glfw)
 endif()


 # # ---------- Dawn ----------
add_subdirectory(lib/dawn EXCLUDE_FROM_ALL)
# glm 
add_subdirectory(lib/glm)

## imgui 
set(IMGUI_DIR "lib/imgui")
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_wgpu.cpp
)
target_sources(app_native PRIVATE ${IMGUI_SOURCES})
target_include_directories(app_native PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
if(APPLE )
    # This specifically fixes the "@class NSString" error
    set_source_files_properties(
        "${PROJECT_SOURCE_DIR}/lib/imgui/backends/imgui_impl_wgpu.cpp" 
        PROPERTIES LANGUAGE OBJCXX
    )
endif()


# tinygltf
add_subdirectory(lib/tinygltf)

# meshoptimizer
add_subdirectory(lib/meshoptimizer)


if (EMSCRIPTEN)
    target_link_libraries(app_native PRIVATE emdawnwebgpu_cpp)
    target_link_options(app_native PRIVATE -sEXPORTED_RUNTIME_METHODS=['specialHTMLTargets'] -sEXPORT_ES6 -sMODULARIZE=1  -sINVOKE_RUN=0 -sNO_EXIT_RUNTIME=1 -sDISABLE_EXCEPTION_CATCHING=0  "-sALLOW_MEMORY_GROWTH=1" "-sASYNCIFY=1" --use-port=contrib.glfw3 )
    
    # input is relative to the build directory
    target_link_options(app_native PRIVATE --preload-file ${CMAKE_CURRENT_SOURCE_DIR}/assets@/assets)
else()

    target_link_libraries(app_native PRIVATE
        glfw
        webgpu_dawn
    )
     target_compile_definitions(app_native PRIVATE IMGUI_IMPL_WEBGPU_BACKEND_DAWN)

endif()


target_link_libraries(app_native PRIVATE webgpu_glfw glm  tinygltf meshoptimizer)


