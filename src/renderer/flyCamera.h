#pragma once
#define GLM_ENABLE_EXPERIMENTAL
#include "inputManager.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

class FlyCamera {
public:
  FlyCamera(float fov, float aspect, float nearPlane, float farPlane);

  void update(InputManager &input, float deltaTime);

  glm::mat4 getViewMatrix() const;
  glm::mat4 getProjectionMatrix() const;
  const glm::vec3 &getPos() const;

  float fov;
  float aspect;
  float nearPlane;
  float farPlane;

  float movementSpeed;
  float mouseSensitivity;
  float zoomSpeed;

  float yaw = 0.0;
  float pitch = 0.0;

  glm::vec3 position;
  glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::quat orientation;
};
