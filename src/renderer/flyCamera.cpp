#include "flyCamera.h"
#include <glm/gtc/constants.hpp>
#include <iostream>

FlyCamera::FlyCamera(float fov, float aspect, float nearPlane, float farPlane)
    : position(0.0f, 0.0f, 7.0f),
      orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)), fov(fov), aspect(aspect),
      nearPlane(nearPlane), farPlane(farPlane), movementSpeed(3.0f),
      mouseSensitivity(0.1f), zoomSpeed(5.0f) {}

void FlyCamera::update(InputManager &input, float deltaTime) {

  // Create yaw and pitch quaternions
  double dx = 0;
  double dy = 0;
  input.getMouseDelta(dx, dy);

  yaw += dx * mouseSensitivity;
  pitch += dy * mouseSensitivity;

  if (pitch > 89.0f)
    pitch = 89.0f;
  if (pitch < -89.0f)
    pitch = -89.0f;

  glm::quat qYaw = glm::angleAxis(glm::radians(-yaw), glm::vec3(0, 1, 0));
  glm::quat qPitch = glm::angleAxis(glm::radians(pitch), glm::vec3(1, 0, 0));

  orientation = glm::normalize(qYaw * qPitch);

  glm::vec3 forward = orientation * glm::vec3(0, 0, -1);
  glm::vec3 right = orientation * glm::vec3(1, 0, 0);

  float velocity = movementSpeed * deltaTime;
  if (input.isKeyPressed(GLFW_KEY_W))
    position += forward * velocity;
  if (input.isKeyPressed(GLFW_KEY_S))
    position -= forward * velocity;
  if (input.isKeyPressed(GLFW_KEY_A))
    position -= right * velocity;
  if (input.isKeyPressed(GLFW_KEY_D))
    position += right * velocity;

  // zoom
  fov -= input.getScrollY() * zoomSpeed;
  if (fov < 1.0f)
    fov = 1.0f;
  if (fov > 120.0f)
    fov = 120.0f;
}

glm::mat4 FlyCamera::getViewMatrix() const {
  glm::mat4 viewRotation = glm::mat4_cast(orientation);
  viewRotation = glm::transpose(viewRotation); // Transpose for inverse
  glm::mat4 viewTranslation = glm::translate(glm::mat4(1.0f), -position);
  return viewRotation * viewTranslation;
}

const glm::vec3 &FlyCamera::getPos() const { return position; }

glm::mat4 FlyCamera::getProjectionMatrix() const {
  return glm::perspectiveZO(glm::radians(fov), aspect, nearPlane, farPlane);
}
