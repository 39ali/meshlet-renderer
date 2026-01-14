#include <chrono>

using Clock = std::chrono::steady_clock;

class Timer {
public:
  Timer() : last(Clock::now()) {}

  double tick() {
    auto now = Clock::now();
    std::chrono::duration<double> dt = now - last;
    last = now;
    return dt.count();
  }

private:
  Clock::time_point last;
};