#include <cmath>

double soft_thresh(const double &val, const double thresh) {
  return std::abs(val) < thresh ? 0 : (val > 0 ? val - thresh : val + thresh);
}
