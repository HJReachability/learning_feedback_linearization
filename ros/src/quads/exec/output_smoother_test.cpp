#include <quads/scalar_output_smoother.h>
#include <quads/types.h>

#include <math.h>
#include <iostream>

using namespace quads;


int main(void) {
  ScalarOutputSmoother smoother;

  // Feed in a sinusoid.
  const double dt = 0.002; // s

  for (double t = 0.0; t < 1000.0*dt; t += dt) {
    const double y = std::sin(t);
    smoother.Update(y, dt);

    // Print out stuff.
    std::cout << "state is: " << smoother.State().transpose() << std::endl;
    std::cout << "cov is:\n" << smoother.Cov() << std::endl;

    // Compute state error.
    Vector4d truth;
    truth << y, std::cos(t), -y, -std::cos(t);

    std::cout << "state error: " << (smoother.State() - truth).transpose() << std::endl;
  }

  return EXIT_SUCCESS;
}
