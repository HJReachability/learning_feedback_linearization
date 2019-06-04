#include <quads/scalar_output_smoother.h>
#include <quads/types.h>

#include <math.h>
#include <iostream>

using namespace quads;


int main(void) {
  ScalarOutputSmoother smoother;

  // Feed in a sinusoid.
  const double dt = 0.01; // s

  for (double t = 0.0; t < 10.0*dt; t += dt) {
    const double y = std::sin(t);
    smoother.Update(y, dt);

    // Print out stuff.
    std::cout << "state is: " << smoother.X() << ", " << smoother.XDot1() << ", " << smoother.XDot2() << ", " << smoother.XDot3() << std::endl;
    std::cout << "cov is:\n" << smoother.Cov() << std::endl;
  }

  return EXIT_SUCCESS;
}
