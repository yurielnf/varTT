#include "catch.hpp"
#include "mps.h"

TEST_CASE("mps canonization", "[mps]") {
  MPS x(20, 3);
  x.FillRandu({3, 2, 3});
  SECTION("initialization and compatibility") {
    for (int i = 0; i < 2; i++) {
      x.PrintSizes();
      int m = 1;
      for (auto t : x.M) {
        REQUIRE(t.dim[0] == m);
        REQUIRE(t.dim[1] == 2);
        REQUIRE(t.dim[2] <= x.m);
        m = t.dim[2];
      }
      x.Canonicalize();
    }
  }
  SECTION("norm") {
    x.Canonicalize();
    REQUIRE(Norm(x.C) == 1);
  }
}
