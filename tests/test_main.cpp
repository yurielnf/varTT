#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>

TEST_CASE("hola", "[main]") {
  std::cout << "hola mundo\n";
  REQUIRE(true);
}
