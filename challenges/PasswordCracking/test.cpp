#include <gtest/gtest.h>

#include "solve.h"
TEST(PasswordCracking, T1) {
  EXPECT_EQ("abc", PasswordCracking(537089824, 3, 2));
}
TEST(PasswordCracking, T2) {
  EXPECT_EQ("abc", PasswordCracking(440920331, 3, 1));
}

// target_hash = 1439553473
// password_length = 8
// R = 10
// Output: [108 117  98 120 111 109 105  97   0]
// Expected: [0 0 0 0 0 0 0 0 0]
// Max difference: 120