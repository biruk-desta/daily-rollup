#include "isa.hpp"

#include <cstdio>
#include <unistd.h>

int main(int, char** argv) {
    if (supported_isa(SHOPPING_VARIANT)) execv("/shopping-tuned", argv);
    execv("/shopping-plain", argv);
    std::perror("shopping");
    return 127;
}
