#define main shopping_main
#include "main.cpp"
#undef main

int main() {
    Ids ids('u');
    std::vector<std::string> values = {
        "0", "00", "01", "9", "A", "A0", "A1234567", "A12345678", "Z", "a", "a0", "q", "qq",
        "u", "u0000000", "u00000000", "u00000001", "u0000000A", "u0000000a", "U00000000",
        "u1234567", "u12345678", "u123456789", "u99999999", "uA2345678", "ua2345678",
        "u~~~~~~~", "u~~~~~~~~", std::string("u\0", 2), std::string("u\0" "0000000", 9),
        std::string(20, 'A'), std::string(300, 'A'), "\xc3\xa9", "\xe4\xb8\xad"
    };
    for (unsigned a = 1; a < 128; ++a) {
        values.emplace_back(1, char(a));
        values.push_back("u1234567" + std::string(1, char(a)));
    }
    std::vector<Person> keys;
    for (const auto& value : values) {
        keys.push_back(ids.key(value));
        char bytes[9];
        if (ids.name(keys.back(), bytes) != value) return 1;
    }
    for (size_t a = 0; a < values.size(); ++a)
        for (size_t b = 0; b < values.size(); ++b)
            if (ids.less(keys[a], keys[b]) != (values[a] < values[b])) return 1;
}
