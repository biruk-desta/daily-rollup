# Build an instrumented copy while leaving the shipped main.cpp untouched.
/void\* address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);/c\
            int flags = MAP_PRIVATE;\
            /* Linux's populate option is only enabled for this variant. */\
            if (std::getenv("R5_POPULATE")) flags |= probe_populate;\
            void* address = mmap(nullptr, size, PROT_READ, flags, fd, 0);
/^class Mapping/i\
#if defined(__linux__)\
constexpr int probe_populate = MAP_POPULATE;\
#else\
constexpr int probe_populate = 0;\
#endif
/if ((xcr0 \& 0xe6) == 0xe6/c\
    if (!std::getenv("R5_AVX2") \&\& (xcr0 \& 0xe6) == 0xe6 \&\& (b \& avx512) == avx512) return Scanner::Avx512;\
    if (std::getenv("R5_AVX512")) return Scanner::Scalar;
/            madvise(address, bytes, MADV_HUGEPAGE);/c\
            if (!std::getenv("R5_NO_THP")) madvise(address, bytes, MADV_HUGEPAGE);
/    ~CandidateArena()/a\
    void prefault() const {\
        if (!rows_) return;\
        size_t page = sysconf(_SC_PAGESIZE);\
        auto* bytes = reinterpret_cast<volatile unsigned char*>(rows_);\
        for (size_t i = 0; i < bytes_; i += page) bytes[i] = 0;\
    }
/    parallel(threads, input.chunks.size() + 1,/i\
    if (std::getenv("R5_PREFAULT")) {\
        parallel(threads, entries.size(), [&](unsigned, size_t i) { entries[i]->prefault(); });\
        trace.mark("prefault");\
    }
/^void write_users/,/^}/ {
  /    while (!users.empty())/i\
    static std::atomic_flag changed;\
    const bool make_wrong = std::getenv("R5_WRONG_ROW");
  /case 3: little_endian(bytes, row.events); break;/c\
                    case 3: {\
                        bool inject = make_wrong \&\& !changed.test_and_set(std::memory_order_relaxed);\
                        little_endian(bytes, row.events + inject);\
                        break;\
                    }
}
