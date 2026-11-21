#pragma once

class ReadFile {
    int fd_;

public:
    explicit ReadFile(const fs::path& path) : fd_(open(path.c_str(), O_RDONLY)) {
        if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
    }
    ReadFile(const ReadFile&) = delete;
    ~ReadFile() { close(fd_); }
    size_t size() const {
        struct stat info{};
        if (fstat(fd_, &info) != 0) throw std::runtime_error("cannot stat input");
        return static_cast<size_t>(info.st_size);
    }

    void read_at(size_t offset, char* bytes, size_t size) const {
        if (lseek(fd_, offset, SEEK_SET) < 0) throw std::runtime_error("cannot seek input");
        while (size) {
            ssize_t count = read(fd_, bytes, size);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("short read from input");
            bytes += count;
            size -= count;
        }
    }

    size_t after_line(size_t offset, size_t end) const {
        // Find the next complete physical line without reading the rest of the chunk.
        char bytes[4096];
        while (offset < end) {
            size_t count = std::min(sizeof(bytes), end - offset);
            read_at(offset, bytes, count);
            if (auto* newline = static_cast<const char*>(std::memchr(bytes, '\n', count)))
                return offset + (newline - bytes) + 1;
            offset += count;
        }
        return end;
    }
};

class FileContents {
    std::unique_ptr<char[]> storage_;
    size_t size_;

public:
    explicit FileContents(const fs::path& path) {
        ReadFile file(path);
        size_ = file.size();
        storage_ = std::make_unique_for_overwrite<char[]>(std::max<size_t>(1, size_));
        if (size_) file.read_at(0, storage_.get(), size_);
    }
    View bytes() const { return {storage_.get(), size_}; }
};

class BytePool {
    std::vector<std::unique_ptr<char[]>> blocks_;
    char* next_ = nullptr;
    size_t left_ = 0;

public:
    View copy(View text) {
        if (text.empty()) return {};
        if (text.size() > left_) {
            left_ = std::max<size_t>(64 << 10, text.size());
            blocks_.push_back(std::make_unique_for_overwrite<char[]>(left_));
            next_ = blocks_.back().get();
        }
        char* start = next_;
        std::memcpy(start, text.data(), text.size());
        next_ += text.size();
        left_ -= text.size();
        return {start, text.size()};
    }
};

class StreamInput {
    struct Range { size_t file, offset, length; };
    std::vector<fs::path> paths_;
    std::vector<Range> chunks_;

public:
    explicit StreamInput(const fs::path& directory) {
        for (const auto& entry : fs::directory_iterator(directory))
            if (entry.path().extension() == ".csv" && entry.is_regular_file())
                paths_.push_back(entry.path());
        std::sort(paths_.begin(), paths_.end());
        for (size_t index = 0; index < paths_.size(); ++index) {
            ReadFile file(paths_[index]);
            size_t end = file.size();
            for (size_t start = file.after_line(0, end); start < end;) {
                size_t next = file.after_line(std::min(end, start + (2 << 20)), end);
                chunks_.push_back({index, start, next - start});
                start = next;
            }
        }
    }

    class Reader {
        std::unique_ptr<ReadFile> file_;
        const StreamInput* input_ = nullptr;
        size_t file_index_ = SIZE_MAX;
        std::unique_ptr<char[]> bytes_;
        size_t capacity_ = 0;

    public:
        View chunk(const StreamInput& input, size_t index) {
            auto [file, offset, length] = input.chunks_[index];
            if (&input != input_ || file != file_index_) {
                file_ = std::make_unique<ReadFile>(input.paths_[file]);
                file_index_ = file;
                input_ = &input;
            }
            if (length > capacity_) {
                capacity_ = std::bit_ceil(length);
                bytes_ = std::make_unique_for_overwrite<char[]>(capacity_);
            }
            file_->read_at(offset, bytes_.get(), length);
            return {bytes_.get(), length};
        }
    };

    size_t size() const { return chunks_.size(); }
    size_t estimated_rows() const {
        size_t rows = 0;
        for (auto range : chunks_) rows += range.length / 64;
        return rows;
    }
};
