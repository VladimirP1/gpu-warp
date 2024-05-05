#pragma once

#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

std::string read_file(char const *path) {
    std::string ret{};
    int fd = open(path, O_RDONLY);

    if (fd < 0) {
        std::runtime_error("cannot open file");
    }
    while (true) {
        char buf[1024];
        int nread = read(fd, buf, sizeof(buf));
        if (nread < 0) {
            std::runtime_error("error reading file");
        } else if (nread == 0) {
            break;
        }
        ret.append(buf, nread);
    }
    return ret;
}