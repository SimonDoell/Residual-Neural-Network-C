cmake_minimum_required(VERSION 4.2.3)
project(Boilerplate)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_EXTENSIONS OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g -Wall -Werror -Wno-unused)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # -fopt-info-vec-optimized
    add_definitions(-DNDEBUG)
    add_compile_options(-O3    -Wall -Werror -Wno-unused)
endif()

add_executable(Boilerplate src/main.c)
target_link_libraries(Boilerplate m)