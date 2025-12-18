cmake_minimum_required(VERSION 3.5)
project(hmc)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hmc hmc.cpp)

target_link_libraries(hmc PRIVATE Eigen3::Eigen)
target_link_libraries(hmc PRIVATE nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(hmc PRIVATE /W4 /Ox)
else()
    target_compile_options(hmc PRIVATE -Wall -Wextra -O3 -march=native)
endif()
