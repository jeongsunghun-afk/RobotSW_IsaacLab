# real_runner_bipedleg — r2s_biped_leg 실기 UDP↔SharedMem 브리지 (라즈베리파이용)
#
# 파이 실빌드 (RobotEmbedded/RobotSharedLib 설치된 상태):
#   mkdir -p build && cd build && cmake .. && make
# 로컬 검증 빌드 (로봇 없음, stub plant):
#   mkdir -p build && cd build && cmake -DSTUB_SHM=ON .. && make
cmake_minimum_required(VERSION 3.16)
project(real_runner_bipedleg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O2 -Wall)

option(STUB_SHM "실기 libRobotSharedMem 대신 로컬 테스트용 stub 사용" OFF)

add_executable(real_runner_bipedleg real_runner_bipedleg.cpp)

if(STUB_SHM)
    target_sources(real_runner_bipedleg PRIVATE stub/RobotSharedMem_stub.cpp)
    # stub 디렉터리를 먼저 뒤지게 해 <RobotSharedMem.h> 가 stub 으로 해석되게 한다.
    target_include_directories(real_runner_bipedleg BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stub)
    target_compile_definitions(real_runner_bipedleg PRIVATE STUB_SHM=1)
else()
    # 실기: /usr/include/RobotSharedMem.h + /usr/lib/libRobotSharedMem.so (RobotSharedLib 설치본)
    target_link_libraries(real_runner_bipedleg RobotSharedMem)
endif()

target_link_libraries(real_runner_bipedleg pthread rt)
