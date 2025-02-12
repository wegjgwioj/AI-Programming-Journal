# 基于WebRTC的低延迟音视频传输系统

## 技术栈

- C++
- WebRTC
- FFmpeg
- Netty

## 核心问题

- 弱网环境下抗丢包与音画同步

## 亮点

1. **NACK重传与FEC前向纠错混合策略**
   - 实现了基于NACK（Negative Acknowledgment）和FEC（Forward Error Correction）的混合策略，提升了在弱网环境下的抗丢包能力。
2. **基于JitterBuffer的自适应播放缓冲控制**
   - 通过JitterBuffer实现了自适应的播放缓冲控制，保证了音视频的同步和流畅播放。
3. **端到端延迟<200ms（1080P@30fps）**
   - 实现了1080P@30fps条件下，端到端延迟小于200毫秒。

## 文件结构

- `src/`: 源代码目录
- `include/`: 头文件目录
- `lib/`: 第三方库目录
- `docs/`: 文档目录
- `tests/`: 测试文件目录

## 编译与运行

### 环境依赖

- C++ 编译器（支持 C++11 或以上）
- CMake
- WebRTC 库
- FFmpeg 库
- Netty 库

## 实现项目

### 步骤

1. **克隆项目代码**

   ```bash
   git clone https://github.com/your-repo/AI-Programming-Journal.git
   cd AI-Programming-Journal/WebRTC
   ```
2. **安装依赖**
   确保已安装以下依赖：

   - C++ 编译器（支持 C++11 或以上）
   - CMake
   - WebRTC 库
   - FFmpeg 库
   - Netty 库
3. **编译项目**

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
4. **运行项目**

   ```bash
   ./webrtc_project_executable
   ```
5. **测试项目**
   运行测试文件以确保项目正常工作：

   ```bash
   cd tests
   ./run_tests.sh
   ```

### 注意事项

- 确保网络环境稳定，以获得最佳性能。
- 根据需要调整缓冲区大小和重传策略，以适应不同的网络条件。
