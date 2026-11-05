@echo off
REM Build llama-cpp-python with MSVC (CPU backend) on Windows.
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VSPATH=%%i
if not defined VSPATH (
  echo Visual Studio not found. Install "Desktop development with C++".
  exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set CMAKE_GENERATOR=Ninja
set CMAKE_ARGS=-DGGML_CUDA=off -DGGML_NATIVE=ON
pip install llama-cpp-python --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
