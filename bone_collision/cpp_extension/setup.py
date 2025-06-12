#!/usr/bin/env python3
"""
Setup script for bone_collision C++ extension module
Builds high-performance bone-based collision detection system
"""

from setuptools import setup, Extension
import subprocess
import sys
import os
import numpy

# Try to import pybind11, install if not available
try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("Installing pybind11...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11[global]"])
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext

def check_cpp_compiler():
    """Check if C++ compiler is available"""
    try:
        result = subprocess.run(['g++', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Found g++: {result.stdout.split()[0]} {result.stdout.split()[1]}")
            return True
    except FileNotFoundError:
        pass
    
    try:
        result = subprocess.run(['clang++', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Found clang++: {result.stdout.split()[0]} {result.stdout.split()[1]}")
            return True
    except FileNotFoundError:
        pass
    
    print("WARNING: No C++ compiler found!")
    print("Install build tools:")
    print("  Ubuntu/Debian: sudo apt-get install build-essential")
    print("  CentOS/RHEL: sudo yum groupinstall 'Development Tools'")
    print("  macOS: xcode-select --install")
    return False

def get_eigen_paths():
    """Try to find Eigen library"""
    eigen_locations = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        '/opt/homebrew/include/eigen3',  # macOS Homebrew
        '/usr/include/eigen3/Eigen',
        '/usr/local/include/eigen3/Eigen'
    ]
    
    found_paths = []
    for location in eigen_locations:
        if os.path.exists(location):
            found_paths.append(location)
            print(f"Found Eigen at: {location}")
    
    if not found_paths:
        print("WARNING: Eigen not found in standard locations!")
        print("Install Eigen3:")
        print("  Ubuntu/Debian: sudo apt-get install libeigen3-dev")
        print("  CentOS/RHEL: sudo yum install eigen3-devel")
        print("  macOS: brew install eigen")
        print("Will proceed without Eigen (may cause compilation errors)")
    
    return found_paths

# Check requirements
cpp_available = check_cpp_compiler()
eigen_paths = get_eigen_paths()

# Include directories
include_dirs = [
    pybind11.get_cmake_dir() + "/../include",
    numpy.get_include(),
    '/usr/include',
    '/usr/local/include',
]

# Add Eigen paths
include_dirs.extend(eigen_paths)

# Remove duplicates while preserving order
include_dirs = list(dict.fromkeys(include_dirs))

print(f"Include directories: {include_dirs}")

# Compiler flags
extra_compile_args = [
    "-O3",              # Maximum optimization
    "-march=native",    # Optimize for current CPU
    "-std=c++14",       # C++14 standard
    "-ffast-math",      # Fast math operations
    "-DWITH_OPENMP",    # Enable OpenMP if available
    "-fopenmp",         # OpenMP support
]

extra_link_args = [
    "-lgomp",  # Link OpenMP for parallel operations
]

# Platform-specific optimizations
if sys.platform == "darwin":  # macOS
    # Remove OpenMP on macOS by default (can be tricky to set up)
    extra_compile_args = [arg for arg in extra_compile_args if "openmp" not in arg.lower()]
    extra_link_args = [arg for arg in extra_link_args if "gomp" not in arg]
    
    # Add macOS-specific optimizations
    extra_compile_args.extend([
        "-msse4.2",         # SIMD instructions
        "-mavx",            # Advanced Vector Extensions
    ])

elif sys.platform.startswith("linux"):  # Linux
    extra_compile_args.extend([
        "-msse4.2",         # SIMD instructions
        "-mavx",            # Advanced Vector Extensions
        "-mavx2",           # AVX2 if available
    ])

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "bone_collision",
        sources=["bone_collision.cpp"],
        include_dirs=include_dirs,
        cxx_std=14,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="bone_collision",
    version="1.0.0",
    author="STAR-Robot Team",
    description="High-performance bone-based collision detection system",
    long_description="""
    High-performance bone-based collision detection system for STAR-Robot.
    Uses bone capsule filtering to dramatically reduce collision detection overhead
    from 38ms to <1ms while maintaining precision.
    
    Features:
    - Auto-tuning bone capsule radii
    - STAR skinning weight integration
    - Direct triangle-capsule collision math
    - Surface normal computation for robot correction
    """,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    setup_requires=[
        "pybind11>=2.6.0",
        "setuptools>=40.0.0",
        "wheel",
        "numpy>=1.19.0",
    ],
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)