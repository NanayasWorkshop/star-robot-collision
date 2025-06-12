#!/usr/bin/env python3
"""
Setup script for fast_collision C++ extension module
Builds high-performance FCL collision detection with BVH refitting
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

def get_fcl_config():
    """Get FCL include and library paths"""
    try:
        # Try pkg-config first
        include_dirs = subprocess.check_output(['pkg-config', '--cflags-only-I', 'fcl']).decode().strip()
        include_dirs = [path[2:] for path in include_dirs.split() if path.startswith('-I')]
        
        lib_dirs = subprocess.check_output(['pkg-config', '--libs-only-L', 'fcl']).decode().strip()
        lib_dirs = [path[2:] for path in lib_dirs.split() if path.startswith('-L')]
        
        libraries = subprocess.check_output(['pkg-config', '--libs-only-l', 'fcl']).decode().strip()
        libraries = [lib[2:] for lib in libraries.split() if lib.startswith('-l')]
        
        return include_dirs, lib_dirs, libraries
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to standard locations
        print("pkg-config not found, using standard FCL paths")
        include_dirs = ['/usr/include', '/usr/local/include']
        lib_dirs = ['/usr/lib', '/usr/local/lib', '/usr/lib/x86_64-linux-gnu']
        libraries = ['fcl']
        return include_dirs, lib_dirs, libraries

def check_fcl_installation():
    """Check if FCL is properly installed"""
    # Common FCL header locations
    fcl_locations = [
        '/usr/include/fcl/fcl.h',
        '/usr/local/include/fcl/fcl.h',
        '/usr/include/fcl/fcl.hpp',
        '/usr/local/include/fcl/fcl.hpp'
    ]
    
    fcl_found = False
    fcl_path = None
    
    for location in fcl_locations:
        if os.path.exists(location):
            fcl_found = True
            fcl_path = os.path.dirname(os.path.dirname(location))  # Get parent of fcl directory
            print(f"Found FCL header at: {location}")
            break
    
    if not fcl_found:
        print("WARNING: FCL headers not found in standard locations!")
        print("Locations checked:")
        for loc in fcl_locations:
            print(f"  {loc}")
        print("Make sure FCL is installed:")
        print("  sudo apt-get install libfcl-dev libeigen3-dev libccd-dev")
    
    return fcl_found, fcl_path

# Check FCL installation
fcl_found, fcl_path = check_fcl_installation()

# Get FCL configuration
include_dirs, lib_dirs, libraries = get_fcl_config()

# Add FCL path if found
if fcl_path:
    include_dirs.insert(0, fcl_path)

# Add common include directories
include_dirs.extend([
    '/usr/include',
    '/usr/local/include',
    '/usr/include/fcl',
    '/usr/local/include/fcl',
    '/usr/include/eigen3',  # Eigen (required by FCL)
    '/usr/local/include/eigen3',
])

# Remove duplicates while preserving order
include_dirs = list(dict.fromkeys(include_dirs))

print(f"Include directories: {include_dirs}")
print(f"Library directories: {lib_dirs}")
print(f"Libraries: {libraries}")

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "fast_collision",
        sources=["fast_collision.cpp"],
        include_dirs=include_dirs,
        library_dirs=lib_dirs,
        libraries=libraries,
        cxx_std=14,  # FCL requires C++14
        extra_compile_args=[
            "-O3",  # Maximum optimization
            "-march=native",  # Optimize for current CPU
            "-DWITH_OPENMP",  # Enable OpenMP if available
        ],
        extra_link_args=[
            "-lgomp",  # Link OpenMP for parallel operations
        ],
    ),
]

setup(
    name="fast_collision",
    version="1.0.0",
    author="STAR-Robot Team",
    description="High-performance FCL collision detection with BVH refitting",
    long_description="High-performance FCL collision detection with BVH refitting for STAR-Robot system",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    setup_requires=[
        "pybind11>=2.6.0",
        "setuptools>=40.0.0",
        "wheel",
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
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)