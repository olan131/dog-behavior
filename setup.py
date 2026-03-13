"""Setup configuration for pet-behavior-clip."""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pet-behavior-clip",
    version="0.1.0",
    description="Pet behaviour anomaly detection via SigLIP zero-shot classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="pet-behavior-clip contributors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "ui"]),
    install_requires=[
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "torch>=2.0.0",
        "transformers>=4.38.0",
        "click>=8.1.0",
        "gradio>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "pet-behavior-clip=pet_behavior_clip.cli:main",
            "pet-behavior-ui=ui.app:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
)
