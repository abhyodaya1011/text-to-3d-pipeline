from setuptools import setup, find_packages

setup(
    name="ringgen",
    version="0.1.0",
    description="Streamlined 3D Ring Generation Pipeline for Colab",
    author="RingGen Contributors",
    author_email="info@ringgen.example.com",
    url="https://github.com/abhyodaya1011/text-to-3d-pipeline",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "trimesh>=4.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "plotly>=5.13.0",
        "pandas>=1.5.0",
        "requests>=2.28.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
