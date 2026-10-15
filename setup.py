# @Maintainer: zack zhao

from setuptools import find_packages, setup


def get_version():
    try:
        from IVAS.__init__ import __version__

        return __version__

    except ImportError:
        raise ImportError("IVAS/__init__.py not found, cannot determine version.")


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ivas-sam",  # 包名
    version=get_version(),  # 版本号
    author="zack zhao",
    description="IVAS visual annotation workbench for image segmentation and temporal video annotation powered by SAM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "annotation tool",
        "segment anything",
        "image annotation",
        "semantic segmentation",
        "instance segmentation",
    ],
    license="Apache2.0",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    python_requires=">=3.8",  # python 版本要求
    install_requires=[  # 必须直接指明，不然pip包不会自动安装
        "imgviz",
        "scikit-image",
        "numpy",
        "opencv_python_headless",
        "pillow",
        "pyqt5",
        "python-vlc>=3.0.21203",
        "pyyaml",
        "torch>=2.3.0",
        "torchvision",
        "pycocotools",
        "timm",
        "shapely",
        "hydra-core>=1.3.2",
        "tqdm>=4.66.1",
        "fuzzywuzzy",
        "python-Levenshtein",
        "iopath",
        "orjson",
        "pydicom",
        "requests",
        "einops",
        "psutil",
        "ftfy",
        "regex",
    ],
    classifiers=[
        "Intended Audience :: Developers",  # 目标用户:开发者
        "Intended Audience :: Science/Research",  # 目标用户:学者
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "ivas-sam=IVAS.main:main",
            "ivas=IVAS.main:main",
            "ivat-sam=IVAS.main:main",
            "ivat=IVAS.main:main",
        ],
        "ivas.plugins": [],  # IVAS plugin registration
        "ivat.plugins": [],  # Previous product name
    },
)
