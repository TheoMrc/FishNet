"""Setuptools build entry point for FishNet."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def get_extensions():
    """Build the optional Cython acceleration module used by FishNet."""
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension

    extensions = [
        Extension(
            "cutils.zones",
            ["cutils/zones.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ]
    return cythonize(extensions, language_level="3")


CORE_DEPENDENCIES = [
    "matplotlib>=3.7",
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "torch>=2.0",
    "tqdm>=4.65",
]
ANNOTATION_DEPENDENCIES = [
    "Flask>=3.0",
    "Pillow>=10",
    "scipy>=1.10",
]
TRAINING_DEPENDENCIES = [
    "h5py>=3.8",
    "imutils>=0.5",
    "opencv-python>=4.8",
    "openpyxl>=3.1",
    "wandb>=0.16",
]
ONNX_DEPENDENCIES = ["onnxruntime>=1.17"]

setup(
    name="fishnet-zebrafish",
    version="1.0.0",
    description="FishNet training, inference, and annotation tools for zebrafish larvae",
    url="https://github.com/TheoMrc/FishNet",
    long_description=README,
    long_description_content_type="text/markdown",
    author=(
        "Théo Mercé, PhD; Emilien Reaud, PhD student; Etienne Windels, Data scientist"
    ),
    packages=find_packages(exclude=["tests", ".github"]),
    include_package_data=True,
    package_data={
        "annotation_app": [
            "templates/*.html",
            "static/*.css",
            "static/*.js",
            "static/*.png",
            "static/*.svg",
        ],
        "cutils": ["*.pyi", "*.pyx"],
    },
    install_requires=(
        CORE_DEPENDENCIES
        + ANNOTATION_DEPENDENCIES
        + TRAINING_DEPENDENCIES
        + ONNX_DEPENDENCIES
    ),
    extras_require={
        "annotation": ANNOTATION_DEPENDENCIES,
        "training": TRAINING_DEPENDENCIES,
        "onnx": ONNX_DEPENDENCIES,
        "test": ["pytest>=8"],
        "format": ["ruff>=0.11", "ty>=0.0.1", "tox>=4"],
    },
    ext_modules=get_extensions(),
    entry_points={
        "console_scripts": [
            "fishnet-infer=fish_net.inference:main",
            "fishnet-train=fish_net.training:main",
            "fishnet-annotate=annotation_app.app:main",
        ]
    },
)
