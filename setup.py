"""
NDR Platform v2.1.0 - Production-Ready Network Detection & Response
Setup configuration for proper package installation
"""

from setuptools import setup, find_packages

setup(
    name="ndr-platform",
    version="2.1.0",
    description="Production-Ready Network Detection & Response Platform with Critical Fixes",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Shizukaax",
    author_email="justinchua@tunglok.com",
    url="https://github.com/Shizukaax/ndr-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies will be read from requirements.txt
        "streamlit>=1.28.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.0",
        "plotly>=5.16.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ndr-platform=app.main:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    include_package_data=True,
)
