from setuptools import setup, find_packages

setup(
    name="deepseek-v3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A from-scratch implementation of DeepSeek V3 model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/deepseek-v3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
