from setuptools import find_packages, setup

setup(
    name="knowledge-rep-llms",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().strip().split("\n"),
    author="Shivam Tyagi",
    author_email="st.shivamtyagi.01@gmail.com",
    description="Research project investigating knowledge representation in LLM MLP layers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FrostNT1/KnowledgeRep-LLMs",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
) 