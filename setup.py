"""
Setup script for 404-AI project
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='404-ai',
    version='0.1.0',
    description='Factory Defect Recognition System',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='WinLakeLee',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
