from setuptools import find_packages, setup

__version__ = '0.9.0'
URL = 'https://github.com/TorchSpatiotemporal/tsl'

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    'einops',
    'numpy',
    'pandas>=1.4,<1.5',
    'pytorch_lightning>=1.5,<1.9',
    'PyYAML',
    'scikit_learn',
    'scipy',
    'tables',
    'torchmetrics>=0.7,<0.11',
    'tqdm',
]

plot_requires = [
    'matplotlib',
    'mpld3'
]

experiment_requires = [
    'hydra-core',
    'omegaconf'
]

full_install_requires = plot_requires + experiment_requires + [
    'holidays',
    'neptune-client>=0.14,<0.17',
    'pytorch_fast_transformers'
]

doc_requires = full_install_requires + [
    'sphinx',
    'sphinx-design',
    'sphinx-copybutton',
    'sphinxext-opengraph',
    'sphinx-hoverxref',
    'myst-nb',
    'furo'
]

setup(
    name='torch_spatiotemporal',
    version=__version__,
    description='A PyTorch library for spatiotemporal data processing',
    author='Andrea Cini, Ivan Marisca',
    author_email='andrea.cini@usi.ch, ivan.marisca@usi.ch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=f'{URL}/archive/v{__version__}.tar.gz',
    license="MIT",
    keywords=[
        'pytorch',
        'pytorch-geometric',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-convolutional-networks',
        'temporal-graph-networks',
        'spatiotemporal-graph-neural-networks',
        'spatiotemporal-processing',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        'experiment': experiment_requires,
        'full': full_install_requires,
        'doc': doc_requires,
    },
    packages=find_packages(exclude=['examples*']),
)
