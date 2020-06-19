import setuptools

setuptools.setup(
    name="ColloidScattering",
    version="0.0.1",
    author="David J. Pine",
    author_email="pine@nyu.edu",
    description="Python package for analysis of single & multiple, static & dynamic light scattering from colloids",
    url="https://github.com/djpine/ColloidScat",
    packages=[colloidscat],
    install_requires=['numpy', 'scipy']
)


