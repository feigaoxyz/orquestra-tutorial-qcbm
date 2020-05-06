import setuptools

setuptools.setup(
    name="qcbm",
    package_dir={"": "python"},
    packages=setuptools.find_namespace_packages(include=["qcbm"]),
    install_requires=["z-quantum-core"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
