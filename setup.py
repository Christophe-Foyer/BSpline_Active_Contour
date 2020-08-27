import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BSpline Active Contours",
    version="1.0.0",
    author="Christophe Foyer",
    author_email="christophe.foyer.19@ucl.ac.uk",
    description="2D and 3D Active Contours using BSpline.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Christophe-Foyer/BSpline_Active_Contour",
    install_requires =[
        'scipy >= 1.5.2',
        'scikit-image >= 0.16.2',
        'numpy >= 1.19.1',
        'vg >= 1.9.0',
        'geomdl >= 5.2.9',
        'pyvista >= 0.24.2',
        'matplotlib >= 3.2.2',
        'plotly >= 4.6.0',
        'pillow >= 7.2.0',
        'pydicom >= 1.4.2',
        'pytransform3d >= 1.2.1',
        'IPython',
        ],
    packages=setuptools.find_packages(),
    package_data={'nurbs_active_contour': ['nurbs_active_contour']},
    keywords=['snakes', 'active contour', 'BSpline', 'image segmentation',
              'surface fitting', 'curve fitting', 'computer vision'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.6',
)