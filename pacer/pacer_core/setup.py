from setuptools import setup, find_packages

package_name = 'pacer_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'torch'],
    zip_safe=True,
    maintainer='Giacomo',
    maintainer_email='giacomotambe@gmail.com',
    description='Libreria NMPC/TUPAC pura Python, senza dipendenze ROS2 — riusata dagli altri nodi.',
    license='Apache-2.0',
    tests_require=['pytest'],
)
