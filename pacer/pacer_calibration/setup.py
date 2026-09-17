from setuptools import setup

package_name = 'pacer_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Giacomo',
    maintainer_email='giacomotambe@gmail.com',
    description='Calibrazione TUPAC online + servizi on-demand di training/ricalibrazione.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibration_node = pacer_calibration.calibration_node:main',
        ],
    },
)
