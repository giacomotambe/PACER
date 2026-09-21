import os
from glob import glob

from setuptools import setup

package_name = 'pacer_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Giacomo',
    maintainer_email='giacomotambe@gmail.com',
    description='Ambiente di test Gazebo (gz-sim) per pacer-tupac-code-ros: '
                 'spawna Jo, aggancia Nav2/PacerController e simula i pedoni '
                 'al posto di Qualisys.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_pedestrian_publisher = pacer_sim.sim_pedestrian_publisher:main',
        ],
    },
)
