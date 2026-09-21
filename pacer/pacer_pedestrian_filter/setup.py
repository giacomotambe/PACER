from setuptools import setup

package_name = 'pacer_pedestrian_filter'

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
    description='Toglie dalla point cloud del lidar i punti vicini ai pedoni '
                 'tracciati, prima della costmap di Nav2 — evitamento pedoni '
                 'resta esclusivo dell\'NMPC.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pedestrian_filter_node = pacer_pedestrian_filter.pedestrian_filter_node:main',
        ],
    },
)
