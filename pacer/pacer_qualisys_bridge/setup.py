from setuptools import setup

package_name = 'pacer_qualisys_bridge'

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
    description='Ponte Qualisys -> pacer_msgs/PedestrianTrackArray.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qualisys_bridge_node = pacer_qualisys_bridge.qualisys_bridge_node:main',
        ],
    },
)
