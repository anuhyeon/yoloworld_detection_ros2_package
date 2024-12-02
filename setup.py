from setuptools import setup

package_name = 'yoloworld_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rcv',
    maintainer_email='rcv@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yoloworld_node = yoloworld_detection.yoloworld_node:main',
            'yoloworld_capture_frame_send_to_server_node = yoloworld_detection.yoloworld_node_save_frame:main',
        ],
        
    },
)
