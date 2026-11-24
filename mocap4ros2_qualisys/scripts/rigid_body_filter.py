#!/usr/bin/env python3
# Ripubblica uno o più rigid body da /rigid_bodies come PoseStamped su /<nome>/pose
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from mocap_msgs.msg import RigidBodies
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class RigidBodyFilter(Node):

    def __init__(self, body_names):
        super().__init__('rigid_body_filter')
        self.pubs = {}
        for name in body_names:
            self.pubs[name] = self.create_publisher(PoseStamped, f'/{name}/pose', 1000)
            self.get_logger().info(f"Filtro '{name}' -> /{name}/pose")
        self.create_subscription(RigidBodies, '/rigid_bodies', self.callback, 1000)

    def callback(self, msg):
        for rb in msg.rigidbodies:
            pub = self.pubs.get(rb.rigid_body_name)
            if pub is not None:
                out = PoseStamped()
                # Stesso timestamp del driver, così resta allineato a /rigid_bodies
                out.header = msg.header
                out.pose = rb.pose
                pub.publish(out)


def main():
    body_names = sys.argv[1:] or ['traquad']
    rclpy.init()
    node = None
    try:
        node = RigidBodyFilter(body_names)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # Interrotto durante l'avvio: il contesto è già chiuso
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
