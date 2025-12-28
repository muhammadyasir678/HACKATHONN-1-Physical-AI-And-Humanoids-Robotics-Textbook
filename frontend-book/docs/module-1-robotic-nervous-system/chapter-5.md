---
title: 'Chapter 5: ROS 2 Tools and Debugging'
sidebar_position: 5
description: 'Essential ROS 2 tools for development, debugging, and system monitoring'
---

# Chapter 5: ROS 2 Tools and Debugging

## Learning Objectives

- Master essential ROS 2 command-line tools for system inspection
- Learn debugging techniques for ROS 2 applications
- Understand system monitoring and profiling approaches
- Gain experience with visualization tools like RViz2

## Introduction

Effective development and debugging of ROS 2 systems require proficiency with the ecosystem of tools provided by ROS 2. These tools enable developers to inspect system state, monitor communication, visualize robot data, and debug complex multi-node systems. For Physical AI systems, where multiple sensors, actuators, and processing nodes must work in coordination, these tools are essential for ensuring proper operation and diagnosing issues.

## Core Theory

ROS 2 provides several categories of tools:

- **System Inspection**: Tools to examine the ROS graph, nodes, topics, and services
- **Communication Monitoring**: Tools to inspect message content and communication patterns
- **Visualization**: Tools like RViz2 for visualizing robot state and sensor data
- **Debugging**: Tools for profiling, logging, and debugging node behavior
- **Performance Analysis**: Tools for measuring system performance and bottlenecks

The ROS 2 toolset enables comprehensive analysis of robot systems from the architectural level down to individual message exchanges.

## Practical Example

Let's examine the essential ROS 2 command-line tools:

```bash
# System inspection tools
ros2 node list                    # List all active nodes
ros2 node info /robot_controller  # Get detailed info about a specific node
ros2 topic list                   # List all active topics
ros2 topic info /cmd_vel          # Get detailed info about a topic
ros2 service list                 # List all available services
ros2 action list                  # List all available actions

# Communication monitoring
ros2 topic echo /sensor_data      # Print messages from a topic
ros2 service call /move_robot my_robot_msgs/srv/MoveRobot "{x: 1.0, y: 2.0, theta: 0.5}"
ros2 param list /robot_controller # List parameters of a node
ros2 param get /robot_controller max_velocity  # Get parameter value
```

Example of using rqt for visualization:

```python
# Python script to demonstrate tool usage programmatically
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')
        self.publisher = self.create_publisher(String, 'debug_info', 10)

        # Create a timer to publish debug information
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Log different levels of messages
        self.get_logger().info('Debug node initialized')
        self.get_logger().warn('This is a warning message')
        self.get_logger().error('This is an error message')

    def timer_callback(self):
        msg = String()
        msg.data = f'Debug info published at {time.time()}'
        self.publisher.publish(msg)
        self.get_logger().debug(f'Published: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    debug_node = DebugNode()

    try:
        rclpy.spin(debug_node)
    except KeyboardInterrupt:
        pass
    finally:
        debug_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Code Snippet

Example of using ROS 2 logging and debugging:

```python
# Advanced debugging with custom logging
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.logging import LoggingSeverity
from std_msgs.msg import Float32
import traceback

class AdvancedDebugNode(Node):
    def __init__(self):
        super().__init__('advanced_debug_node')

        # Create publisher for diagnostic data
        qos_profile = QoSProfile(depth=10)
        self.diag_publisher = self.create_publisher(Float32, 'diagnostics', qos_profile)

        # Set up custom logging
        self.get_logger().set_level(LoggingSeverity.DEBUG)

        # Create timer with error handling
        self.timer = self.create_timer(0.1, self.safe_timer_callback)

        self.counter = 0

    def safe_timer_callback(self):
        try:
            # Simulate some work that might fail
            result = self.perform_calculation()

            # Publish diagnostic information
            msg = Float32()
            msg.data = float(result)
            self.diag_publisher.publish(msg)

            self.get_logger().debug(f'Calculation result: {result}')

        except Exception as e:
            # Log the error with traceback
            self.get_logger().error(f'Error in timer callback: {str(e)}')
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

            # Publish error indicator
            error_msg = Float32()
            error_msg.data = -1.0  # Error indicator
            self.diag_publisher.publish(error_msg)

    def perform_calculation(self):
        # Simulate a calculation that might fail
        if self.counter % 100 == 99:  # Every 100th iteration
            raise ValueError("Simulated calculation error")

        result = (self.counter * 2.5) % 100
        self.counter += 1
        return result

def main(args=None):
    rclpy.init(args=args)
    debug_node = AdvancedDebugNode()

    try:
        rclpy.spin(debug_node)
    except KeyboardInterrupt:
        debug_node.get_logger().info('Node interrupted by user')
    except Exception as e:
        debug_node.get_logger().error(f'Unexpected error: {str(e)}')
        debug_node.get_logger().error(f'Traceback: {traceback.format_exc()}')
    finally:
        debug_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Useful debugging commands:

```bash
# Logging and debugging commands
ros2 run my_package advanced_debug_node --ros-args --log-level debug

# Monitoring system performance
ros2 doctor  # Check ROS 2 system health
ros2 run demo_nodes_cpp talker --ros-args --log-level info

# Using rqt tools
rqt
rqt_graph                    # Visualize ROS graph
rqt_console                  # View logs from all nodes
rqt_plot /debug_info/data    # Plot numerical values
rqt_bag                      # Record and replay messages

# Profiling and performance
ros2 run topic_tools relay /original_topic /monitored_topic
ros2 lifecycle list          # For lifecycle nodes
```

## Exercises

1. **Conceptual Question**: Explain how ROS 2's distributed logging system works. How does it differ from traditional logging approaches in multi-process applications?

2. **Practical Exercise**: Create a ROS 2 node that publishes diagnostic information and use various ROS 2 tools to monitor its operation. Experiment with different logging levels and message types.

3. **Code Challenge**: Implement a monitoring node that subscribes to multiple topics and publishes aggregated diagnostic information. Use ROS 2 tools to visualize the system state.

4. **Critical Thinking**: How do ROS 2 debugging tools compare to traditional debugging approaches? What are the challenges of debugging distributed systems in Physical AI applications?

## Summary

This chapter covered essential ROS 2 tools for development, debugging, and system monitoring. We explored command-line tools for system inspection, communication monitoring, and parameter management, as well as visualization tools like RViz2 and rqt. These tools are crucial for developing and maintaining complex Physical AI systems where multiple nodes must work in coordination. Understanding and effectively using these tools is essential for efficient development and troubleshooting of robot systems.