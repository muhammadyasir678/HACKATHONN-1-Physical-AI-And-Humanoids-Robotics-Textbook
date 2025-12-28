---
title: 'Chapter 1: ROS 2 Fundamentals and Architecture'
sidebar_position: 1
description: 'Introduction to ROS 2 concepts and architecture for Physical AI systems'
---

# Chapter 1: ROS 2 Fundamentals and Architecture

## Learning Objectives

- Understand the core concepts of ROS 2 and its architecture
- Learn about nodes, topics, services, and actions in ROS 2
- Explore how ROS 2 enables communication in Physical AI systems
- Gain familiarity with ROS 2 tools and development practices

## Introduction

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It is a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms. For Physical AI systems, ROS 2 provides the communication infrastructure necessary for coordinating sensors, actuators, and AI algorithms.

## Core Theory

ROS 2 is designed to be a flexible, distributed system that allows different components of a robot to communicate with each other. The core concepts include:

- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Goal-oriented communication with feedback
- **Parameters**: Configuration values that can be set at runtime
- **Lifecycle**: Management of node states

ROS 2 uses DDS (Data Distribution Service) as its underlying communication middleware, providing reliable, real-time communication between nodes.

## Practical Example

Let's examine a simple ROS 2 publisher-subscriber pattern that demonstrates basic communication:

```python
# Publisher node example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

```python
# Subscriber node example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

## Code Snippet

Complete the publisher-subscriber example by creating a launch file:

```python
# launch file example
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='minimal_publisher',
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='minimal_subscriber',
        ),
    ])
```

```bash
# Common ROS 2 commands for running the example
ros2 run demo_nodes_py talker
ros2 run demo_nodes_py listener
ros2 topic list
ros2 topic echo /topic
```

## Exercises

1. **Conceptual Question**: Explain the difference between topics and services in ROS 2. When would you use each communication pattern?

2. **Practical Exercise**: Create a simple ROS 2 package with a publisher that publishes sensor data (e.g., temperature readings) and a subscriber that logs this data.

3. **Code Challenge**: Implement a ROS 2 service that accepts two numbers and returns their sum. Create both the service server and client nodes.

4. **Critical Thinking**: How does ROS 2's DDS-based communication architecture support the requirements of Physical AI systems that need real-time, reliable communication between components?

## Summary

This chapter introduced the fundamental concepts of ROS 2, including its architecture, core communication patterns, and development tools. ROS 2 provides the communication infrastructure necessary for Physical AI systems to coordinate sensors, actuators, and AI algorithms effectively.