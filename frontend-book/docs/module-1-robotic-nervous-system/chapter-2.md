---
title: 'Chapter 2: Nodes, Topics, and Message Passing'
sidebar_position: 2
description: 'Understanding ROS 2 communication patterns: nodes, topics, and message passing'
---

# Chapter 2: Nodes, Topics, and Message Passing

## Learning Objectives

- Understand the fundamental ROS 2 communication concepts: nodes, topics, and messages
- Learn how to create publishers and subscribers
- Explore message passing patterns and data flow in ROS 2
- Gain practical experience with ROS 2 tools for monitoring communication

## Introduction

ROS 2's communication system is built on a distributed architecture where processes called "nodes" communicate with each other through "topics". This publish-subscribe model enables loose coupling between components, allowing for flexible and scalable robot systems. Understanding nodes, topics, and message passing is crucial for developing robust Physical AI systems that require real-time communication between sensors, actuators, and processing units.

## Core Theory

The fundamental concepts in ROS 2 communication include:

- **Nodes**: Processes that perform computation and communication
- **Topics**: Named buses over which nodes exchange messages
- **Messages**: Data structures that are passed between nodes
- **Publishers**: Nodes that send messages to topics
- **Subscribers**: Nodes that receive messages from topics
- **ROS Graph**: The network of nodes and their connections

Communication in ROS 2 is asynchronous and follows a data-centric approach where publishers send messages to topics without knowing who will receive them, and subscribers receive messages from topics without knowing who sent them.

## Practical Example

Let's examine a practical example of a publisher-subscriber pair with custom message types:

```python
# Custom message definition (my_robot_msgs/msg/SensorData.msg)
float64 temperature
float64 humidity
int32 sensor_id
string sensor_name
```

```python
# Publisher node: sensor_publisher.py
import rclpy
from rclpy.node import Node
from my_robot_msgs.msg import SensorData
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(SensorData, 'sensor_data', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = SensorData()
        msg.temperature = random.uniform(20.0, 30.0)
        msg.humidity = random.uniform(30.0, 70.0)
        msg.sensor_id = 1
        msg.sensor_name = "TemperatureHumiditySensor"

        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: Temp={msg.temperature:.2f}, Humidity={msg.humidity:.2f}')

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# Subscriber node: sensor_subscriber.py
import rclpy
from rclpy.node import Node
from my_robot_msgs.msg import SensorData

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')
        self.subscription = self.create_subscription(
            SensorData,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(
            f'Received sensor data: {msg.sensor_name} (ID: {msg.sensor_id}) - '
            f'Temperature: {msg.temperature:.2f}°C, Humidity: {msg.humidity:.2f}%'
        )

def main(args=None):
    rclpy.init(args=args)
    sensor_subscriber = SensorSubscriber()

    try:
        rclpy.spin(sensor_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Code Snippet

Example of using ROS 2 command-line tools to monitor communication:

```bash
# List all available topics
ros2 topic list

# Echo messages from a specific topic
ros2 topic echo /sensor_data

# Show information about a topic
ros2 topic info /sensor_data

# Publish a message directly from command line
ros2 topic pub /sensor_data my_robot_msgs/SensorData "{temperature: 25.0, humidity: 50.0, sensor_id: 1, sensor_name: 'ManualInput'}"

# Get topic statistics
ros2 topic hz /sensor_data
```

```python
# Quality of Service (QoS) configuration for message passing
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Configure QoS for real-time critical data
qos_profile = QoSProfile(
    depth=10,  # Number of messages to keep in queue
    reliability=ReliabilityPolicy.RELIABLE,  # Ensure delivery
    history=HistoryPolicy.KEEP_LAST  # Keep only the most recent messages
)

# Use QoS profile in publisher
publisher = node.create_publisher(SensorData, 'sensor_data', qos_profile)

# Use QoS profile in subscriber
subscription = node.create_subscription(
    SensorData,
    'sensor_data',
    callback_function,
    qos_profile
)
```

## Exercises

1. **Conceptual Question**: Explain the difference between ROS 1's centralized master-based architecture and ROS 2's decentralized approach. How does this affect communication reliability and scalability?

2. **Practical Exercise**: Create a ROS 2 package with custom message types for a robot's sensor data. Implement a publisher that simulates a camera node and a subscriber that processes the camera data.

3. **Code Challenge**: Design a communication system with multiple publishers and subscribers on the same topic. Implement a system where different sensor nodes publish to a common "robot_sensors" topic and a fusion node aggregates the data.

4. **Critical Thinking**: How do Quality of Service (QoS) policies in ROS 2 affect the reliability of Physical AI systems? Discuss the trade-offs between reliability and real-time performance in robot communication.

## Summary

This chapter explored the fundamental ROS 2 communication patterns: nodes, topics, and message passing. We learned how to create publishers and subscribers, work with custom message types, and use ROS 2 tools for monitoring communication. The publish-subscribe model provides a flexible, scalable communication framework essential for Physical AI systems that need to coordinate multiple sensors, actuators, and processing units.