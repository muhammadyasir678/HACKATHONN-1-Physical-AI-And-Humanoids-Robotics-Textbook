---
title: 'Chapter 3: Services and Actions in ROS 2'
sidebar_position: 3
description: 'Understanding ROS 2 services and actions for synchronous and goal-oriented communication'
---

# Chapter 3: Services and Actions in ROS 2

## Learning Objectives

- Understand ROS 2 services for synchronous request/response communication
- Learn about ROS 2 actions for goal-oriented, long-running operations
- Explore when to use services vs. actions vs. topics
- Gain practical experience with service and action clients and servers

## Introduction

While topics enable asynchronous communication in ROS 2, services and actions provide mechanisms for synchronous communication and goal-oriented operations. Services offer request/response patterns similar to remote procedure calls, while actions are designed for long-running operations that require feedback and the ability to cancel. These communication patterns are essential for Physical AI systems that need to coordinate complex, multi-step tasks and ensure completion of critical operations.

## Core Theory

ROS 2 provides three primary communication patterns:

- **Topics**: Asynchronous, many-to-many communication via publish/subscribe
- **Services**: Synchronous, request/response communication for immediate results
- **Actions**: Goal-oriented communication for long-running operations with feedback

Services are appropriate for operations that:
- Have a clear request and response
- Complete quickly
- Don't require intermediate feedback
- Need guaranteed delivery and response

Actions are appropriate for operations that:
- Take a long time to complete
- Require intermediate feedback
- Need to be cancellable
- Have status updates during execution

## Practical Example

Let's examine examples of services and actions in ROS 2:

```python
# Service definition: my_robot_msgs/srv/MoveRobot.srv
float64 x
float64 y
float64 theta
---
bool success
string message
```

```python
# Service server: move_robot_server.py
import rclpy
from rclpy.node import Node
from my_robot_msgs.srv import MoveRobot

class MoveRobotService(Node):
    def __init__(self):
        super().__init__('move_robot_service')
        self.srv = self.create_service(MoveRobot, 'move_robot', self.move_robot_callback)

    def move_robot_callback(self, request, response):
        self.get_logger().info(f'Received request to move to: ({request.x}, {request.y}, {request.theta})')

        # Simulate robot movement
        success = self.execute_movement(request.x, request.y, request.theta)

        response.success = success
        response.message = "Movement completed successfully" if success else "Movement failed"

        return response

    def execute_movement(self, x, y, theta):
        # Simulate movement execution
        self.get_logger().info(f'Executing movement to ({x}, {y}, {theta})')
        # In a real robot, this would control actual motors
        return True  # Simulate success

def main(args=None):
    rclpy.init(args=args)
    move_robot_service = MoveRobotService()

    try:
        rclpy.spin(move_robot_service)
    except KeyboardInterrupt:
        pass
    finally:
        move_robot_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# Service client: move_robot_client.py
import rclpy
from rclpy.node import Node
from my_robot_msgs.srv import MoveRobot

class MoveRobotClient(Node):
    def __init__(self):
        super().__init__('move_robot_client')
        self.cli = self.create_client(MoveRobot, 'move_robot')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, x, y, theta):
        request = MoveRobot.Request()
        request.x = x
        request.y = y
        request.theta = theta
        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    move_robot_client = MoveRobotClient()

    response = move_robot_client.send_request(1.0, 2.0, 0.5)
    move_robot_client.get_logger().info(f'Result: {response.success}, {response.message}')

    move_robot_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Code Snippet

Example of an action server and client:

```python
# Action definition: my_robot_msgs/action/Navigation.action
# Goal: float64 x, float64 y
# Result: bool success, string message
# Feedback: float64 distance_to_goal

from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time

# Action server implementation
class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = ActionServer(
            self,
            Navigation,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

    def goal_callback(self, goal_request):
        self.get_logger().info('Received navigation goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing navigation goal...')

        # Simulate navigation process
        feedback_msg = Navigation.Feedback()
        result = Navigation.Result()

        # Simulate progress toward goal
        for i in range(10):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = 'Goal canceled'
                return result

            # Simulate progress
            feedback_msg.distance_to_goal = 10.0 - i
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)  # Simulate work

        goal_handle.succeed()
        result.success = True
        result.message = 'Navigation completed successfully'
        return result
```

```python
# Action client implementation
from rclpy.action import ActionClient
from rclpy.duration import Duration

class NavigationActionClient(Node):
    def __init__(self):
        super().__init__('navigation_action_client')
        self._action_client = ActionClient(self, Navigation, 'navigate_to_pose')

    def send_goal(self, x, y):
        goal_msg = Navigation.Goal()
        goal_msg.x = x
        goal_msg.y = y

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Distance to goal: {feedback_msg.feedback.distance_to_goal}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, {result.message}')
```

## Exercises

1. **Conceptual Question**: Compare and contrast the use of topics, services, and actions in ROS 2. When would you use each communication pattern in a Physical AI system?

2. **Practical Exercise**: Create a ROS 2 package with custom service and action definitions. Implement a service that calculates the distance between two points and an action that moves a robot to a target position with feedback.

3. **Code Challenge**: Design a robot system where multiple nodes coordinate using different communication patterns. Implement a system with a navigation service, a mapping action, and sensor data topics.

4. **Critical Thinking**: How do services and actions impact the real-time performance of Physical AI systems? Discuss the trade-offs between synchronous communication and system responsiveness.

## Summary

This chapter covered ROS 2's synchronous communication patterns: services for request/response interactions and actions for goal-oriented operations. We learned when to use each pattern and implemented practical examples of both. Services and actions provide essential communication mechanisms for Physical AI systems that need to coordinate complex, multi-step tasks and ensure completion of critical operations with feedback and cancellation capabilities.