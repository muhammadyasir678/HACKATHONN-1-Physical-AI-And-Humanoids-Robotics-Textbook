---
title: 'Chapter 5: Simulation to Real-World Transfer'
sidebar_position: 5
description: 'Techniques and challenges for transferring AI models from simulation to real robots'
---

# Chapter 5: Simulation to Real-World Transfer

## Learning Objectives

- Understand the concept of sim-to-real transfer in robotics
- Learn about domain randomization and domain adaptation techniques
- Explore methods to bridge the reality gap between simulation and real robots
- Gain knowledge of evaluation and validation approaches for transfer learning

## Introduction

Simulation-to-real-world transfer (sim-to-real) is a critical challenge in Physical AI and robotics. While simulation provides a safe, cost-effective, and controllable environment for training AI models, the ultimate goal is to deploy these models on real robots. The "reality gap" between simulated and real environments often prevents directly transferring models from simulation to reality. Understanding and addressing this gap is essential for practical Physical AI applications.

## Core Theory

The reality gap encompasses several factors:

- **Visual Differences**: Lighting, textures, colors, and visual artifacts
- **Physical Differences**: Dynamics, friction, material properties, and sensor noise
- **Modeling Imperfections**: Simplified physics, inaccurate robot models, and environmental assumptions
- **Temporal Differences**: Timing, latency, and update rates

Techniques to address the reality gap include:

- **Domain Randomization**: Training with randomized simulation parameters
- **Domain Adaptation**: Adapting models to new domains using limited real data
- **System Identification**: Calibrating simulation parameters to match reality
- **Progressive Domain Transfer**: Gradual transition from simulation to reality
- **Sim-to-Real Systematic Approach**: Methodical validation and adjustment

Domain randomization involves varying parameters during training such as:
- Lighting conditions and camera properties
- Object textures, colors, and appearances
- Physical properties (friction, mass, damping)
- Sensor noise characteristics
- Environmental conditions

## Practical Example

Let's examine domain randomization implementation in simulation:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import random
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DomainRandomizationNode(Node):
    def __init__(self):
        super().__init__('domain_randomization_node')
        self.bridge = CvBridge()

        # Publishers for randomized parameters
        self.friction_pub = self.create_publisher(Float32, '/randomized_friction', 10)
        self.lighting_pub = self.create_publisher(Float32, '/randomized_lighting', 10)

        # Timer for parameter updates
        self.randomization_timer = self.create_timer(5.0, self.randomize_parameters)

        # Store current parameters
        self.current_friction = 0.5
        self.current_lighting = 1.0

        self.get_logger().info('Domain randomization node initialized')

    def randomize_parameters(self):
        """Randomize simulation parameters"""
        # Randomize friction coefficient (0.1 to 1.0)
        self.current_friction = random.uniform(0.1, 1.0)
        friction_msg = Float32()
        friction_msg.data = self.current_friction
        self.friction_pub.publish(friction_msg)

        # Randomize lighting intensity (0.5 to 2.0)
        self.current_lighting = random.uniform(0.5, 2.0)
        lighting_msg = Float32()
        lighting_msg.data = self.current_lighting
        self.lighting_pub.publish(lighting_msg)

        self.get_logger().info(
            f'Randomized parameters - Friction: {self.current_friction:.2f}, '
            f'Lighting: {self.current_lighting:.2f}'
        )

    def randomize_visual_features(self, image):
        """Apply visual domain randomization to an image"""
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # Apply random color jittering
        brightness_factor = random.uniform(0.7, 1.3)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.9, 1.1)

        # Convert to HSV for easier manipulation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor  # V channel
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)

        # Convert back to BGR
        randomized_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add random noise
        noise = np.random.normal(0, random.uniform(1, 5), randomized_image.shape).astype(np.uint8)
        randomized_image = cv2.add(randomized_image, noise)

        return randomized_image

def main(args=None):
    rclpy.init(args=args)
    domain_randomization_node = DomainRandomizationNode()

    try:
        rclpy.spin(domain_randomization_node)
    except KeyboardInterrupt:
        pass
    finally:
        domain_randomization_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Code Snippet

Example of system identification for sim-to-real transfer:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.optimize import minimize
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import pickle

class SystemIdentificationNode(Node):
    def __init__(self):
        super().__init__('system_identification_node')

        # Subscribers for real robot data
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        # Publisher for identified parameters
        self.param_pub = self.create_publisher(Float32MultiArray, '/identified_params', 10)

        # Data storage for system identification
        self.joint_positions = []
        self.joint_velocities = []
        self.commands = []
        self.timestamps = []

        # Timer for parameter identification
        self.identification_timer = self.create_timer(10.0, self.perform_identification)

        # Initial parameter estimates
        self.current_params = np.array([1.0, 0.1, 0.05])  # mass, damping, friction

        self.get_logger().info('System identification node initialized')

    def joint_state_callback(self, msg):
        """Store joint state data for system identification"""
        self.joint_positions.append(list(msg.position))
        self.joint_velocities.append(list(msg.velocity))
        self.timestamps.append(self.get_clock().now().nanoseconds / 1e9)

    def cmd_callback(self, msg):
        """Store command data for system identification"""
        cmd_data = [msg.linear.x, msg.angular.z]
        self.commands.append(cmd_data)

    def robot_dynamics_model(self, params, state, command):
        """
        Simplified robot dynamics model
        params: [mass, damping, friction]
        state: [position, velocity]
        command: [linear_vel_cmd, angular_vel_cmd]
        """
        mass, damping, friction = params

        # Simplified dynamics: acceleration = (command - damping*vel - friction*sign(vel)) / mass
        acceleration = (command[0] - damping * state[1] - friction * np.sign(state[1])) / mass

        return acceleration

    def simulation_error(self, params):
        """Calculate error between simulation and real data"""
        if len(self.joint_positions) < 10:  # Need sufficient data
            return float('inf')

        total_error = 0.0
        for i in range(1, min(len(self.joint_positions), len(self.commands))):
            # Get state and command
            pos_prev = self.joint_positions[i-1][0] if len(self.joint_positions[i-1]) > 0 else 0
            vel_prev = self.joint_velocities[i-1][0] if len(self.joint_velocities[i-1]) > 0 else 0
            state_prev = [pos_prev, vel_prev]

            cmd = self.commands[i]

            # Simulate with current parameters
            dt = self.timestamps[i] - self.timestamps[i-1]
            acceleration = self.robot_dynamics_model(params, state_prev, cmd)
            vel_sim = vel_prev + acceleration * dt
            pos_sim = pos_prev + vel_sim * dt

            # Calculate error
            pos_real = self.joint_positions[i][0] if len(self.joint_positions[i]) > 0 else 0
            vel_real = self.joint_velocities[i][0] if len(self.joint_velocities[i]) > 0 else 0

            pos_error = abs(pos_sim - pos_real)
            vel_error = abs(vel_sim - vel_real)

            total_error += pos_error + vel_error

        return total_error

    def perform_identification(self):
        """Perform system identification to match simulation to reality"""
        if len(self.joint_positions) < 10:
            self.get_logger().warn('Insufficient data for system identification')
            return

        # Optimize parameters to minimize simulation error
        result = minimize(
            self.simulation_error,
            self.current_params,
            method='BFGS',
            options={'disp': True}
        )

        if result.success:
            self.current_params = result.x
            self.get_logger().info(f'System identification completed. New parameters: {self.current_params}')

            # Publish identified parameters
            param_msg = Float32MultiArray()
            param_msg.data = self.current_params.tolist()
            self.param_pub.publish(param_msg)

            # Save parameters to file
            self.save_parameters()
        else:
            self.get_logger().error('System identification failed')

    def save_parameters(self):
        """Save identified parameters to file"""
        try:
            with open('/tmp/robot_parameters.pkl', 'wb') as f:
                pickle.dump({
                    'parameters': self.current_params,
                    'timestamp': self.get_clock().now().nanoseconds / 1e9
                }, f)
            self.get_logger().info('Parameters saved to /tmp/robot_parameters.pkl')
        except Exception as e:
            self.get_logger().error(f'Failed to save parameters: {e}')

def main(args=None):
    rclpy.init(args=args)
    sys_id_node = SystemIdentificationNode()

    try:
        rclpy.spin(sys_id_node)
    except KeyboardInterrupt:
        pass
    finally:
        sys_id_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Validation and evaluation techniques:

```bash
# Compare simulation vs real robot performance
# Run the same task in simulation and on real robot

# Example evaluation metrics
# 1. Success rate
# 2. Task completion time
# 3. Trajectory accuracy
# 4. Energy efficiency

# Use simulation with identified parameters
ros2 run my_robot_package system_identification_node

# Apply identified parameters to simulation
ros2 param set /gazebo_robot_controller mass 1.2
ros2 param set /gazebo_robot_controller damping 0.15
ros2 param set /gazebo_robot_controller friction 0.08
```

## Exercises

1. **Conceptual Question**: Explain the "reality gap" and its impact on sim-to-real transfer. What are the main factors that contribute to this gap?

2. **Practical Exercise**: Implement a domain randomization technique for a simple robot task in simulation. Train a model with randomized parameters and evaluate its performance.

3. **Code Challenge**: Create a system identification node that calibrates simulation parameters based on real robot data to minimize the reality gap.

4. **Critical Thinking**: What are the limitations of current sim-to-real transfer techniques? How might emerging technologies like digital twins and advanced simulation engines address these limitations?

## Summary

This chapter covered the critical challenge of simulation-to-real-world transfer in Physical AI. We explored the reality gap, domain randomization techniques, system identification methods, and evaluation approaches. Successful sim-to-real transfer requires careful consideration of visual, physical, and temporal differences between simulation and reality. Techniques like domain randomization, system identification, and progressive domain transfer help bridge this gap, enabling the deployment of AI models trained in simulation to real robots.