---
title: 'Chapter 4: Parameter Management and Launch Systems'
sidebar_position: 4
description: 'Managing configuration and launching ROS 2 systems with parameters and launch files'
---

# Chapter 4: Parameter Management and Launch Systems

## Learning Objectives

- Understand ROS 2 parameter management for configuration
- Learn to create and use launch files for system startup
- Explore parameter declaration, access, and dynamic reconfiguration
- Gain experience with launch arguments and conditional execution

## Introduction

Effective parameter management and system launching are critical for deploying and operating complex Physical AI systems. ROS 2 provides robust mechanisms for managing configuration parameters and orchestrating the startup of multiple nodes. Parameters allow for runtime configuration without recompilation, while launch files enable the coordinated startup of complex multi-node systems. These tools are essential for creating maintainable, configurable, and deployable Physical AI applications.

## Core Theory

ROS 2 parameter and launch systems include:

- **Parameters**: Configuration values that can be set at runtime
- **Parameter Descriptors**: Constraints and metadata for parameters
- **Launch Files**: XML/YAML/Python files that specify how to launch multiple nodes
- **Launch Arguments**: Dynamic values passed to launch files
- **Composable Nodes**: Nodes that can be loaded into a single process container

Parameters in ROS 2 are strongly typed and can be:
- Declared with constraints (min, max, range, etc.)
- Set at node startup or changed dynamically
- Accessed programmatically from within nodes
- Loaded from configuration files

Launch files can specify:
- Which nodes to run
- Their parameters
- Their remappings
- Their required dependencies
- Conditional execution based on arguments

## Practical Example

Let's examine parameter declaration and usage in a ROS 2 node:

```python
# Parameter management example: robot_controller.py
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import ParameterType

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Declare parameters with descriptors
        self.declare_parameter(
            'max_velocity',
            1.0,
            ParameterDescriptor(
                name='max_velocity',
                type=ParameterType.PARAMETER_DOUBLE,
                description='Maximum velocity for robot movement',
                additional_constraints='Must be positive',
                floating_point_range=[Parameter.FloatingPointRange(from_value=0.0, to_value=10.0, step=0.1)]
            )
        )

        self.declare_parameter(
            'robot_name',
            'default_robot',
            ParameterDescriptor(
                name='robot_name',
                type=ParameterType.PARAMETER_STRING,
                description='Name of the robot',
                additional_constraints='Alphanumeric characters only'
            )
        )

        # Get parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(f'Robot {self.robot_name} initialized with max velocity: {self.max_velocity}')

        # Set up parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                if 0.0 <= param.value <= 10.0:
                    self.max_velocity = param.value
                    self.get_logger().info(f'Max velocity updated to: {param.value}')
                    return SetParametersResult(successful=True)
                else:
                    self.get_logger().error('Max velocity must be between 0.0 and 10.0')
                    return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Code Snippet

Example launch file in Python:

```python
# Launch file: robot_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import TextSubstitution

def generate_launch_description():
    # Declare launch arguments
    robot_name_launch_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    max_velocity_launch_arg = DeclareLaunchArgument(
        'max_velocity',
        default_value='1.0',
        description='Maximum velocity for the robot'
    )

    # Get launch configuration values
    robot_name = LaunchConfiguration('robot_name')
    max_velocity = LaunchConfiguration('max_velocity')

    # Define robot controller node
    robot_controller_node = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {
                'robot_name': robot_name,
                'max_velocity': max_velocity
            }
        ],
        remappings=[
            ('/cmd_vel', '/robot1/cmd_vel'),
            ('/odom', '/robot1/odom')
        ]
    )

    # Define sensor node
    sensor_node = Node(
        package='my_robot_package',
        executable='sensor_driver',
        name='sensor_driver',
        parameters=[
            {
                'sensor_range': 10.0,
                'update_rate': 10.0
            }
        ]
    )

    return LaunchDescription([
        robot_name_launch_arg,
        max_velocity_launch_arg,
        robot_controller_node,
        sensor_node
    ])
```

YAML parameter file:

```yaml
# Parameter file: config/robot_params.yaml
/**:  # Applies to all nodes
  ros__parameters:
    robot_name: "configured_robot"
    max_velocity: 2.0
    safety_margin: 0.5

robot_controller:  # Applies to robot_controller node only
  ros__parameters:
    control_frequency: 50.0
    acceleration_limit: 1.5

sensor_driver:
  ros__parameters:
    sensor_range: 5.0
    update_rate: 20.0
```

Command-line examples for parameter management:

```bash
# List parameters of a node
ros2 param list /robot_controller

# Get a specific parameter
ros2 param get /robot_controller max_velocity

# Set a parameter
ros2 param set /robot_controller max_velocity 2.5

# Load parameters from a file
ros2 param load /robot_controller config/robot_params.yaml

# Save current parameters to a file
ros2 param dump /robot_controller --output current_params.yaml

# Launch with arguments
ros2 launch my_robot_package robot_system.launch.py robot_name:=test_robot max_velocity:=3.0
```

## Exercises

1. **Conceptual Question**: Explain the advantages of using launch files over manually starting individual nodes. How do launch files improve the maintainability of Physical AI systems?

2. **Practical Exercise**: Create a ROS 2 package with a node that declares multiple parameters with different types and constraints. Create a launch file that starts the node with parameter values loaded from a YAML file.

3. **Code Challenge**: Design a launch system that can conditionally start different sets of nodes based on launch arguments. Create a launch file that can start either a simulation or real robot configuration.

4. **Critical Thinking**: How do ROS 2 parameters compare to environment variables or configuration files in terms of flexibility and runtime reconfiguration? What are the performance implications of using parameters in real-time systems?

## Summary

This chapter explored ROS 2's parameter management and launch systems, which are essential for configuring and deploying complex Physical AI systems. We learned how to declare and manage parameters with constraints, create launch files for coordinated system startup, and use launch arguments for dynamic configuration. These tools enable the creation of maintainable, configurable, and deployable Physical AI applications that can adapt to different environments and requirements.