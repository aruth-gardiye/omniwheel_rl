# Description: Implementation of environment class for the moving target

import pygame
import math
import random
import numpy as np


# Define the Environment class
class Environment:
    def __init__(self):
        # Define the constants for the environment
        self.WIDTH, self.HEIGHT = 800, 600
        self.FIELD = pygame.Rect(50, 50, self.WIDTH - 100, self.HEIGHT - 100)
        self.ROBOT_RADIUS = 20
        self.WHEEL_RADIUS = 5
        self.TARGET_RADIUS = 10
        self.FONT = pygame.font.SysFont("Arial", 24)

        self.robot_pose = [
            random.randint(self.FIELD.left, self.FIELD.right),
            random.randint(self.FIELD.top, self.FIELD.bottom),
            random.randint(0, 359),
        ]

        self.target_pos = [
            random.randint(self.FIELD.left, self.FIELD.right),
            random.randint(self.FIELD.top, self.FIELD.bottom),
        ]

        self.target_vel = [
            random.uniform(0.2, 0.6),
            random.uniform(-0.6, 0.6),
        ]

        self.time_step = 0

        # calculate the distance between the robot and the target
        distance_to_target = math.sqrt(
            (self.robot_pose[0] - self.target_pos[0]) ** 2
            + (self.robot_pose[1] - self.target_pos[1]) ** 2
        )

        # calculate the angle between the robot and the target
        angle_to_target = math.degrees(
            math.atan2(
                self.target_pos[1] - self.robot_pose[1],
                self.target_pos[0] - self.robot_pose[0],
            )
        )

        # if the angle is negative, add 360 to it
        if angle_to_target < 0:
            angle_to_target += 360

        self.ROBOT_TO_TARGET_DISTANCE = distance_to_target
        self.ROBOT_TO_TARGET_ANGLE = angle_to_target
        self.action = None
        self.episode = 0
        self.score = 0
        self.screen = None

    def move_target(self):
        # Move the target based on its velocity
        x, y = self.target_pos
        vx, vy = self.target_vel
        x_prime = x + vx
        y_prime = y + vy

        # if the target is out of bounds, bounce off the sides
        if not self.FIELD.collidepoint(x_prime, y_prime):
            if x_prime < self.FIELD.left or x_prime > self.FIELD.right:
                vx *= -1
            if y_prime < self.FIELD.top or y_prime > self.FIELD.bottom:
                vy *= -1

        return [x_prime, y_prime], [vx, vy]

    def reset(self, first_episode=False):
        # Reset the environment for a new episode
        self.robot_pose = [
            random.randint(self.FIELD.left, self.FIELD.right),
            random.randint(self.FIELD.top, self.FIELD.bottom),
            random.randint(0, 359),
        ]

        self.target_pos = [
            random.randint(self.FIELD.left, self.FIELD.right),
            random.randint(self.FIELD.top, self.FIELD.bottom),
        ]

        self.target_vel = [
            random.uniform(0.2, 0.6),
            random.uniform(-0.6, 0.6),
        ]

        # calculate the distance between the robot and the target
        distance_to_target = math.sqrt(
            (self.robot_pose[0] - self.target_pos[0]) ** 2
            + (self.robot_pose[1] - self.target_pos[1]) ** 2
        )

        # calculate the angle between the robot and the target
        angle_to_target = math.degrees(
            math.atan2(
                self.target_pos[1] - self.robot_pose[1],
                self.target_pos[0] - self.robot_pose[0],
            )
        )

        # if the angle is negative, add 360 to it
        if angle_to_target < 0:
            angle_to_target += 360

        self.ROBOT_TO_TARGET_DISTANCE = distance_to_target
        self.ROBOT_TO_TARGET_ANGLE = angle_to_target
        self.action = None
        self.score = 0
        if first_episode:
            self.episode = 0
        else:
            self.episode += 1
        self.time_step = 0
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        state, reward, done = self.get_state()

        return state

    def update_pose(self, omega_0, omega_1, omega_2):
        # Get the current pose of the robot
        x, y, theta = self.robot_pose

        # Update the pose of the robot based on the given wheel speeds
        step_size = self.time_step
        R = 0.5

        # calculate the velocity of the robot in the x and y directions
        V_x = R * (
            omega_0 * math.cos(math.radians(60))
            + omega_1 * math.cos(math.radians(180))
            + omega_2 * math.cos(math.radians(300))
        )
        V_y = R * (
            omega_0 * math.sin(math.radians(60))
            + omega_1 * math.sin(math.radians(180))
            + omega_2 * math.sin(math.radians(300))
        )

        # rotate the velocity vector by the angle of the robot
        V_x_rotated = V_x * math.cos(math.radians(theta)) - V_y * math.sin(
            math.radians(theta)
        )
        V_y_rotated = V_x * math.sin(math.radians(theta)) + V_y * math.cos(
            math.radians(theta)
        )

        # calculate the new pose of the robot
        omega = omega_0 + omega_1 + omega_2
        x_prime = x + V_x_rotated * step_size
        y_prime = y + V_y_rotated * step_size

        # theta_prime is the new angle of the robot
        theta_prime = theta + omega * step_size
        theta_prime = theta_prime % 360

        return [x_prime, y_prime, theta_prime]

    def step(self, omega_0, omega_1, omega_2, render=False):
        # Update the current action of the environment
        self.action = [omega_0, omega_1, omega_2]

        # Move the target
        self.target_pos, self.target_vel = self.move_target()

        # Get the new pose of the robot
        x_prime, y_prime, theta_prime = self.update_pose(
            omega_0,
            omega_1,
            omega_2,
        )

        # Increment the time step
        self.time_step += 1

        # Update current robot pose
        self.robot_pose = [x_prime, y_prime, theta_prime]

        # Get the state of the environment after the action
        state, reward, done = self.get_state()

        # Update self.ROBOT_TO_TARGET_DISTANCE
        self.ROBOT_TO_TARGET_DISTANCE = math.sqrt(
            (x_prime - self.target_pos[0]) ** 2 + (y_prime - self.target_pos[1]) ** 2
        )

        # Add the reward to the score
        self.score += reward

        # if state is 0, the robot has reached the target
        if state == 0:
            (
                self.robot_pose,
                self.target_pos,
                self.episode,
                self.time_step,
            ) = self.new_episode()

        # if state is 1, the robot has gone out of bounds
        if state == 1:
            (
                self.robot_pose,
                self.target_pos,
                self.episode,
                self.time_step,
            ) = self.new_episode()

        # Draw the game if the 'render' flag is set to True
        if render:
            self.screen.fill((0, 0, 0))
            # Draw field
            pygame.draw.rect(self.screen, (25, 25, 25), self.FIELD)
            x = self.robot_pose[0]
            y = self.robot_pose[1]
            theta = self.robot_pose[2]
            target_pos = self.target_pos

            # Draw robot
            pygame.draw.circle(
                self.screen,
                (200, 200, 200),
                (int(x), self.HEIGHT - int(y)),
                self.ROBOT_RADIUS,
            )

            # Draw target
            pygame.draw.circle(
                self.screen,
                (255, 165, 0),
                (int(target_pos[0]), self.HEIGHT - int(target_pos[1])),
                self.TARGET_RADIUS,
            )

            # Draw wheels
            for i, colour in zip(
                [60, 180, 300], [(255, 0, 0), (255, 0, 255), (0, 0, 255)]
            ):
                wheel_x = int(
                    x + self.ROBOT_RADIUS * math.cos(math.radians(i + theta - 90))
                )
                wheel_y = self.HEIGHT - int(
                    y - self.ROBOT_RADIUS * math.sin(math.radians(i + theta - 90))
                )
                pygame.draw.circle(
                    self.screen, colour, (wheel_x, wheel_y), self.WHEEL_RADIUS
                )

            score_surface = self.FONT.render(
                f"Epsiode: {self.episode}  Step: {self.time_step}  Score: {self.score:.2f}",
                True,
                (255, 255, 255),
            )

            score_pos = (self.WIDTH - score_surface.get_rect().width - 10, 10)

            action_surface = self.FONT.render(
                f"Action: {self.action}",
                True,
                (255, 255, 255),
            )

            action_pos = (10, self.HEIGHT - action_surface.get_rect().height - 10)

            self.screen.blit(score_surface, score_pos)
            self.screen.blit(action_surface, action_pos)

            pygame.display.flip()
            pygame.time.delay(40)

        return state, reward, done

    def new_episode(self):
        # Start a new episode of the game
        robot_pose = [
            random.randint(self.FIELD.left, self.FIELD.right),
            random.randint(self.FIELD.top, self.FIELD.bottom),
            random.randint(0, 359),
        ]
        target_pos = [
            random.randint(self.FIELD.left, self.FIELD.right),
            random.randint(self.FIELD.top, self.FIELD.bottom),
        ]

        # calculate the distance between the robot and the target
        distance_to_target = math.sqrt(
            (robot_pose[0] - target_pos[0]) ** 2 + (robot_pose[1] - target_pos[1]) ** 2
        )

        # calculate the angle between the robot and the target
        angle_to_target = math.degrees(
            math.atan2(target_pos[1] - robot_pose[1], target_pos[0] - robot_pose[0])
        )

        # if the angle is negative, add 360 to it
        if angle_to_target < 0:
            angle_to_target += 360

        self.ROBOT_TO_TARGET_DISTANCE = distance_to_target
        self.ROBOT_TO_TARGET_ANGLE = angle_to_target
        self.action = None
        self.score = 0

        return robot_pose, target_pos, self.episode, self.time_step

    def get_state(self):
        # Return the current state of the environment
        robot_x = self.robot_pose[0]
        robot_y = self.robot_pose[1]
        target_x = self.target_pos[0]
        target_y = self.target_pos[1]

        # Check for target, timeout, or out-of-bounds
        distance_to_target = math.sqrt(
            (robot_x - target_x) ** 2 + (robot_y - target_y) ** 2
        )

        # calculate the angle between the robot and the target
        angle_to_target = math.degrees(
            math.atan2(target_y - robot_y, target_x - robot_x)
        )

        # if the angle is negative, add 360 to it
        if angle_to_target < 0:
            angle_to_target += 360

        # if ball is close to the robot and distance is less than robot radius
        if distance_to_target <= self.ROBOT_RADIUS:
            print("Robot reached target")
            return (0, 10, True)

        # calculate if robot is within the field
        elif not self.FIELD.collidepoint(robot_x, robot_y):
            return (1, -10, True)

        # see if robot is facing the target
        elif abs(angle_to_target - self.robot_pose[2]) < 10:
            return (2, -0.01, False)

        # else if robot has furthered away from the target
        elif distance_to_target > self.ROBOT_TO_TARGET_DISTANCE:
            return (3, -0.04, False)

        # else if ball moved closer to the robot
        else:
            return (4, -0.01, False)

    # returns numpy array of robot details and target details
    def get_observation(self):
        # Return the current state of the environment
        robot_x = self.robot_pose[0]
        robot_y = self.robot_pose[1]
        robot_theta = self.robot_pose[2]
        target_x = self.target_pos[0]
        target_y = self.target_pos[1]
        target_vel_x = self.target_vel[0]
        target_vel_y = self.target_vel[1]

        return np.array(
            [
                robot_x,
                robot_y,
                robot_theta,
                target_x,
                target_y,
                target_vel_x,
                target_vel_y,
            ]
        )
