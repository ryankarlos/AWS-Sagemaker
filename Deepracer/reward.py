import math


def reward_function(params):
    MAX_REWARD = 1e2
    MIN_REWARD = 1e-3
    INITIAL_REWARD = 1
    DIRECTION_THRESHOLD = 10.0
    ABS_STEERING_THRESHOLD = 15
    TOTAL_NUM_STEPS = 2700

    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle'])  # Only need the absolute steering angle for calculations
    progress = params['progress']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    steps = params['steps']

    reward = INITIAL_REWARD

    def distance_from_center_reward(current_reward, track_width, distance_from_center):
        # Calculate 3 marks that are farther and father away from the center line
        marker_1 = 0.1 * track_width
        marker_2 = 0.25 * track_width
        marker_3 = 0.5 * track_width

        # Give higher reward if the car is closer to center line and vice versa
        if distance_from_center <= marker_1:
            current_reward *= 1.4
        elif distance_from_center <= marker_2:
            current_reward *= 0.7
        elif distance_from_center <= marker_3:
            current_reward += 0.3
        else:
            current_reward = MIN_REWARD  # likely crashed/ close to off track

        return current_reward

    def steering_reward(current_reward, steering):
        # Penalize reward if the car is steering too much (your action space will matter)
        if abs(steering) > ABS_STEERING_THRESHOLD:
            current_reward += 0.8
        return current_reward

    def track_completion_reward(current_reward, progress):
        if progress > 50:
            current_reward += 30
        elif progress > 60:
            current_reward += 60
        elif progress > 70:
            current_reward += 100
        elif progress == 100:
            current_reward += 1000
        return current_reward

    def direction_reward(current_reward, waypoints, closest_waypoints, heading):
        next_point = waypoints[closest_waypoints[1]]
        prev_point = waypoints[closest_waypoints[0]]

        # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
        direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
        # Convert to degrees
        direction = math.degrees(direction)

        # Cacluate difference between track direction and car heading angle
        direction_diff = abs(direction - heading)
        if direction_diff > 180:
            direction_diff = 360 - direction_diff

        # Penalize if the difference is too large
        if direction_diff > DIRECTION_THRESHOLD:
            current_reward *= 0.75

        return current_reward

    def fewer_steps_reward(current_reward, steps, progress):

        # Give additional reward if the car pass every 100 steps faster than expected
        if (steps % 100) == 0 and progress > (steps / TOTAL_NUM_STEPS) * 100:
            current_reward += 10.0

        return float(current_reward)

    reward = distance_from_center_reward(reward, track_width, distance_from_center)
    reward = steering_reward(reward, steering)
    reward = direction_reward(reward, waypoints, closest_waypoints, heading)
    reward = track_completion_reward(reward, progress)
    reward = fewer_steps_reward(reward, steps, progress)

    return float(reward)
