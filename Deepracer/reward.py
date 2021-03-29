def reward_function(params):

    MAX_REWARD = 1e2
    MIN_REWARD = 1e-3
    DIRECTION_THRESHOLD = 10.0
    ABS_STEERING_THRESHOLD = 29

    on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    steering = abs(params['steering_angle']) # Only need the absolute steering angle for calculations
    speed = params['speed']
    progress = params['progress']
    
    # initialise reward
    reward = 1e-2
    

    def on_track_reward(current_reward, on_track):
        if not on_track:
            current_reward = MIN_REWARD
        else:
            current_reward = MAX_REWARD
        return current_reward

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

    def straight_line_reward(current_reward, steering, speed):
        # Positive reward if the car is in a straight line going fast
        if abs(steering) < 0.1 and speed > 2.5:
            current_reward *= 1.8
        return current_reward


    def steering_reward(current_reward, steering):
        # Penalize reward if the car is steering too much (your action space will matter)
        if abs(steering) > ABS_STEERING_THRESHOLD:
            current_reward += 0.9
        return current_reward
        
    def track_completion_reward(current_reward,progress ):
        if progress > 50:
            current_reward *= 1.2
        elif progress > 75:
            current_reward *= 2
        elif progress == 100:
            current_reward *= 5
        return current_reward


    reward = on_track_reward(reward, on_track)
    reward = distance_from_center_reward(reward, track_width, distance_from_center)
    reward = straight_line_reward(reward, steering, speed)
    reward = steering_reward(reward, steering)
    reward = track_completion_reward(reward, progress)
    

    return float(reward)
