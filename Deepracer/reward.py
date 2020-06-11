import math
def reward_function(params):

    # Read input variables
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    progress = params['progress']
    all_wheels_on_track = params['all_wheels_on_track']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    steering = params['steering_angle']
    
    marker_1 = 0.03 * track_width
    marker_2 = 0.12 * track_width
    marker_3 = 0.26 * track_width
    marker_4 = 0.53 * track_width
    marker_5 = 0.75 * track_width
    
    SPEED_THRESHOLD = 2
    DIRECTION_THRESHOLD = 10.0
    ABS_STEERING_THRESHOLD = 15
    
    # initialise the reward
    reward = 1
    
    if distance_from_center >= 0.0 and distance_from_center <= marker_1:
        reward = 1
    elif distance_from_center <= marker_2:
        reward = 0.8
    elif distance_from_center <= marker_3:
        reward = 0.5
    elif distance_from_center <= marker_4:
        reward = 0.1
    elif distance_from_center <= marker_5:
        reward = 0.01
    else:
        reward = 1e-3  # likely crashed/ close to off track
   
    if not (all_wheels_on_track):
        reward = 1e-3
    elif speed < SPEED_THRESHOLD:
        reward *= 0.8
    else:
        reward = 1
        
    if progress == 100:    
        reward += 100
    if progress == 85:    
        reward += 75
    if progress == 70:    
        reward += 50
        
    # Penalize if car steer too much to prevent zigzag
    if steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
        
    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0]) 
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of thecar
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # Penalize the reward if the difference is too large

    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.6
        
    return float(reward)