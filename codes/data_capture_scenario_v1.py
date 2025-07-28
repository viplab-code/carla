#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to the capture sensor data script, a script that provides users with a baseline for data collection,
which they can later modify to their specific needs, easying the process of creating a database.

This script will start with a CARLA recorder log, spawning the desired sensor configuration at the ego vehicle,
and saving their data into a folder.

At the end of the recorder, all replayed actors will be destroyed,and take note
that the recorder teleports the vehicle, so no dynamic information can be gathered from them.
If the record has a `log.json`, the IMU can be used to extract information about the ego vehicle.

Modify the parameters at the very top of the script to match the desired use-case:

- SENSORS: List of all the sensors tha will be spawned in the simulation
- WEATHER: Weather of the simulation
- RECORDER_INFO: List of all the CARLA recorder logs that will be run. Each recorder has four elements:
    쨌 folder: path to the folder with the recorder files
    쨌 name: name of the endpoint folder
    쨌 start_time: start time of the recorder
    쨌 duration: duration of the recorder. 0 to replay it until the end
- DESTINATION_FOLDER: folder where all sensor data will be stored

"""

import time
import os
import carla
import argparse
import random
import json
import threading
import glob
import cv2
import numpy as np
import math
from queue import Queue, Empty

################### User simulation configuration ####################
TL_ATTRIBUTE = {
    carla.TrafficLightState.Off : 0,
    carla.TrafficLightState.Unknown: 0,
    carla.TrafficLightState.Red: 1,
    carla.TrafficLightState.Green: 2,
    carla.TrafficLightState.Yellow: 3,
    'go_straight': 4,
    'turn_left': 5,
    'turn_right': 6,
    'no_left_turn': 7,
    'no_right_turn': 8,
    'u_turn': 9,
    'no_u_turn': 10,
    'slight_left': 11,
    'slight_right': 12
}

SENSORS = [
    [
        'ring_front_center',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 1550, 'image_size_y': 2048, 'fov': 50,
            'x': 0.53, 'y': 0.00, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        }
    ],
    [
        'ring_front_left',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
            'x': 0.45, 'y': -0.2, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0
        }
    ],
    [
        'ring_front_right',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
            'x': 0.45, 'y': 0.2, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0
        }
    ],
    [
        'ring_rear_left',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
            'x': 0.0, 'y': -0.12, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': -153.0
        }
    ],
    [
        'ring_rear_right',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
            'x': 0.0, 'y': 0.12, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 153.0
        }
    ],
    [
        'ring_side_left',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
            'x': 0.21, 'y': -0.27, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': -99.2
        }
    ],
    [
        'ring_side_right',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': 2048, 'image_size_y': 1550, 'fov': 60,
            'x': 0.21, 'y': 0.27, 'z': 1.70, 'roll': 0.0, 'pitch': 0.0, 'yaw': 99.2
        }
    ]
]
# 3) Choose the recorder files

scinarios = os.listdir('ScenarioLogs')

#47개 시나리오

RECORDER_INFO = [
    {
        'folder': f"ScenarioLogs/{scinario}",
        'name': f'{scinario}',
        'start_time': 0,
        'duration': 0
    } for scinario in scinarios
]

# 4) Choose the destination folder
DESTINATION_FOLDER = "database"
################# End user simulation configuration ##################

FPS = 2
THREADS = 5
CURRENT_THREADS = 0
AGENT_TICK_DELAY = 10


def create_folders(endpoint, sensors):
    for sensor_id, sensor_bp in sensors:
        sensor_endpoint = f"{endpoint}/{sensor_id}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)


def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))


def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return


def get_ego_id(recorder_file):
    found_lincoln = False
    found_id = None

    for line in recorder_file.split("\n"):

        # Check the role_name for hero
        if found_lincoln:
            if not line.startswith("  "):
                found_lincoln = False
                found_id = None
            else:
                data = line.split(" = ")
                if 'role_name' in data[0] and 'hero' in data[1]:
                    return found_id

        # Search for all lincoln vehicles
        if not found_lincoln and line.startswith(" Create ") and 'vehicle.lincoln' in line:
            found_lincoln = True
            found_id =  int(line.split(" ")[2][:-1])

    return found_id

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def is_within_fov(cam_forward, cam_location, target_location, fov_deg):
    ray = target_location - cam_location
    ray_norm = ray.make_unit_vector()
    dot = cam_forward.dot(ray_norm)
    if dot <= 0:
        return False
    angle = math.degrees(math.acos(dot))
    return angle < (fov_deg / 2.0)

def get_image_point(loc, K, w2c, debug=False, actor_id=None):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    if point_camera[2] <= 0.1:
        if debug:
            print(f"[⚠️] Actor {actor_id} is behind camera. Camera z: {point_camera[2]:.2f}")
        return None

    img_point = np.dot(K, point_camera)
    img_point /= img_point[2]
    x, y = img_point[0], img_point[1]

    if debug and (x < 0 or y < 0 or x > K[0, 2] * 2 or y > K[1, 2] * 2):
        print(f"[❌] Actor {actor_id} projected outside image bounds: ({x:.1f}, {y:.1f})")
    return np.array([x, y])

def get_rotation_matrix(yaw, pitch, roll):
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    R_yaw = np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
        [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R_pitch = np.array([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad),  math.cos(roll_rad)]
    ])
    return R_yaw @ R_pitch @ R_roll

def get_bbox_facing(bbox_rot):
    R = get_rotation_matrix(bbox_rot.yaw, bbox_rot.pitch, bbox_rot.roll)
    y_axis = np.array([0, 1, 0])
    return R @ y_axis

def bbox_sanity_check(x_min, x_max, y_min, y_max, image_w, image_h, min_wh=10, max_area_ratio=0.5):
    w = x_max - x_min
    h = y_max - y_min
    area = w * h
    img_area = image_w * image_h
    if w < min_wh or h < min_wh:
        return False
    if area > max_area_ratio * img_area:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True

def get_projected_bbox(verts, K, world_2_camera, image_w, image_h):
    x_max = -1e6
    x_min = 1e6
    y_max = -1e6
    y_min = 1e6
    valid_projection = True
    for vert in verts:
        p = get_image_point(vert, K, world_2_camera)
        if p is None or np.isnan(p[0]) or np.isnan(p[1]):
            valid_projection = False
            break
        x, y = p
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    if not valid_projection:
        return None, None, None, None, False
    x_min = int(max(0, min(image_w - 1, x_min)))
    x_max = int(max(0, min(image_w - 1, x_max)))
    y_min = int(max(0, min(image_h - 1, y_min)))
    y_max = int(max(0, min(image_h - 1, y_max)))
    return x_min, x_max, y_min, y_max, True

def filter_traffic_lights(vehicle, K, world_2_camera, image_w, image_h):
    traffics = []
    id = 0
    # 모든 traffic light actor 탐색 (bounding_box_set이 아닌 traffic_light actor 기준)
    tls = [a for a in vehicle.get_world().get_actors() if 'traffic_light' in a.type_id]
    for tl in tls:
        id = tl.get_opendrive_id()
        state = tl.state
        category_val = TL_ATTRIBUTE.get(state, 0)  # default 4: Unknown

        tl_loc = tl.get_location()
        ve_loc = vehicle.get_transform().location
        dist = math.hypot(tl_loc.x - ve_loc.x, tl_loc.y - ve_loc.y)
        if not (3 < dist < 50):
            continue

        forward_vec = vehicle.get_transform().get_forward_vector()
        ray = tl.get_transform().location - ve_loc
        if forward_vec.dot(ray) <= 0:
            continue
        
        for idx, bbox in enumerate(tl.get_light_boxes()):
            bbox_center = bbox.location
            if bbox_center.z < 2:
                continue
            bbox_facing = get_bbox_facing(bbox.rotation)
            vehicle_forward = vehicle.get_transform().get_forward_vector()
            v_forward = np.array([vehicle_forward.x, vehicle_forward.y, vehicle_forward.z])
            dot = np.dot(bbox_facing, v_forward)
            # vehicle 기준 불빛이 정면을 향하는 경우만 (마주봄 75도 이내)
            if dot > -math.cos(math.radians(75)):
                continue

            verts = bbox.get_local_vertices()
            x_min, x_max, y_min, y_max, valid_projection = get_projected_bbox(
                verts, K, world_2_camera, image_w, image_h
            )
            if not valid_projection:
                continue
            if not bbox_sanity_check(x_min, x_max, y_min, y_max, image_w, image_h):
                continue
            
            affected = [[wp.transform.location.x,-1*wp.transform.location.y]  for wp in tl.get_affected_lane_waypoints()]
            
            traffics.append({
                'id': f"tl_{str(id)}_{idx}",
                'points': [[x_min, y_min], [x_max, y_max]],
                'category': 0,
                'attribute': category_val,
                'affected_lanelet' : affected 
            })
    return traffics


def is_location_clear(world, transform, threshold=5.0):
    loc = transform.location
    nearby_actors = world.get_actors().filter('vehicle.*')
    for actor in nearby_actors:
        if actor.get_location().distance(loc) < threshold:
            return False
    return True

def save_data_to_disk(sensor_id, frame, data, endpoint):
    """
    Saves the sensor data into file:
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'
    """
    global CURRENT_THREADS
    CURRENT_THREADS += 1
    # openlane 학습을 위해 코드 수정
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.jpg"
        os.makedirs(os.path.dirname(sensor_endpoint), exist_ok=True)
        img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
        cv2.imwrite(sensor_endpoint, img)

    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

    CURRENT_THREADS -= 1


def carla_to_openlane_pose(x, y, z, roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R_carla = Rz @ Ry @ Rx
    flip_Y = np.diag([1, -1, 1])
    R_openlane = flip_Y @ R_carla
    U, _, Vt = np.linalg.svd(R_openlane)
    R_openlane_fixed = U @ Vt
    if np.linalg.det(R_openlane_fixed) < 0:
        Vt[-1, :] *= -1
        R_openlane_fixed = U @ Vt
    t_openlane = flip_Y @ np.array([x, y, z])
    return R_openlane_fixed.tolist(), t_openlane.tolist()

def add_agent_delay(recorder_log):
    """
    The agent logs are delayed from the simulation recorder, which depends on the leaderboard setup.
    As the vehicle is stopped at the beginning, fake them with all 0 values, and the initial transform
    """

    init_tran = recorder_log['records'][0]['state']['transform']
    for _ in range(AGENT_TICK_DELAY):

        elem = {}
        elem['control'] = {
            'brake': 0.0, 'gear': 0, 'hand_brake': False, 'manual_gear_shift': False,
            'reverse': False, 'steer': 0.0, 'throttle': 0.0
        }
        elem['state'] = {
            'acceleration': {'value': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': { 'value': 0.0, 'x': -0.0, 'y': 0.0, 'z': 0.0},
            'velocity': {'value': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'transform': {
                'pitch': init_tran['pitch'], 'yaw': init_tran['yaw'], 'roll': init_tran['roll'],
                'x': init_tran['x'], 'y': init_tran['y'], 'z': init_tran['z']
            }
        }
        recorder_log['records'].insert(0, elem)

    return recorder_log

def save_annotations(vehicle, frame, save_path, name, traffics=None):
    # traffic 등은 없을 수도 있으므로 기본값 None
    pose_path = "openlane_v2_default.json"
    os.makedirs(save_path, exist_ok=True)
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    R, t = carla_to_openlane_pose(round(location.x, 6), round(location.y, 6), round(location.z, 6),
                                  round(rotation.roll, 6), round(rotation.pitch, 6), round(rotation.yaw, 6))

    anno = {}
    if os.path.exists(pose_path):
        with open(pose_path, 'r') as f:
            anno = json.load(f)
    else:
        anno['pose'] = {}
        anno['sensor'] = {}
        anno['annotation'] = {'traffic_element': []}
        anno['meta_data'] = {}

    anno['meta_data']['source'] = "Carla"
    anno['meta_data']['source_id'] = name + '_' + str(frame)

    anno['pose']['rotation'] = R
    anno['pose']['translation'] = t
    anno['pose']['heading'] = rotation.yaw
    anno['timestamp'] = frame
    anno['segment_id'] = os.path.basename(save_path)
    for i in anno.get('sensor', {}).keys():
        anno['sensor'][i]['image_path'] = f"{save_path}/{i}/{frame}.jpg"
    if traffics is not None:
        anno['annotation']['traffic_element'] = traffics
    os.makedirs(os.path.join(save_path, "info"), exist_ok=True)
    with open(os.path.join(save_path, "info", f"{frame}.json"), 'w') as f:
        json.dump(anno, f, indent=2)

def set_endpoint(recorder_info):
    def get_new_endpoint(endpoint):
        i = 2
        new_endpoint = endpoint + "_" + str(i)
        while os.path.isdir(new_endpoint):
            i += 1
            new_endpoint = endpoint + "_" + str(i)
        return new_endpoint

    endpoint = f"{DESTINATION_FOLDER}/{recorder_info['name']}"
    if os.path.isdir(endpoint):
        return None
        old_endpoint = endpoint
        endpoint = get_new_endpoint(old_endpoint)
        print(f"\033[93mWARNING: Given endpoint already exists, changing {old_endpoint} to {endpoint}\033[0m")

    os.makedirs(endpoint)
    return endpoint


def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()
    print(__doc__)

    active_sensors = []

    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(120.0)
        world = client.get_world()
        
        with open("weather_presets.json", "r") as file:
            weather_data = json.load(file)
            
        preset_names = list(weather_data.keys())

        for recorder_info in RECORDER_INFO:

            print(f"\n\033[1m> Getting the recorder information\033[0m")
            recorder_folder = recorder_info['folder']
            recorder_start = recorder_info['start_time']
            recorder_duration = recorder_info['duration']

            recorder_path_list = glob.glob(f"{os.getcwd()}/{recorder_folder}/*.log")
            if recorder_path_list:
                recorder_path = recorder_path_list[0]
                print(f"\033[1m> Running recorder '{recorder_path}'\033[0m")
            else:
                print(f"\033[91mCouldn't find the recorder file for the folder '{recorder_folder}'\033[0m")
                continue

            endpoint = set_endpoint(recorder_info)
            if not endpoint:
                print("already played scinario")
                continue
            
            print(f"\033[1m> Preparing the world. This may take a while...\033[0m")
            recorder_str = client.show_recorder_file_info(recorder_path, False)

            recorder_map = recorder_str.split("\n")[1][5:]
            world = client.load_world(recorder_map)
            world.tick()
            
            selected_weather_name = random.choice(preset_names)
            selected_weather = weather_data[selected_weather_name]
            weather = carla.WeatherParameters(**selected_weather)
            world.set_weather(weather)
            
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / FPS
            settings.synchronous_mode = True
            world.apply_settings(settings)

            world.tick()

            max_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
            if recorder_duration == 0:
                recorder_duration = max_duration
            elif recorder_start + recorder_duration > max_duration:
                print("\033[93mWARNING: Found a duration that exceeds the recorder length. Reducing it...\033[0m")
                recorder_duration = max_duration - recorder_start
            if recorder_start >= max_duration:
                print("\033[93mWARNING: Found a start point that exceeds the recoder duration. Ignoring it...\033[0m")
                continue


            client.replay_file(recorder_path, recorder_start, recorder_duration, get_ego_id(recorder_str), False)
            with open(f"{recorder_path[:-4]}.txt", 'w') as fd:
                fd.write(recorder_str)
            world.tick()


            hero = None
            while hero is None:
                possible_vehicles = world.get_actors().filter('vehicle.*')
                for vehicle in possible_vehicles:
                    if vehicle.attributes['role_name'] == 'hero':
                        hero = vehicle
                        break
                time.sleep(1)

            print(f"\033[1m> Creating the sensors\033[0m")
            create_folders(endpoint, [[s[0], s[1].get('bp')] for s in SENSORS])
            blueprint_library = world.get_blueprint_library()
            sensor_queue = Queue()
            for sensor in SENSORS:

                # Extract the data from the sesor configuration
                sensor_id, attributes = sensor
                blueprint_name = attributes.get('bp')
                sensor_transform = carla.Transform(
                    carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
                    carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
                )

                # Get the blueprint and add the attributes
                blueprint = blueprint_library.find(blueprint_name)
                for key, value in attributes.items():
                    if key in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        continue
                    blueprint.set_attribute(str(key), str(value))

                # Create the sensors and its callback
                sensor = world.spawn_actor(blueprint, sensor_transform, hero)
                add_listener(sensor, sensor_queue, sensor_id)
                active_sensors.append(sensor)

            for _ in range(10):
                world.tick()

            print(f"\033[1m> Running the replayer\033[0m")
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            sensor_amount = len(SENSORS)

            max_threads = THREADS
            results = []

            image_w, image_h, fov = int(active_sensors[0].attributes['image_size_x']), int(active_sensors[0].attributes['image_size_y']), float(active_sensors[0].attributes['fov'])
            K = build_projection_matrix(image_w, image_h, fov)


            while True:
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
                print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")
                annotation_saved = False
                # Get and save the sensor data from the queue.
                missing_sensors = sensor_amount
                while True:

                    frame = world.get_snapshot().frame
                    try:
                        sensor_data = sensor_queue.get(True, 2.0)
                        if sensor_data[1] != frame: continue  # Ignore previous frame data
                        missing_sensors -= 1
                    except Empty:
                        raise ValueError("A sensor took too long to send their data")

                    # Get the data
                    sensor_id = sensor_data[0]
                    frame_diff = sensor_data[1] - start_frame
                    data = sensor_data[2]

                    # --- 여기서 annotation 저장 ---
                    if not annotation_saved:
                        world_2_camera = np.array(active_sensors[0].get_transform().get_inverse_matrix())
                        traffics = []
                        traffics = filter_traffic_lights(vehicle, K, world_2_camera, image_w, image_h)
                        save_annotations(hero, frame_diff, endpoint, recorder_info['name'], traffics)
                        annotation_saved = True
                    # ---------------------------

                    res = threading.Thread(target=save_data_to_disk, args=(sensor_id, frame_diff, data, endpoint))
                    results.append(res)
                    res.start()

                    if CURRENT_THREADS > max_threads:
                        for res in results:
                            res.join()
                        results = []

                    if missing_sensors <= 0:
                        break

                world.tick()

            for res in results:
                res.join()

            for sensor in active_sensors:
                sensor.stop()
                sensor.destroy()
            active_sensors = []

            for _ in range(50):
                world.tick()

    # End the simulation
    finally:
        # stop and remove cameras
        for sensor in active_sensors:
            sensor.stop()
            sensor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # Remove all replay actors
        client.stop_replayer(False)
        client.apply_batch([carla.command.DestroyActor(x.id) for x in world.get_actors().filter('static.prop.mesh')])
        world.wait_for_tick()
        for v in world.get_actors().filter('*vehicle*'):
            v.destroy()
        world.wait_for_tick()


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')