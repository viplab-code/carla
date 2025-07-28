import os
import math
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from contextlib import contextmanager

import cv2
import pygame
from pygame.locals import QUIT
import numpy as np

import carla


PROFILE_STATS: Dict[str, List[float]] = defaultdict(list)

@contextmanager
def timer(name: str):
    """
    Measure wall‑clock time of a code block and record it in the global
    PROFILE_STATS dictionary under *name*.
    """
    _start = time.perf_counter()
    try:
        yield
    finally:
        PROFILE_STATS[name].append(time.perf_counter() - _start)


SEG_ID_LIGHT_BACK = 35
SEG_ID_LIGHT_FRONT = 36

LIGHT_FRONT_FLAGS = carla.VehicleLightState(
    carla.VehicleLightState.LowBeam
    | carla.VehicleLightState.HighBeam
)

LIGHT_BACK_FLAGS = carla.VehicleLightState(
    carla.VehicleLightState.Position
    | carla.VehicleLightState.Brake
    | carla.VehicleLightState.RightBlinker
    | carla.VehicleLightState.LeftBlinker
    | carla.VehicleLightState.Reverse
    | carla.VehicleLightState.Fog
)

CAR_IDS = [
    "vehicle.tesla.model3", 
    "vehicle.ford.crown", 
    "vehicle.mercedes.coupe_2020", 
    "vehicle.mini.cooper_s_2021", 
    "vehicle.dodge.charger_2020", 
    "vehicle.dodge.charger_police_2020", 
    "vehicle.lincoln.mkz_2020", 
    "vehicle.nissan.patrol_2021", 
    "vehicle.audi.tt", 
    "vehicle.volkswagen.t2_2021", 
    # "vehicle.mercedes.sprinter", 
    # "vehicle.ford.ambulance", 
]


def parse_arguments():
    argparser = argparse.ArgumentParser(description='CARLA Light Dataset Client')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int)
    argparser.add_argument('-m', '--map-name', metavar='M', default=None)
    argparser.add_argument('-w', '--weather', metavar='WEATHER', default="ClearNight")
    argparser.add_argument('-e', '--ego', metavar='VEH_ID', default='vehicle.lincoln.mkz_2020')
    argparser.add_argument('-r', '--res', metavar='WIDTHxHEIGHT', default='1280x704')
    argparser.add_argument('-o', '--output-dir', metavar='DIR', default=False)
    argparser.add_argument('-n', '--max-frames', type=int, default=0)
    argparser.add_argument('--headless', action='store_true', default=False)
    argparser.add_argument('--num-vehicles', type=int, default=100)
    argparser.add_argument('--fps', type=int, default=24)
    argparser.add_argument('--cams', metavar='LIST', default='front', help='front,rear')
    argparser.add_argument('--debug-cam', default='front', choices=['front', 'rear'])
    argparser.add_argument('--front-offset-x', type=float, default=1.46)
    argparser.add_argument('--front-offset-z', type=float, default=1.3)
    argparser.add_argument('--rear-offset-x', type=float, default=-2.05)
    argparser.add_argument('--rear-offset-z', type=float, default=1.3)

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.cams = [c.strip() for c in args.cams.split(',') if c.strip()]

    return args


class LightManager:
    def __init__(self, front_flags: carla.VehicleLightState, back_flags: carla.VehicleLightState, blk_len: float = 1.0, duty: float = 0.5):
        self._front_flags = front_flags
        self._back_flags  = back_flags
        self._blk_len = blk_len
        self._duty    = duty
        self._table: Dict[int, Tuple[int, float, int, float]] = {}
        self.current_flags: Dict[int, int] = {}

    def _is_on(self, seed: int, phase: float, sim_t: float):
        block = int((sim_t + phase) // self._blk_len)
        rng   = np.random.RandomState(seed + block)
        return rng.rand() < self._duty

    def register(self, actor: carla.Actor):
        aid = actor.id
        seed_f = (aid << 1) ^ 0xA5A5
        seed_b = (aid << 1) ^ 0x5A5A
        phase_f = np.random.rand() * self._blk_len
        phase_b = np.random.rand() * self._blk_len
        self._table[aid] = (seed_f, phase_f, seed_b, phase_b)

    def update(self, world: carla.World, actors: List[carla.Actor]):
        sim_t = world.get_snapshot().timestamp.elapsed_seconds
        for a in actors:
            if a.id not in self._table:
                self.register(a)
            seed_f, phase_f, seed_b, phase_b = self._table[a.id]
            flags = carla.VehicleLightState.NONE
            if self._is_on(seed_f, phase_f, sim_t):
                flags |= self._front_flags
            if self._is_on(seed_b, phase_b, sim_t):
                flags |= self._back_flags
            a.set_light_state(carla.VehicleLightState(flags))
            self.current_flags[a.id] = int(flags)


class Display:
    def __init__(self, width: int, height: int, headless: bool):
        self.headless = headless
        self.width = width
        self.height = height
        if headless:
            self.screen = None
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption('CARLA Light Demo')

    def handle_events(self):
        if self.headless:
            return
        for event in pygame.event.get():
            if event.type == QUIT:
                raise KeyboardInterrupt

    def draw(self, surface, lines=None, overlays=None):
        if self.headless or self.screen is None:
            return

        if surface is not None:
            frame = pygame.transform.scale(surface, (self.width, self.height))
            self.screen.blit(frame, (0, 0))
        else:
            self.screen.fill((0, 0, 0))

        if overlays:
            for ov in overlays:
                if ov is None:
                    continue
                ov_resized = pygame.transform.scale(ov, (self.width, self.height))
                self.screen.blit(ov_resized, (0, 0))

        if lines:
            for x0, y0, x1, y1, w in lines:
                pygame.draw.line(self.screen, (255, 0, 0), (x0, y0), (x1, y1), w)

    def flip(self):
        if not self.headless and self.screen is not None:
            pygame.display.flip()

    def cleanup(self):
        if not self.headless and self.screen is not None:
            pygame.quit()


class App:
    def __init__(self, args):
        self.frame_idx = 0
        self.args = args
        self._init_client()
        self._configure_world()
        self._spawn_actors()
        self._init_light_manager()
        self._setup_spectator()
        self._init_sensors()

        self.output_dir = self.args.output_dir or None
        if self.output_dir:
            os.makedirs(os.path.join(self.output_dir, "meta"), exist_ok=True)
            for sub in ("rgb", "seg"):
                for cam in self.args.cams:
                    os.makedirs(os.path.join(self.output_dir, sub, cam), exist_ok=True)

            cam_meta = {
                cam: {
                    "width":  self.args.width,
                    "height": self.args.height,
                    "fov": 90.0,
                    "sensor_tick": 1.0 / self.args.fps
                } for cam in self.args.cams
            }
            with open(os.path.join(self.output_dir, "cameras.json"), "w") as f:
                json.dump(cam_meta, f, indent=2, ensure_ascii=False)

        self._pending: Dict[int, Dict[str, dict]] = defaultdict(lambda: defaultdict(dict))

        self.display = Display(self.args.width, self.args.height, self.args.headless)

        self.world.tick()

    def _init_client(self):
        self.client = carla.Client(self.args.host, self.args.port)
        self.world  = (self.client.load_world(self.args.map_name)
                       if self.args.map_name else self.client.get_world())
        self.blueprint_library = self.world.get_blueprint_library()
        self.tmap = self.world.get_map()
    
    def _configure_world(self):
        # Weather
        weather_param = getattr(carla.WeatherParameters, self.args.weather)
        self.world.set_weather(weather_param)

        # Traffic Manager
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.tm_port = self.tm.get_port()

        # 고정 Δt + sync
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.args.fps
        self.world.apply_settings(settings)

    def _spawn_actors(self):
        spawns = self.tmap.get_spawn_points()
        np.random.shuffle(spawns)
        ego_tf  = np.random.choice(spawns)
        spawns.remove(ego_tf)
        ego_bp  = self.blueprint_library.find(self.args.ego)
        self.ego = self.world.spawn_actor(ego_bp, ego_tf)
        self._configure_vehicle(self.ego)

        self.traffic_vehicles = []
        if self.args.num_vehicles > 0 and spawns:
            allowed = set(CAR_IDS)
            veh_bps = [bp for bp in self.blueprint_library.filter('vehicle.*')
                       if bp.id in allowed]
            for tf in spawns[:self.args.num_vehicles]:
                bp = np.random.choice(veh_bps)
                tv = self.world.spawn_actor(bp, tf)
                self._configure_vehicle(tv)
                self.traffic_vehicles.append(tv)

    def _init_light_manager(self):
        self.light_manager = LightManager(LIGHT_FRONT_FLAGS, LIGHT_BACK_FLAGS)
        for vehicle in self.traffic_vehicles:
            self.light_manager.register(vehicle)
    
    def _init_sensors(self):
        self.cam_sensors: Dict[str, carla.Actor] = {}
        self.seg_sensors: Dict[str, carla.Actor] = {}
        self.camera_surface: Dict[str, pygame.Surface] = {}
        self.camera_rgb: Dict[str, np.ndarray] = {}
        self.seg_overlay: Dict[str, List[pygame.Surface]] = {}
        self.seg_frame: Dict[str, Tuple[int, float, np.ndarray]] = {}
        self.active_cam_sensor: carla.Actor = None  # for bbox projection

        for cam in self.args.cams:
            rgb_sensor = self._setup_sensor('sensor.camera.rgb', cam)
            seg_sensor = self._setup_sensor('sensor.camera.instance_segmentation', cam)
            self.cam_sensors[cam] = rgb_sensor
            self.seg_sensors[cam] = seg_sensor

            # buffers
            self.camera_surface[cam] = None
            self.camera_rgb[cam] = None
            self.seg_overlay[cam] = []
            self.seg_frame[cam] = None

            def _mk_rgb_cb(name):
                def _cb(image):
                    rgb = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                        (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
                    self.camera_rgb[name] = rgb
                    if not self.args.headless and (self.args.debug_cam in (name, 'both')):
                        self.camera_surface[name] = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

                    d = self._pending[image.frame][name]
                    d["rgb"] = rgb
                    d["ts"] = image.timestamp
                    tf = image.transform
                    d["extrinsic"] = {
                        "location": [tf.location.x, tf.location.y, tf.location.z],
                        "rotation": [tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll],
                    }
                return _cb

            def _mk_seg_cb(name):
                def _cb(image):
                    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                        (image.height, image.width, 4))
                    self.seg_frame[name] = (image.frame, image.timestamp, arr)
                    if not self.args.headless and (self.args.debug_cam in (name, 'both')):
                        back = self.make_seg_overlay(arr, SEG_ID_LIGHT_BACK, (0,255,0), True)
                        front = self.make_seg_overlay(arr, SEG_ID_LIGHT_FRONT, (0,0,255), True)
                        self.seg_overlay[name] = [back, front]
                    self._pending[image.frame][name]["seg"] = arr
                return _cb

            rgb_sensor.listen(_mk_rgb_cb(cam))
            seg_sensor.listen(_mk_seg_cb(cam))

            if self.active_cam_sensor is None and (self.args.debug_cam in (cam, 'both')):
                self.active_cam_sensor = rgb_sensor

        if self.active_cam_sensor is None:
            self.active_cam_sensor = next(iter(self.cam_sensors.values()))

        self.cam_sensor = self.active_cam_sensor

    def _configure_vehicle(self, v: carla.Vehicle):
        v.set_autopilot(True, self.tm_port)
        self.tm.auto_lane_change(v, True)
        self.tm.ignore_lights_percentage(v, 0)
        self.tm.ignore_signs_percentage(v, 0)
        self.tm.update_vehicle_lights(v, False)

    def _setup_spectator(self):
        bbox = self.ego.bounding_box.extent
        self.spec_offset_pitch = 0.0
        # Per‑camera offsets driven by CLI arguments
        self.cam_offsets = {
            'front': (self.args.front_offset_x, self.args.front_offset_z),
            'rear':  (self.args.rear_offset_x,  self.args.rear_offset_z),
        }
        self.spectator = self.world.get_spectator()

    def _update_spectator(self):
        if self.args.headless:
            return
        spec_x, spec_z = self.cam_offsets['front']
        vehicle_tf = self.ego.get_transform()
        yaw = vehicle_tf.rotation.yaw

        offset_loc = carla.Location(
            spec_x * math.cos(math.radians(yaw)),
            spec_x * math.sin(math.radians(yaw)),
            spec_z,
        )
        spectator_loc = vehicle_tf.location + offset_loc
        spectator_tf  = carla.Transform(
            spectator_loc,
            carla.Rotation(pitch=self.spec_offset_pitch, yaw=yaw)
        )
        self.spectator.set_transform(spectator_tf)

    def _setup_sensor(self, sensor_name, where: str = 'front'):
        sensor_bp = self.blueprint_library.find(sensor_name)
        sensor_bp.set_attribute('image_size_x', str(self.args.width))
        sensor_bp.set_attribute('image_size_y', str(self.args.height))
        sensor_bp.set_attribute('fov', '90')
        sensor_bp.set_attribute('sensor_tick', f'{(1.0/self.args.fps):.6f}')

        # Use unified offsets defined in _setup_spectator
        loc_x, loc_z = self.cam_offsets.get(where, self.cam_offsets['front'])
        loc = carla.Location(x=loc_x, y=0.0, z=loc_z)

        yaw = 0 if where == 'front' else (180 if where == 'rear' else 0)
        rot = carla.Rotation(pitch=self.spec_offset_pitch, yaw=yaw)

        return self.world.spawn_actor(
            sensor_bp,
            carla.Transform(loc, rot),
            attach_to=self.ego
        )

    def cleanup(self):
        for sensor_dict in (getattr(self, "cam_sensors", {}), getattr(self, "seg_sensors", {})):
            for s in sensor_dict.values():
                s.stop()

            for s in sensor_dict.values():
                s.destroy()

        self.ego.destroy()

        for vehicle in getattr(self, 'traffic_vehicles', []):
            vehicle.destroy()

        if hasattr(self, "display"):
            self.display.cleanup()

    def _collect_world_bboxes_meta(self) -> List[dict]:
        with timer("_collect_world_bboxes_meta"):
            meta = []
            for veh in self.world.get_actors().filter("vehicle.*"):
                corners = veh.bounding_box.get_world_vertices(veh.get_transform())
                meta.append({
                    "actor_id": int(veh.id),
                    "type_id":  veh.type_id,
                    "world_corners": [[v.x, v.y, v.z] for v in corners]
                })
            return meta

    def _collect_light_meta(self) -> List[dict]:
        meta = []
        for aid, mask in self.light_manager.current_flags.items():
            meta.append({"actor_id": aid, "light_state": mask})
        return meta

    def _collect_visible_bbox_lines(self):
        with timer("collect_visible_bbox_lines"):
            surf = self.camera_surface.get(self.args.debug_cam, next(iter(self.camera_surface.values()))) \
                if isinstance(self.camera_surface, dict) else self.camera_surface[0]
            if surf is None:
                return []

            w_img, h_img = self.args.width, self.args.height
            fov   = 90.0
            focal = w_img / (2.0 * math.tan(math.radians(fov) / 2.0))
            cx, cy = w_img / 2.0, h_img / 2.0

            cam_tf = self.active_cam_sensor.get_transform()
            try:
                inv_mat = np.array(cam_tf.get_inverse_matrix())
            except AttributeError:
                inv_mat = np.linalg.inv(np.array(cam_tf.get_matrix()))

            EDGE = [
                (0, 1), (1, 3), (3, 2), (2, 0), # rear face
                (0, 4), (1, 5), (2, 6), (3, 7), # sides
                (4, 5), (5, 7), (7, 6), (6, 4), # front face
            ]

            lines = []

            for veh in self.world.get_actors().filter('vehicle.*'):
                if veh.id == self.ego.id:
                    continue

                corners = veh.bounding_box.get_world_vertices(veh.get_transform())
                pts2d, depths = [], []
                for v in corners:
                    cp = inv_mat.dot(np.array([v.x, v.y, v.z, 1.0]))
                    u, v_, z = cp[1], -cp[2], (cp[0] if cp[0] > 0 else 0.001)
                    x2d = int((focal * u / z) + cx)
                    y2d = int((focal * v_ / z) + cy)
                    pts2d.append((x2d, y2d))
                    depths.append(z)

                if len(pts2d) == 8 and not all(d <= 0 for d in depths):
                    for e0, e1 in EDGE:
                        lines.append((pts2d[e0][0], pts2d[e0][1],
                                    pts2d[e1][0], pts2d[e1][1], 3))
            return lines

    def _save_frame_sync(self,
                         fid: int,
                         ts: float,
                         rgb: np.ndarray,
                         seg: np.ndarray,
                         cam: str):
        if not self.output_dir:
            return

        ts_us = int(ts * 1e6)
        base  = f"{fid:06d}_{ts_us:016d}"

        # RGB
        cv2.imwrite(
            os.path.join(self.output_dir, "rgb", cam, f"{base}.jpg"),
            rgb[:, :, ::-1],  # RGB→BGR
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        # Seg
        seg_bgr = seg[:, :, :3]
        cv2.imwrite(os.path.join(self.output_dir, "seg", cam, f"{base}.png"), seg_bgr)

    def _save_meta(self,
                   fid: int,
                   ts: float,
                   cameras_extrinsic: Dict[str, dict],
                   bbox_meta: List[dict],
                   light_meta: List[dict]):
        if not self.output_dir:
            return
        ts_us = int(ts * 1e6)
        base  = f"{fid:06d}_{ts_us:016d}"
        path  = os.path.join(self.output_dir, "meta", f"{base}.json")
        if os.path.exists(path):
            return  # already written
        with open(path, "w") as f:
            json.dump({
                "frame": fid,
                "timestamp_us": ts_us,
                "bounding_boxes": bbox_meta,
                "lights": light_meta,
                "cameras": cameras_extrinsic
            }, f, indent=2, ensure_ascii=False)

    def _flush_ready_frames(self):
        for fid in sorted(list(self._pending.keys())):
            cam_dict = self._pending[fid]
            if not all(cam in cam_dict and "rgb" in cam_dict[cam] and "seg" in cam_dict[cam]
                       for cam in self.args.cams):
                continue

            bbox_meta  = self._collect_world_bboxes_meta()
            light_meta = self._collect_light_meta()
            cameras_extr = {cam: d["extrinsic"] for cam, d in cam_dict.items()}
            ts_any = next(iter(cam_dict.values()))["ts"]

            # save files
            for cam, d in cam_dict.items():
                self._save_frame_sync(
                    fid, d["ts"], d["rgb"], d["seg"], cam
                )

            self._save_meta(fid, ts_any, cameras_extr, bbox_meta, light_meta)

            # cleanup
            self._pending.pop(fid)

    @staticmethod
    def make_seg_overlay(seg_arr: np.ndarray,
                         seg_id: int,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         outline: bool = False) -> pygame.Surface:
        with timer("make_seg_overlay"):
            mask = seg_arr[:, :, 2] == seg_id
            if outline:
                neighbour_and = (
                    np.roll(mask, 1, 0) & np.roll(mask, -1, 0) &
                    np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
                )
                mask &= ~neighbour_and # edge만 남김
            overlay = np.zeros(seg_arr.shape[:2] + (3,), np.uint8)
            overlay[mask] = color
            surf = pygame.surfarray.make_surface(overlay.swapaxes(0, 1))
            surf.set_colorkey((0, 0, 0))
            return surf


def main():
    args = parse_arguments()
    app = App(args)

    pbar = tqdm(total=(args.max_frames if args.max_frames else None),
                desc="Ticks", unit="tick", dynamic_ncols=True, smoothing=0.05)

    try:
        while True:
            if args.max_frames and app.frame_idx >= args.max_frames:
                raise KeyboardInterrupt

            with timer("display"):
                if not args.headless:
                    app.display.handle_events()
                    bbox_lines = app._collect_visible_bbox_lines()
                    surf = app.camera_surface.get(app.args.debug_cam,
                              next(iter(app.camera_surface.values())))
                    ov   = app.seg_overlay.get(app.args.debug_cam, [])
                    app.display.draw(surf, lines=bbox_lines, overlays=ov)
                    app.display.flip()

            with timer("light_manager.update"):
                app.light_manager.update(app.world, app.traffic_vehicles)

            if app.output_dir:
                with timer("_flush_ready_frames"):
                    app._flush_ready_frames()

            with timer("world.tick"):
                app.world.tick()

            app._update_spectator()

            pbar.update(1)
            app.frame_idx += 1

            # # Print a short profiling summary every 100 ticks
            # if len(PROFILE_STATS["world.tick"]) % 100 == 0 and PROFILE_STATS["world.tick"]:
            #     print("\n=== Profiling summary (last 100 ticks) ===")
            #     for key, samples in PROFILE_STATS.items():
            #         print(f"{key:25s}: {statistics.mean(samples[-100:]) * 1000:.2f} ms (avg)")
            #     print("──────────────────────────────────────────────────────────")

    except KeyboardInterrupt:
        pass

    finally:
        app.cleanup()


if __name__ == '__main__':
    main()
