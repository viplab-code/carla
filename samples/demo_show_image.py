# OpenCV demo

import carla
import random
import numpy as np
import cv2
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 차량 생성
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    # 카메라 센서 설정
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 차량 앞쪽 위
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 비디오 저장 설정
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('carla_output.avi', fourcc, 20.0, (800, 600))

    # 이미지 콜백 함수
    def save_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        frame = array[:, :, :3]  # BGRA → BGR
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB로 변환
#        out.write(frame)
        cv2.imshow("CARLA View", frame)
        cv2.waitKey(-1)

    camera.listen(lambda image: save_image(image))

    # 10초 녹화
    time.sleep(10)

    # 정리
    print("Stopping and cleaning up")
    camera.stop()
#    out.release()
    cv2.destroyAllWindows()
    camera.destroy()
    vehicle.destroy()

if __name__ == '__main__':
    main()
