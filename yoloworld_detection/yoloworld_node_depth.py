import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import Extrinsics
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
from ultralytics import YOLOWorld
import requests
import os
import io
import json

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        ###======================= 토픽 관련 세팅 =======================
        qos_profile = rclpy.qos.QoSProfile(depth=10)

        # 카메라 이미지 토픽 구독
        self.image_subscription = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.detect_callback,
            qos_profile
        )

        # Depth 이미지 토픽 구독
        self.depth_subscription = self.create_subscription(
            Image,
            'camera/camera/depth/image_rect_raw',
            self.depth_callback,
            qos_profile
        )
        # 사용자 입력 텍스트 토픽 구독
        self.text_subscription = self.create_subscription(
            String,
            '/user_text_input',  # 텍스트를 받아오는 토픽
            self.update_class_callback,
            qos_profile
        )

        # Depth 카메라 내부 매개변수 구독
        # self.camera_info_subscription = self.create_subscription(
        #     CameraInfo,
        #     'camera/camera/depth/camera_info',
        #     self.camera_info_callback,
        #     qos_profile
        # )

        # Depth-to-Color 외부 매개변수 구독
        # self.extrinsics_subscription = self.create_subscription(
        #     Extrinsics,
        #     'camera/camera/extrinsics/depth_to_color',
        #     self.extrinsics_callback,
        #     qos_profile
        # )

        self.bridge = CvBridge()
        self.depth_image = None  # Depth 데이터를 저장할 변수
        # self.camera_info = None  # 카메라 내부 매개변수
        # self.extrinsics = None  # 카메라 외부 매개변수
        self.get_logger().info("Detection Node started")

        ###======================= YOLOWorld 설정 부분 =======================
        # YOLOWorld 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU or CPU
        self.get_logger().info(f"Using device: {self.device}")

        # YOLOWorld 모델 불러오기
        self.model = YOLOWorld("yolov8s-world.pt")  # YOLOWorld 모델 가중치
        self.model.to(self.device)

        # 기본 클래스 설정
        self.current_class = "pink doll"  # 기본 클래스
        self.model.set_classes([self.current_class])
        self.get_logger().info(f"Initial YOLOWorld class: {self.current_class}")

        # 서버 URL 설정
        self.server_url = "http://192.168.0.186:8000/upload-image"  # 서버 URL

        # 이미지 저장 경로
        self.save_path = "/home/rcv/ros2_ws/frame"
        os.makedirs(self.save_path, exist_ok=True)  # 디렉토리가 없으면 생성

        ###=============================================================

    # def camera_info_callback(self, msg):
    #     """카메라 내부 매개변수 수신"""
    #     self.camera_info = msg

    # def extrinsics_callback(self, msg):
    #     """카메라 외부 매개변수 수신"""
    #     self.extrinsics = msg
    def update_class_callback(self, msg):
        """/user_text_input 토픽에서 클래스 업데이트"""
        new_class = msg.data.strip()  # 텍스트 데이터 가져오기
        if not new_class:
            self.get_logger().warn("Received empty class input. Ignoring.")
            return
        
        self.current_class = new_class
        self.model.set_classes([self.current_class])  # YOLOWorld 클래스 업데이트
        self.get_logger().info(f"Updated YOLOWorld class to: {self.current_class}")

    def depth_callback(self, data):
        """Depth 이미지 데이터를 수신"""
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough") * 0.001  # mm -> m 변환
        # self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")  # Depth 이미지를 저장

    def detect_callback(self, data):
        """이미지 데이터 처리 및 객체 탐지"""
        # ROS Image 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')  # YOLO는 BGR 형식을 처리 가능

        # if self.depth_image is None or self.camera_info is None or self.extrinsics is None:
        #     self.get_logger().warn("Missing depth data or camera parameters. Skipping detection.")
        #     #return

        # YOLOWorld 모델 추론
        results = self.model.predict(cv_image)  # YOLOWorld 모델 추론 수행

        # 탐지된 바운딩 박스와 레이블 정보를 저장할 리스트
        detection_data = []
        
######################################### depth 카메라와 base 좌표계 변환관계 알경우 : depth 카메라 기준의 point cloud 좌표 ##############################################################################################

        # # YOLOWorld 결과에서 바운딩 박스, 클래스, 신뢰도 추출 및 시각화
        # for result in results:
        #     for i, box in enumerate(result.boxes):
        #         x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표

        #         conf = box.conf[0]  # 신뢰도
        #         cls = box.cls[0]  # 클래스 ID
        #         label = self.model.names[int(cls)]  # 클래스 이름

        #         # 바운딩 박스 중심점의 Depth 값 계산
        #         center_x = (x1 + x2) // 2
        #         center_y = (y1 + y2) // 2
        #         depth_value = self.depth_image[center_y, center_x]  # 중심점의 Depth 값
                
        #         if depth_value == 0: # 너무 가까워서 댑스정보가 추출이 되지 않는 경우는 Pass
        #             continue

        #         # Depth를 바탕으로 3D 좌표 계산 - point cloud 점 하나
        #         z = depth_value  # 깊이 (미터 단위)
        #         x = (center_x - cx) * z / fx  # X 좌표
        #         y = (center_y - cy) * z / fy  # Y 좌표

        #         # 바운딩 박스와 레이블 표시
        #         cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.putText(cv_image, f"{label} ({x:.2f}, {y:.2f}, {z:.2f})", (x1, y1 - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #         # 탐지 정보를 리스트에 추가
        #         detection_data.append({
        #             "idx": i + 1,
        #             "label": label,
        #             "confidence": float(conf),
        #             "bounding_box": [x1, y1, x2, y2],
        #             "3d_coordinates": {"x": x, "y": y, "z": z}
        #         })
#######################################################################################################################################

############################################## 우리 상황 : rgbcamera - base 변환관계 알경우 : rgb 카메라 기준의 point cloud - extrinsic parameter 사용 ##########################################################################    
        # 카메라 매개변수 추출

        # fx = self.camera_info.k[0]  # Focal length x
        # fy = self.camera_info.k[4]  # Focal length y
        # cx = self.camera_info.k[2]  # Principal point x
        # cy = self.camera_info.k[5]  # Principal point y
        
        fx = 935.8029174804688 #self.camera_info.k[0]  # Focal length x
        fy = 935.8029174804688 #self.camera_info.k[4]  # Focal length y
        cx = 633.5607299804688 #self.camera_info.k[2]  # Principal point x
        cy = 340.0840148925781 #self.camera_info.k[5]  # Principal point y
        
        # print('###',self.extrinsics)
        
        # Rotation과 Translation 데이터를 리스트로 정의 -아래는 self.extrinsics으로 depth 카메라와 rgb카메라 사이의 변환행렬임!
        rotation = [
            0.9999957084655762, 0.0022826031781733036, 0.001842103316448629,
            -0.0022804313339293003, 0.9999967217445374, -0.0011803284287452698,
            -0.0018447914626449347, 0.0011761225759983063, 0.999997615814209
        ]

        translation = [
            0.014946837909519672,
            -7.593750979140168e-07,
            -0.00020997639512643218
        ]
        # NumPy 배열로 변환하여 회전 행렬(R)과 변환 벡터(T) 생성
        R = np.array(rotation).reshape(3, 3)  # 3x3 행렬로 변환
        T = np.array(translation).reshape(3, 1)  # 3x1 벡터로 변환
        
        # # Extrinsics에서 회전 행렬(R)과 변환 벡터(T) 추출
        # R = np.array(self.extrinsics.r).reshape(3, 3)
        # T = np.array(self.extrinsics.p[0:3]).reshape(3, 1)
        
        for result in results:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                
                conf = box.conf[0]  # 신뢰도
                cls = box.cls[0]  # 클래스 ID
                label = self.model.names[int(cls)]  # 클래스 이름

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                depth_value = self.depth_image[center_y, center_x]

                if depth_value == 0:
                    continue

                # Depth 좌표계에서의 3D 좌표
                z = depth_value
                depth_coords = np.array([[z * (center_x - cx) / fx], [z * (center_y - cy) / fy], [z]]) # point cloud 점 x,y,z
                # depth_coords에서 x, y, z 추출
                x, y, z = depth_coords.flatten()  # NumPy 배열을 1D로 변환 후 값 추출

                # RGB 좌표계로 변환- rgb_coords는 3x1로  NumPy 배열로 저장되어 있
                rgb_coords = R @ depth_coords + T # rgb_coords는 x',y',z' 세 개의 값을 가지고 이는 RGB 카메라 좌표계 기준의 3D 좌표를 나타낸다. -> 이 좌표를 base 좌표로 변환하면됨.
                # rgb_coords에서 x', y', z' 추출
                x_prime, y_prime, z_prime = rgb_coords.flatten()  # NumPy 배열을 1D로 변환 후 각각의 값 추출
                
                # 재투영하여 RGB 이미지 좌표로 변환 -> 이거는 픽셀 기준 좌표계임.
                u = int(rgb_coords[0] * fx / rgb_coords[2] + cx)
                v = int(rgb_coords[1] * fy / rgb_coords[2] + cy)

                # 바운딩 박스와 레이블 표시
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f"{label}x:{center_x},y:{center_y},z:{z:.2f} (u:{u} v:{v})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                 # 탐지 정보를 리스트에 추가
                detection_data.append({
                    "idx": i + 1,
                    "label": label,
                    "confidence": float(conf),
                    "bounding_box": [x1, y1, x2, y2],
                    "depth_3d_coordinates": {"x": x, "y": y, "z": z},
                    "rgb_3d_coordinates": {"x'": x_prime, "y'": y_prime, "z'": z_prime}
                })
########################################################################################################################
        # 탐지 결과를 화면에 표시
        cv2.imshow('YOLOWorld Object Detection', cv_image)

        # 'C' 키를 눌렀을 때 현재 프레임 저장 및 서버 전송
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):  # 'C' 키 입력
            self.capture_and_send_frame(cv_image, detection_data)

    def capture_and_send_frame(self, frame, detection_data):
        """현재 프레임과 탐지 정보를 저장하고 서버에 전송"""
        # 저장 파일 경로 설정
        filename = "captured_frame.jpg"
        file_path = os.path.join(self.save_path, filename)

        # 이미지를 로컬에 저장
        cv2.imwrite(file_path, frame)
        self.get_logger().info(f"Frame saved locally at: {file_path}")

        # 이미지를 JPEG 형식으로 메모리에 저장
        _, buffer = cv2.imencode('.jpg', frame)
        image_data = io.BytesIO(buffer)  # io.BytesIO는 서버로 이미지를 전송하기 위해 JPEG 데이터를 메모리 버퍼로 처리하는 데 사용됨.

        detection_data_json = json.dumps(detection_data)  # JSON 문자열로 변환
        # 서버로 POST 요청 전송
        try:
            response = requests.post(
                self.server_url,
                files={"file": (filename, image_data, "image/jpeg")},  # 이미지 데이터
                data={"detection_data": detection_data_json}  # 탐지 정보 JSON 문자열
            )
            if response.status_code == 200:
                self.get_logger().info(f"Frame and detection data sent to server successfully: {response.json()}")
            else:
                self.get_logger().error(f"Failed to send frame and detection data to server. Status code: {response.status_code}")
        except Exception as e:
            self.get_logger().error(f"Error while sending frame and detection data to server: {e}")
        finally:  # 추가 작성 - 객체가 제대로 해제되지 않으면 메모리 누수가 발생할 수 있음
            image_data.close()  # 메모리 해제

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # 추가 작성


if __name__ == '__main__':
    main()
