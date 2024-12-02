import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
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
        
        #  # 카메라 뎁스 맵 토픽 구독 t
        # self.depth_subscription = self.create_subscription(
        #     Image,
        #     'camera/camera/color/image_depth',
        #     self.detect_callback,
        #     qos_profile
        # )

        # 사용자 입력 텍스트 토픽 구독
        self.text_subscription = self.create_subscription(
            String,
            '/user_text_input',  # 텍스트를 받아오는 토픽
            self.update_class_callback,
            qos_profile
        )

        self.bridge = CvBridge()
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

    def update_class_callback(self, msg):
        """/user_text_input 토픽에서 클래스 업데이트"""
        new_class = msg.data.strip()  # 텍스트 데이터 가져오기
        if not new_class:
            self.get_logger().warn("Received empty class input. Ignoring.")
            return
        
        self.current_class = new_class
        self.model.set_classes([self.current_class])  # YOLOWorld 클래스 업데이트
        self.get_logger().info(f"Updated YOLOWorld class to: {self.current_class}")

    def detect_callback(self, data):
        """이미지 데이터 처리 및 객체 탐지"""
        # ROS Image 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')  # YOLO는 BGR 형식을 처리 가능

        # YOLOWorld 모델 추론
        results = self.model.predict(cv_image)  # YOLOWorld 모델 추론 수행

        # 탐지된 바운딩 박스와 레이블 정보를 저장할 리스트
        detection_data = []

        # YOLOWorld 결과에서 바운딩 박스, 클래스, 신뢰도 추출 및 시각화
        for result in results:
            for i,box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                
                print(f'##############{x1},{y1},{x2},{y2}###############')
                
                conf = box.conf[0]  # + 0.4 신뢰도
                cls = box.cls[0]  # 클래스 ID
                label = f"{self.model.names[int(cls)]} {conf:.2f}"  # 클래스 이름과 신뢰도

                # 바운딩 박스와 레이블 표시
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 탐지 정보를 리스트에 추가
                detection_data.append({
                    "idx": i+1 ,#int(cls.item()), # 인덱스 정보 추가
                    "label": self.model.names[int(cls)],
                    "confidence": float(conf),  # JSON 직렬화를 위해 float 변환
                    "bounding_box": [x1, y1, x2, y2]
                })

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
        image_data = io.BytesIO(buffer)  # . io.BytesIO는 서버로 이미지를 전송하기 위해 JPEG 데이터를 메모리 버퍼로 처리하는 데 사용됨.
        
        print('-=-=-=-=-=-=789098789',detection_data,'9890989098-=-=-=-=-=-=-==-')
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
