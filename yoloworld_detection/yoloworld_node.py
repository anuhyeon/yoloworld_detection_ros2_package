import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLOWorld

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

        # YOLOWorld 결과에서 바운딩 박스, 클래스, 신뢰도 추출 및 시각화
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                conf = box.conf[0]  # + 0.4 신뢰도
                cls = box.cls[0]  # 클래스 ID
                label = f"{self.model.names[int(cls)]} {conf:.2f}"  # 클래스 이름과 신뢰도

                # 바운딩 박스와 레이블 표시
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 탐지 결과를 화면에 표시
        cv2.imshow('YOLOWorld Object Detection', cv_image)
        cv2.waitKey(1)

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

if __name__ == '__main__':
    main()
