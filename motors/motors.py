import RPi.GPIO as GPIO
import time



class MotorController:
    def __init__(self):
        # GPIO 설정
        GPIO.setmode(GPIO.BCM)
        # 좌측 모터 핀
        self.left_pins = {'IN1': 14, 'IN2': 15, 'IN3': 18, 'IN4': 23, 'PWM': 12}

        # 우측 모터 핀
        self.right_pins = {'IN1': 17, 'IN2': 27, 'IN3': 22, 'IN4': 10, 'PWM': 13}

        self.left_pwm = GPIO.PWM(self.left_pins['PWM'], 2300)
        self.right_pwm = GPIO.PWM(self.right_pins['PWM'], 2300)

        self.motorInit()
        
    def motorInit(self):
        # 핀 초기화
        for pin in list(self.left_pins.values()) + list(self.right_pins.values()):
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
      
        self.left_pwm.start(0)
        self.right_pwm.start(0)
    
    def left_motor_control(self, direction):
        if direction == 'FW':
            GPIO.output(self.left_pins['IN1'], GPIO.LOW)
            GPIO.output(self.left_pins['IN2'], GPIO.HIGH)
            GPIO.output(self.left_pins['IN3'], GPIO.HIGH)
            GPIO.output(self.left_pins['IN4'], GPIO.LOW)
        elif direction == 'BW':
            GPIO.output(self.left_pins['IN1'], GPIO.HIGH)
            GPIO.output(self.left_pins['IN2'], GPIO.LOW)
            GPIO.output(self.left_pins['IN3'], GPIO.LOW)
            GPIO.output(self.left_pins['IN4'], GPIO.HIGH)


    def right_motor_control(self, direction):
        if direction == 'FW':
            # 우측 모터 앞으로 회전
            GPIO.output(self.right_pins['IN1'], GPIO.LOW)
            GPIO.output(self.right_pins['IN2'], GPIO.HIGH)
            GPIO.output(self.right_pins['IN3'], GPIO.HIGH)
            GPIO.output(self.right_pins['IN4'], GPIO.LOW)
        elif direction == 'BW':
            GPIO.output(self.right_pins['IN1'], GPIO.HIGH)
            GPIO.output(self.right_pins['IN2'], GPIO.LOW)
            GPIO.output(self.right_pins['IN3'], GPIO.LOW)
            GPIO.output(self.right_pins['IN4'], GPIO.HIGH)
    
    def motor_forward(self):
    # 좌측 모터 정방향
        self.left_pwm.ChangeFrequency(2300)
        self.right_pwm.ChangeFrequency(2300)
        self.left_motor_control('FW')
        self.right_motor_control('FW')        
        # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(74)
        self.right_pwm.ChangeDutyCycle(74)
    
    def motor_backward(self):
        self.left_pwm.ChangeFrequency(2300)
        self.right_pwm.ChangeFrequency(2300)
        self.left_motor_control('BW')
        self.right_motor_control('BW') 
        # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(100)
        self.right_pwm.ChangeDutyCycle(100)

    def motor_rotate_CW(self):
        self.left_pwm.ChangeFrequency(100)
        self.right_pwm.ChangeFrequency(100)
        self.left_motor_control('FW')
        self.right_motor_control('BW')
          # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(100)
        self.right_pwm.ChangeDutyCycle(100)
       

    def motor_rotate_CCW(self):
        self.left_pwm.ChangeFrequency(100)
        self.right_pwm.ChangeFrequency(100)
        self.left_motor_control('BW')
        self.rigth_motor_control('FW')
          # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(100)
        self.right_pwm.ChangeDutyCycle(100)
       

    def motor_stop(self):
        for pin in [self.left_pins['IN1'], self.left_pins['IN2'], self.left_pins['IN3'], self.left_pins['IN4'],
                    self.right_pins['IN1'], self.right_pins['IN2'], self.right_pins['IN3'], self.right_pins['IN4']]:
            GPIO.output(pin, GPIO.LOW)
        self.left_pwm.ChangeDutyCycle(0)
        self.right_pwm.ChangeDutyCycle(0)

    def calculate_rotate_speed(self):
        pass

    def my_position(self,realsenseCamera,box, depth_frame):
        # 바운딩 박스 중심 좌표 계산
        x1, y1, width, height = box
        px = int(x1 + width / 2)
        py = int(y1 + height / 2)
        
        # 거리 측정
        distance = realsenseCamera.measuredistance(depth_frame, px, py)
        
        # 각도 계산
        yaw, pitch = realsenseCamera.measureangle(px, py, distance)
        
        if distance >= 0.75:

            # 이동 거리: distance - 0.5m
            move_distance = max(0, distance - 0.5)  # 음수 방지
            
            # 실험 데이터 기반 속도
            linear_speed = 1 / 3  # 1m에 3초 → 0.3333 m/s
            angular_speed = 20  # 180도에 9초 → 20 deg/s
            
            # 회전 시간 계산
            rotate_time = abs(yaw) / angular_speed
            
            # 이동 시간 계산
            move_time = move_distance / linear_speed
            
            # 회전 수행
            if yaw > 0:
                self.motor_rotate_CW()
                time.sleep(rotate_time)
            elif yaw < 0:
                self.motor_rotate_CCW()
                time.sleep(rotate_time)
            
            # 이동 수행
            if move_distance > 0:
                self.motor_forward()
                time.sleep(move_time)
            
            # 모터 정지
            self.motor_stop()

            return distance, yaw
        elif:
            return distance, yaw

            
        
        

