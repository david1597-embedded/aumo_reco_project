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
        self.left_motor_control('FW')
        self.rigth_motor_control('FW')        
        # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(74)
        self.right_pwm.ChangeDutyCycle(74)
    
    def motor_backward(self):
        self.left_motor_control('BW')
        self.rigth_motor_control('BW') 
        # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(74)
        self.right_pwm.ChangeDutyCycle(74)

    def motor_rotate_CW(self):
        self.left_motor_control('FW')
        self.rigth_motor_control('BW')
          # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(74)
        self.right_pwm.ChangeDutyCycle(74)
       

    def motor_rotate_CCW(self):
        self.left_motor_control('BW')
        self.rigth_motor_control('FW')
          # PWM 속도 설정
        self.left_pwm.ChangeDutyCycle(74)
        self.right_pwm.ChangeDutyCycle(74)
       

    def motor_stop(self):
        for pin in [self.left_pins['IN1'], self.left_pins['IN2'], self.left_pins['IN3'], self.left_pins['IN4'],
                    self.right_pins['IN1'], self.right_pins['IN2'], self.right_pins['IN3'], self.right_pins['IN4']]:
            GPIO.output(pin, GPIO.LOW)
        self.left_pwm.ChangeDutyCycle(0)
        self.right_pwm.ChangeDutyCycle(0)

    def calculate_rotate_speed(self):
        pass

    def my_position(self, box):
        pass

    

