import RPi.GPIO as GPIO
import time

# GPIO 설정
GPIO.setmode(GPIO.BCM)

# 좌측 모터 핀
left_pins = {'IN1': 14, 'IN2': 15, 'IN3': 18, 'IN4': 23, 'PWM': 12}

# 우측 모터 핀
right_pins = {'IN1': 17, 'IN2': 27, 'IN3': 22, 'IN4': 10, 'PWM': 13}

# 핀 초기화
for pin in list(left_pins.values()) + list(right_pins.values()):
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# PWM 설정
left_pwm = GPIO.PWM(left_pins['PWM'], 1000)
right_pwm = GPIO.PWM(right_pins['PWM'], 1000)
left_pwm.start(0)
right_pwm.start(0)

def drive_forward():
    # 좌측 모터 정방향
    GPIO.output(left_pins['IN1'], GPIO.LOW)
    GPIO.output(left_pins['IN2'], GPIO.HIGH)
    GPIO.output(left_pins['IN3'], GPIO.HIGH)
    GPIO.output(left_pins['IN4'], GPIO.LOW)
    
    # 우측 모터 정방향
    GPIO.output(right_pins['IN1'], GPIO.LOW)
    GPIO.output(right_pins['IN2'], GPIO.HIGH)
    GPIO.output(right_pins['IN3'], GPIO.HIGH)
    GPIO.output(right_pins['IN4'], GPIO.LOW)
    
    # PWM 속도 설정
    left_pwm.ChangeDutyCycle(70)
    right_pwm.ChangeDutyCycle(70)

def stop():
    for pin in [left_pins['IN1'], left_pins['IN2'], left_pins['IN3'], left_pins['IN4'],
                right_pins['IN1'], right_pins['IN2'], right_pins['IN3'], right_pins['IN4']]:
        GPIO.output(pin, GPIO.LOW)
    left_pwm.ChangeDutyCycle(0)
    right_pwm.ChangeDutyCycle(0)

try:
    while True:
        cmd = input("명령 입력 (s: 시작, x: 정지, q: 종료): ").strip().lower()
        if cmd == 's':
            drive_forward()
            print("모터 작동 중")
        elif cmd == 'x':
            stop()
            print("모터 정지")
        elif cmd == 'q':
            print("종료합니다.")
            break
        else:
            print("잘못된 입력입니다. 's', 'x', 'q' 중 하나를 입력하세요.")

except KeyboardInterrupt:
    print("\n강제 종료됨")

finally:
    stop()
    GPIO.cleanup()
