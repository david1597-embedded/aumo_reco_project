# Au-mo deto solution (Auto-moto call with object and hand gesture detection)
객체인식모델을 활용하여 손동작에 따른 이동장치 제어. 요청자와 장치간의 거리를 측정하고, 직선 경로를 정의하고 사용자의 손동작에 따라 모터제어 입력값을 다르게하여 차량을 제어하는 솔루션을 개발하는 것이 목표.

나아가 솔루션을 활용해 거동불편자들을 위한 물건 운반, 공사현장에서의 자재 운반 등의 다양한 프로젝트에 적용 가능한 솔루션 개발이 목표


## High Level Desgin
![high-level-desing-img](./doc/hld.png)

## Use case
![use-case-img](./doc/usecase.jpg)

## 손동작출력 라벨
전진 -> one\
후진 -> two\
제자리회전(우) ->three2\
제자리회전(좌) ->three\
정지 -> fist\
내 자리로 오기 ->four\
따라오게 하기 ->stop\
일반 모드 전환(대기 상태 해제) -> rock\
