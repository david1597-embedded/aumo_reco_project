from aumo.model import YOLOModel

model = YOLOModel('yolov5nu', 320)

model.load()

model.export('./')