from aumo.evaluate import Evaluator
from aumo.model import YOLOModel  # predict(img) 반환 모델
from aumo.ov_model import OVDetectionModel
import json
import os

def load_coco_dataset(images_dir, annotation_json_path):
    with open(annotation_json_path, 'r') as f:
        coco = json.load(f)

    # 이미지 id -> 파일명 맵핑
    img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

    dataset = []

    # 이미지 id 별로 annotation 모으기
    img_id_to_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, w, h]
        category_id = ann['category_id']

        # xywh -> xyxy 변환
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = x1 + bbox[2]
        y2 = y1 + bbox[3]

        if img_id not in img_id_to_annotations:
            img_id_to_annotations[img_id] = []

        img_id_to_annotations[img_id].append([x1, y1, x2, y2, category_id])

    for img_id, boxes in img_id_to_annotations.items():
        img_path = os.path.join(images_dir, img_id_to_filename[img_id])
        dataset.append((img_path, boxes))

    return dataset

# 데이터셋 로드
images_dir = "datasets/val2017"
annotation_json_path = "datasets/annotations/instances_val2017.json"
dataset = load_coco_dataset(images_dir, annotation_json_path)

model_names = ['yolov5nu', 'yolov5su', 'yolov8n', 'yolov8s', 'yolov10n', 'yolov10s']

for model_name in model_names:
    model = YOLOModel(model_name)
    model.load()
    model.export(overwrite=True)
    model.convert(overwrite=True)

    evaluator = Evaluator(model, dataset)
    evaluator.benchmark()

    model = OVDetectionModel(f'models/xml/{model_name}.xml')
    model.load()
    evaluator = Evaluator(model, dataset)
    evaluator.benchmark()


# # PyTorch 모델
# pt_model = YOLOModel("yolov8n")
# pt_model.load()
# evaluator_pt = Evaluator(pt_model, dataset)
# evaluator_pt.benchmark()
# evaluator_pt.evaluate_map()

# # OpenVINO 모델
# ov_model = OVDetectionModel("models/xml/yolov8n.xml")
# ov_model.load()
# evaluator_ov = Evaluator(ov_model, dataset)
# evaluator_ov.benchmark()
# eevaluator_ov.evaluate_map()
