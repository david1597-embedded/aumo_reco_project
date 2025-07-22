from inference import InferenceEngine
from evaluator import Evaluator

if __name__ == "__main__":
    model_name = 'yolov5nu'
    engine = InferenceEngine(
        model_path=f"models/{model_name}.pt",
        backend="yolo",
        device="cpu"  # "cuda" 또는 "cpu"
    )
    evaluator = Evaluator(
        engine=engine,
        coco_json="datasets/coco/annotations/instances_val2017.json",
        img_dir="datasets/coco/val2017",
        runs=1
    )
    # evaluator.benchmark(10)
    evaluator.run()

    engine = InferenceEngine(
        model_path=f"models/xml/{model_name}.xml",
        backend="openvino",
        device="cpu"
    )
    evaluator = Evaluator(
        engine=engine,
        coco_json="datasets/coco/annotations/instances_val2017.json",
        img_dir="datasets/coco/val2017",
        runs=1
    )
    # evaluator.benchmark(10)
    evaluator.run()
