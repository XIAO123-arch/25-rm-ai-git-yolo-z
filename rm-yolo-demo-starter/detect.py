from argparse import ArgumentParser
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = ArgumentParser(description="Minimal YOLOv8 demo detection starter for RM algorithm tasks.")
    parser.add_argument("--source", required=True, help="Path to an input image. Video is a student task.")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLOv8 model weights.")
    parser.add_argument("--output", default="outputs/result.jpg", help="Path to save the annotated image.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--show", action="store_true", help="Show result in an OpenCV window.")
    return parser.parse_args()


def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if class_id < len(names):
        return names[class_id]
    return str(class_id)


def resolve_input_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def resolve_output_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def draw_detections(image, result):
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = get_class_name(result.names, class_id)

        label = f"{class_name} {conf:.2f}"
        color = (255, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA,
        )


def run_image(source, model_path, output, conf, show=False):
    # 1.读取图片输入
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    # 2.YOLO模型推理
    model = YOLO(str(model_path))
    results = model.predict(image, conf=conf, verbose=False)

    annotated = image.copy()
    if results:
        # 3.检测画框
        draw_detections(annotated, results[0])

    output.parent.mkdir(parents=True, exist_ok=True)
    # 4.图片结果保存
    if not cv2.imwrite(str(output), annotated):
        raise RuntimeError(f"Failed to write result: {output}")

    if show:
        cv2.imshow("RM YOLO Demo Starter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Saved result to {output}")


def main():
    args = parse_args()
    source = resolve_input_path(args.source)
    model_path = resolve_input_path(args.model)
    output = resolve_output_path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        run_image(source, model_path, output, args.conf, args.show)
        return

    if suffix in VIDEO_EXTENSIONS:
        raise NotImplementedError(
            "Video input is intentionally left as a student task. "
            "Implement cv2.VideoCapture and cv2.VideoWriter here."
        )

    raise ValueError(f"Unsupported source type: {source}")


if __name__ == "__main__":
    main()
