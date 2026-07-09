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
    parser.add_argument("--output", default="outputs", help="Folder to save output files.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--show", action="store_true", help="Show result in an OpenCV window.")
    parser.add_argument('--max-det',type=int,default=10,help='Maximum number of objects to keep after sorting by confidence')
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
import torch

def filter_boxes(results, max_det):
    boxes = results[0].boxes
    if len(boxes) == 0:
        return results
    confs = boxes.conf
    sort_idx = torch.sort(confs, descending=True).indices
    sorted_boxes = boxes[sort_idx]
    if len(sorted_boxes) > max_det:
        sorted_boxes = sorted_boxes[:max_det]
    results[0].boxes = sorted_boxes
    return results
def print_statistics(results):
    final_boxes = results[0].boxes
    total_num = len(final_boxes)
    print(f"\n=====检测统计信息=====")
    print(f"总共保留目标数量: {total_num}")
    class_stat = {}
    for box in final_boxes:
        cid = int(box.cls[0])
        cname = get_class_name(results[0].names, cid)
        class_stat[cname] = class_stat.get(cname, 0)+1
    for cls_name, count in class_stat.items():
        print(f"类别 {cls_name}：{count} 个")
    print(f"=====================\n")
def resolve_output_dir(path):
    out_path = Path(path)
    if out_path.is_absolute():
        return out_path
    return PROJECT_ROOT / out_path
def run_video(source, model_path, out_dir, conf, max_det):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / source.name
    writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    model = YOLO(str(model_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, verbose=False)
        results = filter_boxes(results, max_det)
        if results:
            draw_detections(frame, results[0])
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"Video saved to {output_file}")
def run_folder(folder_path, model_path, out_dir, conf, max_det):
    img_paths = list(folder_path.glob("*"))
    for img_file in img_paths:
        if img_file.suffix.lower() in IMAGE_EXTENSIONS:
            run_image(img_file, model_path, out_dir, conf, max_det)


def run_image(source, model_path, out_dir, conf, max_det, show=False):
    # 1.读取图片输入
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
        # 2.YOLO模型推理
    model = YOLO(str(model_path))
    results = model.predict(image, conf=conf, verbose=False)
        # 取出检测框，按置信度从高到低排序
    # 改动2：调用已定义的filter_boxes函数，复用过滤逻辑
    if results:
        results = filter_boxes(results, max_det)
        print_statistics(results)
        annotated = image.copy()
        if results:
            # 3.检测画框
            draw_detections(annotated, results[0])

        #改动4：out_dir文件夹 + 原图名称生成输出文件，批量处理不会覆盖文件
        source_file = Path(source)
        output = Path(out_dir) / source_file.name
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # 保存图片
        if not cv2.imwrite(str(output), annotated):
            raise RuntimeError(f"Failed to write result: {output}")

    if show:
        cv2.imshow("RM YOLO Demo Starter", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #改动3：调用统计函数输出信息
        print_statistics(results)
    print(f"Saved result to {output}")


def main():
    args = parse_args()
    source = resolve_input_path(args.source)
    model_path = resolve_input_path(args.model)
    out_dir = resolve_output_path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    # ============第6点改动：新增3分支判断============
    if source.is_dir():
        # 分支1：文件夹批量模式
        run_folder(source, model_path, out_dir, args.conf, args.max_det)
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        # 分支2：单张图片模式
        run_image(source, model_path, out_dir, args.conf, args.max_det, args.show)
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        # 分支3：视频文件模式
        run_video(source, model_path, out_dir, args.conf, args.max_det)
    else:
        raise ValueError(f"Unsupported source type: {source}")

if __name__ == "__main__":
    main()
