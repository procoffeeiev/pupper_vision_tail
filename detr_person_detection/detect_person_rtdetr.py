import argparse
from pathlib import Path

import cv2
from ultralytics import RTDETR


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "rtdetr-l.pt"

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5


def draw_label(image, text, x1, y1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    padding = 6

    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y1 = max(y1 - text_height - baseline - 2 * padding, 0)
    label_y2 = label_y1 + text_height + baseline + 2 * padding
    label_x2 = x1 + text_width + 2 * padding

    cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), (30, 120, 255), -1)
    cv2.putText(
        image,
        text,
        (x1 + padding, label_y2 - baseline - padding),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser(description="Detect people in an image using RT-DETR.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "pupper_camera_frame.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <input_stem>_rtdetr_person_detection.jpg.",
    )
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = PROJECT_DIR / input_path

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_DIR / f"{input_path.stem}_rtdetr_person_detection.jpg"
    elif not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path

    if not input_path.exists():
        raise SystemExit(f"Could not find input image: {input_path}")

    image = cv2.imread(str(input_path))
    if image is None:
        raise SystemExit(f"Could not read input image: {input_path}")

    model = RTDETR(str(MODEL_PATH))
    results = model.predict(str(input_path), conf=args.confidence, verbose=False)

    people = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id != PERSON_CLASS_ID or confidence < args.confidence:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            people.append((confidence, x1, y1, x2, y2))

    people.sort(reverse=True)

    for i, (confidence, x1, y1, x2, y2) in enumerate(people, start=1):
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 120, 255), 3)
        draw_label(image, f"RT-DETR person {confidence:.2f}", x1, y1)

        box_width = x2 - x1
        box_height = y2 - y1
        center_x = x1 + box_width / 2
        center_y = y1 + box_height / 2
        area_ratio = (box_width * box_height) / (image.shape[1] * image.shape[0])
        horizontal_error = (center_x - image.shape[1] / 2) / (image.shape[1] / 2)

        print(
            f"Person {i}: confidence={confidence:.2f}, "
            f"box=({x1}, {y1}, {x2}, {y2}), "
            f"center=({center_x:.1f}, {center_y:.1f}), "
            f"horizontal_error={horizontal_error:.3f}, "
            f"area_ratio={area_ratio:.3f}"
        )

    cv2.imwrite(str(output_path), image)
    print(f"People detected: {len(people)}")
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
