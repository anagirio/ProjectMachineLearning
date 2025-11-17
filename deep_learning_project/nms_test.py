import argparse
import os
import time
from detector import FaceDetector, non_max_suppression
from detect_faces import draw_boxes


def run_nms_test(image_path, model_path='best_model_augmented.pth',
                 confidence=0.5, iou=0.3, window_size=64, stride=16,
                 output_path=None):
    """Run detector + NMS on a single image, print detections and save visualization.

    Args:
        image_path: path to input image
        model_path: path to model file
        confidence: detector confidence threshold
        iou: IoU threshold for NMS
        window_size, stride: sliding-window params forwarded to detector
        output_path: where to save visualization (defaults to detected_<filename>)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    print(f'Loading model from {model_path} (confidence={confidence})...')
    detector = FaceDetector(model_path, confidence)
    t0 = time.time()
    detections = detector.detect(image_path, window_size=window_size, stride=stride)
    print(f'Raw detections ({len(detections)}):')
    for d in detections:
        print('  ', d)

    detections_nms = non_max_suppression(detections, iou)
    print(f'After NMS ({len(detections_nms)}):')
    for d in detections_nms:
        print('  ', d)

    elapsed = time.time() - t0
    print(f'Processing time: {elapsed:.3f}s')

    if output_path is None:
        base = os.path.basename(image_path)
        output_path = os.path.join(os.path.dirname(image_path), f'detected_{base}')

    draw_boxes(image_path, detections_nms, output_path)
    print(f'Visualization saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Quick NMS test on one image')
    parser.add_argument('--image', '-i', default='test.jpg', required=True, help='Path to the image to test')
    parser.add_argument('--model', '-m', default='best_model_augmented.pth', help='Path to model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Detector confidence threshold')
    parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold for NMS')
    parser.add_argument('--window-size', type=int, default=64, help='Detection window size')
    parser.add_argument('--stride', type=int, default=16, help='Sliding window stride')
    parser.add_argument('--output', '-o', help='Output path for visualization')
    args = parser.parse_args()

    run_nms_test(args.image, model_path=args.model, confidence=args.confidence,
                 iou=args.iou, window_size=args.window_size, stride=args.stride,
                 output_path=args.output)


if __name__ == '__main__':
    main()
