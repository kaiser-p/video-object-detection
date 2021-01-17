import argparse
import pytube
import pytube.cli
import cv2
import tqdm
import subprocess
from pathlib import Path

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances


def preprocess_video(args):
    yt_url = f"https://www.youtube.com/watch?v={args.video_id}"
    filename = f"video_{args.video_id}"
    video_file_path = args.working_dir / f"{filename}.mp4"

    if video_file_path.exists():
        print(f"Video {video_file_path} already downloaded.")
    else:
        print(f"Downloading {yt_url} to {video_file_path}")
        video = pytube.YouTube(yt_url, on_progress_callback=pytube.cli.on_progress)
        video.streams.filter(res="720p").first().download(output_path=args.working_dir, filename=filename)
        print()

    frames_dir = args.working_dir / f"frames_{args.video_id}"
    if frames_dir.exists():
        print(f"Frames for {args.video_id} already extracted.")
    else:
        print(f"Extracting frames for {args.video_id} to {frames_dir}")
        frames_dir.mkdir(parents=False, exist_ok=False)
        vidcap = cv2.VideoCapture(str(video_file_path))
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        with tqdm.tqdm(total=num_frames) as pbar:
            while (frame_info := vidcap.read())[0]:
                frame_file_path = frames_dir / f"frame_{count:04d}.png"
                cv2.imwrite(str(frame_file_path), frame_info[1])
                count += 1
                pbar.update(1)
        print(f"Extracted {count+1} frames.")


def restrict_predictions(cfg, predictions, allowed_classes=None):
    if allowed_classes is None:
        return predictions

    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    allowed_class_indices = {classes.index(c) for c in allowed_classes}
    instances_to_keep = [i for i,c in enumerate(predictions["instances"].pred_classes.numpy()) if c in allowed_class_indices]

    return {"instances": Instances(
        image_size=predictions["instances"].image_size,
        pred_boxes=predictions["instances"].pred_boxes[instances_to_keep],
        scores=predictions["instances"].scores[instances_to_keep],
        pred_classes=predictions["instances"].pred_classes[instances_to_keep]
    )}


def detect_objects(args):
    source_frames_dir = args.working_dir / f"frames_{args.video_id}"
    target_frames_dir = args.working_dir / f"frames_{args.video_id}__default"

    target_frames_dir.mkdir(exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(Path(detectron2.__file__).resolve().parent / "model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.DEVICE = "cpu"

    print("Loading model.")
    predictor = DefaultPredictor(cfg)

    print("Starting to detect objects in frames using the default approach.")
    frame_list = sorted(source_frames_dir.glob("frame_*.png"))
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        for frame_path in frame_list:
            pbar.update(1)
            output_frame_path = target_frames_dir / frame_path.name
            if output_frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))

            predictions = predictor(frame)
            predictions = restrict_predictions(cfg, predictions, {args.class_to_detect})

            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(predictions["instances"])

            cv2.imwrite(str(output_frame_path), v.get_image()[:, :, ::-1])
    print("All frames processed.")


def assemble_result(args):
    source_frames_dir = args.working_dir / f"frames_{args.video_id}__default"
    subprocess.run([
        "ffmpeg", "-r", "25", "-i",
        str(source_frames_dir / "frame_%04d.png"),
        "-y", str(args.working_dir / f"video_{args.video_id}__default.mp4")
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="One of: preprocess_video, ...")
    parser.add_argument("--working_dir", default=Path("/tmp/video-object-detection/"))
    parser.add_argument("--video_id", default="eKKdRy20HXI")
    parser.add_argument("--class_to_detect", default="car")
    args = parser.parse_args()

    if args.action == "preprocess_video":
        preprocess_video(args)
    elif args.action == "detect_objects":
        detect_objects(args)
    elif args.action == "assemble_result":
        assemble_result(args)
    else:
        raise NotImplementedError(f"Unknown action: {args.action}")