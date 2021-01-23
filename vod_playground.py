import argparse
import pytube
import pytube.cli
import cv2
import tqdm
import subprocess
import pickle
from pathlib import Path

import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes


def preprocess_video(args):
    yt_url = f"https://www.youtube.com/watch?v={args.video_id}"
    filename = f"video_{args.video_id}"
    video_file_path = args.working_dir / f"{filename}.mp4"

    # Download video
    if video_file_path.exists():
        print(f"Video {video_file_path} already downloaded.")
    else:
        print(f"Downloading {yt_url} to {video_file_path}")
        video = pytube.YouTube(yt_url, on_progress_callback=pytube.cli.on_progress)
        video.streams.filter(res="720p").first().download(output_path=args.working_dir, filename=filename)
        print()

    # Extract frames
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

    # Generate frame-based object proposals
    proposals_dir = args.working_dir / f"proposals_{args.video_id}"
    print(f"Extracting proposals for {args.video_id} to {frames_dir}")
    cfg, model = load_model()

    if not proposals_dir.exists():
        proposals_dir.mkdir(parents=False, exist_ok=False)
    frame_list = list(frames_dir.glob(f"frame_*.png"))
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        for frame_path in frame_list:
            pbar.update(1)
            proposal_path = proposals_dir / frame_path.with_suffix(".pickle").name
            if proposal_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            proposals = generate_poposals(frame, model)
            with proposal_path.open("wb") as f_out:
                pickle.dump(proposals, f_out)


def display_frame(args):
    if args.frame_id is None:
        print("Please specify a frame to visualize")
        return

    cfg, _ = load_model(config_only=True)

    proposals_dir = args.working_dir / f"proposals_{args.video_id}"
    proposals_path = proposals_dir / f"frame_{int(args.frame_id):04d}.pickle"

    frames_dir = args.working_dir / f"frames_{args.video_id}"
    frame_path = frames_dir / f"frame_{int(args.frame_id):04d}.png"

    print(f"Visualizing proposals for {args.video_id}.")
    print(f"Loading prpopsals from {proposals_path}")

    with proposals_path.open("rb") as f_in:
        proposals = pickle.load(f_in)

    #proposals = restrict_predictions(cfg, proposals, {args.class_to_detect})

    frame = cv2.imread(str(frame_path))
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(proposals["instances"])

    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


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


def inference_image(image, model, score_threshold=0.01):
    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        proposals_scores, proposals_deltas = model.roi_heads.box_predictor(box_features)
        proposal_class_predictions = proposals_scores.softmax(-1)

        boxes = model.roi_heads.box_predictor.predict_boxes((proposals_scores, proposals_deltas), proposals)
        scores = model.roi_heads.box_predictor.predict_probs((proposals_scores, proposals_deltas), proposals)

        scores = scores[0][:, :-1]

        image_size = proposals[0].image_size
        num_bbox_reg_classes = boxes[0].shape[1] // 4
        boxes = Boxes(boxes[0].reshape(-1, 4))
        boxes.clip(image_size)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)

        filter_mask = scores > score_threshold
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]

        pred_instances, pred_inds = model.roi_heads.box_predictor.inference((proposals_scores, proposals_deltas), proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
        surviving_boxes = model._postprocess(pred_instances, inputs, images.image_sizes)
        surviving_features = box_features[pred_inds]
        surviving_class_predictions = proposal_class_predictions[pred_inds]

    return {"instances": result}, proposal_class_predictions, surviving_boxes[0], surviving_class_predictions


def generate_poposals(image, model, score_threshold=0.01):
    height, width = image.shape[:2]
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)
        proposals_scores, proposals_deltas = model.roi_heads.box_predictor(box_features)
        proposal_class_predictions = proposals_scores.softmax(-1)

        boxes = model.roi_heads.box_predictor.predict_boxes((proposals_scores, proposals_deltas), proposals)
        scores = model.roi_heads.box_predictor.predict_probs((proposals_scores, proposals_deltas), proposals)

        scores = scores[0][:, :-1]

        image_size = proposals[0].image_size
        num_bbox_reg_classes = boxes[0].shape[1] // 4
        boxes = Boxes(boxes[0].reshape(-1, 4))
        boxes.clip(image_size)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)

        filter_mask = scores > score_threshold
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]

    return {"instances": result, "proposal_class_predictions": proposal_class_predictions}


def load_model(config_only=False):
    cfg = get_cfg()
    cfg.merge_from_file(Path(detectron2.__file__).resolve().parent / "model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.DEVICE = "cpu"

    if config_only:
        return cfg, None

    print("Loading model.")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor.model


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
    if args.frame_id is None:
        frame_list = sorted(source_frames_dir.glob("frame_*.png"))
    else:
        frame_list = [source_frames_dir / f"frame_{args.frame_id}.png"]
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        for frame_path in frame_list:
            pbar.update(1)
            output_frame_path = target_frames_dir / frame_path.with_suffix(".json").name
            if output_frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            proposal_boxes, proposal_class_predictions, surviving_boxes, surviving_class_predictions = inference_image(frame, predictor.model)

            proposal_boxes = restrict_predictions(cfg, proposal_boxes, {args.class_to_detect})

            v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            v = v.draw_instance_predictions(proposal_boxes["instances"])

            cv2.imshow("image", v.get_image()[:, :, ::-1])
            cv2.waitKey(0)




def detect_objects_old(args):
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
    if args.frame_id is None:
        frame_list = sorted(source_frames_dir.glob("frame_*.png"))
    else:
        frame_list = [source_frames_dir / f"frame_{args.frame_id}.png"]
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

            cv2.imshow("image", v.get_image()[:, :, ::-1])
            cv2.waitKey(0)

            #cv2.imwrite(str(output_frame_path), v.get_image()[:, :, ::-1])
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
    parser.add_argument("--working_dir", default="/tmp/video-object-detection/")
    parser.add_argument("--video_id", default="eKKdRy20HXI")
    parser.add_argument("--class_to_detect", default="car")
    parser.add_argument("--frame_id", default=None)
    args = parser.parse_args()

    args.working_dir = Path(args.working_dir)

    if args.action == "preprocess_video":
        preprocess_video(args)
    elif args.action == "display":
        display_frame(args)
    elif args.action == "detect_objects":
        detect_objects(args)
    elif args.action == "assemble_result":
        assemble_result(args)
    else:
        raise NotImplementedError(f"Unknown action: {args.action}")