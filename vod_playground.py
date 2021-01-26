import argparse
import pytube
import pytube.cli
import cv2
import tqdm
import subprocess
import pickle
import itertools
from pathlib import Path
import numpy as np

import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask, _create_text_labels
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes, pairwise_iou


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
    frame_list = sorted(frames_dir.glob(f"frame_*.png"))
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
    print(f"Loading proposals from {proposals_path}")

    with proposals_path.open("rb") as f_in:
        proposals = pickle.load(f_in)

    proposals = restrict_predictions(cfg, proposals, {args.class_to_detect})

    frame = cv2.imread(str(frame_path))
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(proposals["instances"])

    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


class Tubelet:
    def __init__(self, start_index, proposal_instance, iou_threshold=0.5):
        self.frame_ids = [start_index]
        self.proposal_instances = [proposal_instance]
        self.iou_threshold = iou_threshold

    def extend(self, frame_index, proposal_instances):
        for i in range(len(proposal_instances)):
            proposal_instance = proposal_instances[i]
            last_index = self.frame_ids[-1]
            if frame_index - last_index > 1:
                # This tubelet died at least one frame ago
                return
            if last_index == frame_index:
                # We already have a candidate for a continuation of this tubelet
                # See if this one fits as well and then decide for the one with the highest score
                cand_instance = self.proposal_instances[-1]
                last_instance = self.proposal_instances[-2]
                iou = pairwise_iou(last_instance.pred_boxes, proposal_instance.pred_boxes)
                if iou > self.iou_threshold and proposal_instance.scores > cand_instance.scores:
                    self.proposal_instances[-1] = proposal_instance
            else:
                # This is the first candidate to consider for extending the tubelet
                last_instance = self.proposal_instances[-1]
                iou = pairwise_iou(last_instance.pred_boxes, proposal_instance.pred_boxes)
                if iou > self.iou_threshold:
                    self.frame_ids.append(frame_index)
                    self.proposal_instances.append(proposal_instance)

    def collides_with(self, frame_index, proposal_instance):
        last_index = self.frame_ids[-1]
        if last_index != frame_index:
            return False
        last_instance = self.proposal_instances[-1]
        iou = pairwise_iou(last_instance.pred_boxes, proposal_instance.pred_boxes)
        if iou > self.iou_threshold:
            return True

    def __len__(self):
        return len(self.frame_ids)

    def is_active(self, frame_id):
        return self.frame_ids[-1] == frame_id

    def get_instance(self, frame_id):
        if frame_id not in self.frame_ids:
            return None
        else:
            i = self.frame_ids.index(frame_id)
            return self.proposal_instances[i]



def generate_tubelets(args, proposals_dict, threshold=0.7):
    if args.method == "none":
        return proposals_dict
    elif args.method == "default":
        return proposals_dict
    elif args.method == "custom":
        tubelets = []
        with tqdm.tqdm(total=len(proposals_dict)) as pbar:
            for i, frame_path in enumerate(sorted(proposals_dict.keys())):
                pbar.update(1)
                #print(f"Frame #{i}: {frame_path}")
                proposal_instances = proposals_dict[frame_path]["instances"]
                for tubelet in tubelets:
                    tubelet.extend(i, proposal_instances)
                key_proposal_indices = torch.nonzero(proposal_instances.scores > threshold)
                #print(f"Frame #{i}: Found {key_proposal_indices.shape[0]} key proposal indices.")
                for key_proposal_index in key_proposal_indices:
                    key_proposal_instance = proposal_instances[key_proposal_index]
                    if not any(t.collides_with(i, key_proposal_instance) for t in tubelets):
                        tubelets.append(Tubelet(i, key_proposal_instance))
        print(f"Tubelet statistics:")
        print(f"    - Overall: {len(tubelets)}, avergae length: {sum(len(t) for t in tubelets) / len(tubelets) if tubelets else 0}")
        return tubelets


def draw_instance_predictions(visualizer, instances):
    def get_color(i):
        colors = "bgrcmykw"
        return colors[i % len(colors)]

    tubelet_ids = [i[0] for i in instances]
    predictions = Instances.cat([i[1] for i in instances])

    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, scores, visualizer.metadata.get("thing_classes", None))
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, visualizer.output.height, visualizer.output.width) for x in masks]
    else:
        masks = None

    colors = [get_color(i) for i in tubelet_ids]
    visualizer.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=0.5,
    )
    return visualizer.output



def render_video(args):
    cfg, _ = load_model(config_only=True)

    proposals_dir = args.working_dir / f"proposals_{args.video_id}"
    frames_dir = args.working_dir / f"frames_{args.video_id}"
    rendered_frames_dir = args.working_dir / f"rendered_frames_{args.video_id}__{args.method}"

    if not rendered_frames_dir.exists():
        rendered_frames_dir.mkdir()

    frame_list = sorted(frames_dir.glob("frame_*.png"))
    print(f"Loading {len(frame_list)} frames and proposals")
    proposals_dict = {}
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        for i, frame_path in enumerate(frame_list):
            pbar.update(1)
            frame_id = int(frame_path.name[frame_path.name.find("_")+1:frame_path.name.find(".")])
            if frame_id < 1500 or frame_id > 1900:
                continue

            proposals_path = proposals_dir / frame_path.with_suffix(".pickle").name
            if not proposals_path.exists():
                continue
            with proposals_path.open("rb") as f_in:
                proposals = pickle.load(f_in)
            proposals_dict[frame_path] = restrict_predictions(cfg, proposals, {args.class_to_detect})

    print(f"Generating tubelets for {args.video_id}")
    tubelets = generate_tubelets(args, proposals_dict)

    print(f"Rendering frames for {args.video_id}.")
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        for i, frame_path in enumerate(frame_list):
            pbar.update(1)
            frame_id = int(frame_path.name[frame_path.name.find("_")+1:frame_path.name.find(".")])
            if frame_id < 1500 or frame_id > 1900:
                continue

            rendered_frame_path = rendered_frames_dir / frame_path.name
            if rendered_frame_path.exists():
                continue

            #proposals = proposals_dict[frame_path]
            instances = []
            for tubelet_id, tubelet in enumerate(tubelets):
                instance = tubelet.get_instance(i - 1500)
                if instance is not None:
                    instances.append((tubelet_id, instance))
            print(f"Frame #{i}: {len(instances)} instances")

            frame = cv2.imread(str(frame_path))
            if instances:
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                v = draw_instance_predictions(v, instances)
                cv2.imwrite(str(rendered_frame_path), v.get_image()[:, :, ::-1])
            else:
                cv2.imwrite(str(rendered_frame_path), frame)


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


def assemble_result(args):
    source_frames_dir = args.working_dir / f"rendered_frames_{args.video_id}"
    subprocess.run([
        "ffmpeg", "-r", "25", "-start_number", "1500", "-i",
        str(source_frames_dir / "frame_%04d.png"),
        "-y", str(args.working_dir / f"video_{args.video_id}.mp4")
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="One of: preprocess_video, ...")
    parser.add_argument("--working_dir", default="/tmp/video-object-detection/")
    parser.add_argument("--video_id", default="eKKdRy20HXI")
    parser.add_argument("--class_to_detect", default="car")
    parser.add_argument("--frame_id", default=None)
    parser.add_argument("--method", default="none", help="One of: none, default, custom")
    args = parser.parse_args()

    args.working_dir = Path(args.working_dir)

    if args.action == "preprocess_video":
        preprocess_video(args)
    elif args.action == "display_frame":
        display_frame(args)
    elif args.action == "render_video":
        render_video(args)
    elif args.action == "detect_objects":
        detect_objects(args)
    elif args.action == "assemble_result":
        assemble_result(args)
    else:
        raise NotImplementedError(f"Unknown action: {args.action}")