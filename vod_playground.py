import argparse
import pytube
import pytube.cli
import cv2
import tqdm
import subprocess
import pickle
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask, _create_text_labels
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes, pairwise_iou, pairwise_intersection
from torchvision.ops import nms


def preprocess_video(args, batch_size=1):
    yt_url = f"https://www.youtube.com/watch?v={args.video_id}"
    filename = f"video"
    video_working_dir = args.working_dir / args.video_id
    video_file_path = video_working_dir / f"video.mp4"

    # Download video
    if video_file_path.exists():
        print(f"Video {video_file_path} already downloaded.")
    else:
        print(f"Downloading {yt_url} to {video_file_path}")
        video = pytube.YouTube(yt_url, on_progress_callback=pytube.cli.on_progress)
        video.streams.filter(res="720p").first().download(output_path=video_working_dir, filename=filename)
        print()

    # Extract frames
    frames_dir = video_working_dir / f"frames"
    if frames_dir.exists():
        print(f"Frames for {args.video_id} already extracted.")
    else:
        print(f"Extracting frames for {args.video_id} to {frames_dir}")
        frames_dir.mkdir(parents=True, exist_ok=False)
        vidcap = cv2.VideoCapture(str(video_file_path))
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        with tqdm.tqdm(total=num_frames) as pbar:
            while (frame_info := vidcap.read())[0]:
                frame_file_path = frames_dir / f"frame_{count:04d}.jpg"
                cv2.imwrite(str(frame_file_path), frame_info[1])
                count += 1
                pbar.update(1)
        print(f"Extracted {count+1} frames.")

    # Generate frame-based object proposals
    proposals_dir = video_working_dir / f"proposals"
    print(f"Extracting proposals for {args.video_id} to {frames_dir}")
    cfg, model = load_model()

    if not proposals_dir.exists():
        proposals_dir.mkdir(parents=False, exist_ok=False)
    frame_list = sorted(frames_dir.glob(f"frame_*.jpg"))
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        batch_images = []
        batch_paths = []
        for frame_path in frame_list:
            pbar.update(1)

            frame_id = int(frame_path.name[frame_path.name.find("_")+1:frame_path.name.find(".")])
            if args.frame_start is not None and frame_id < args.frame_start:
                continue
            if args.frame_stop is not None and frame_id > args.frame_stop:
                continue

            proposal_path = proposals_dir / frame_path.with_suffix(".pickle").name
            if proposal_path.exists():
                continue

            batch_images.append(cv2.imread(str(frame_path)))
            batch_paths.append(proposal_path)

            if len(batch_images) == batch_size or frame_path == frame_list[-1]:
                batch_proposals = generate_poposals(batch_images, model)
                for proposals, proposal_path in zip(batch_proposals, batch_paths):
                    batch_images.clear()
                    batch_paths.clear()
                    with proposal_path.open("wb") as f_out:
                        pickle.dump(proposals, f_out)


class Tubelet:
    def __init__(self, start_index, proposal_instance, iou_threshold=0.7, num_skippable_frames=7, extend_class_only=False, class_to_detect=0, max_dimension_change_ratio=None, max_dimension_change_abs=None, perform_projection=True):
        proposal_instance.generation_process = ["K"]

        self.frame_ids = [start_index]
        self.last_key_instance_index = [0]
        self.proposal_instances = [proposal_instance]
        self.projected_proposal_instances = [proposal_instance]
        self.proposal_instances_ious = [0]
        self.iou_threshold = iou_threshold
        self.num_skippable_frames = num_skippable_frames
        self.extend_class_only = extend_class_only
        self.class_to_detect = class_to_detect
        self.max_dimension_change_ratio = max_dimension_change_ratio
        self.max_dimension_change_abs = max_dimension_change_abs
        self.perform_projection = perform_projection

    def project_proposal_instance(self, frame_index):
        if len(self.proposal_instances) == 1:
            # We consider the first frame, there is nothing to project here
            return self.proposal_instances[0]

        instance_index_current = self.last_key_instance_index[-1]
        instance_index_before = self.last_key_instance_index[-2]
        frame_index_current = self.frame_ids[instance_index_current]

        assert frame_index > frame_index_current
        if frame_index == frame_index_current:
            # We are replacing the most recent proposal instance
            if len(self.proposal_instances) < 3:
                instance_index_current = self.last_key_instance_index[-2]
                return self.proposal_instances[instance_index_current]
            else:
                instance_index_current = self.last_key_instance_index[-2]
                instance_index_before = self.last_key_instance_index[-3]
                frame_index_current = self.frame_ids[instance_index_current]

        instance_current = self.proposal_instances[instance_index_current]
        instance_before = self.proposal_instances[instance_index_before]

        centers_current = instance_current.pred_boxes.get_centers()
        centers_before = instance_before.pred_boxes.get_centers()
        centers_delta = (centers_current - centers_before) / (instance_index_current - instance_index_before) * (frame_index - frame_index_current)

        projected_instance = Instances(instance_current.image_size)
        projected_instance.pred_boxes = Boxes(instance_current.pred_boxes.tensor + centers_delta.repeat(1, 2))
        projected_instance.scores = instance_current.scores
        projected_instance.pred_classes = instance_current.pred_classes
        projected_instance.class_distributions = instance_current.class_distributions
        projected_instance.generation_process = ["P"]
        return projected_instance

    def is_dead(self, frame_index):
        last_index = self.frame_ids[-1]
        return frame_index - last_index > self.num_skippable_frames

    def extend_after(self, frame_index, proposal_instances):
        if self.is_dead(frame_index):
            return False

        last_key_instance_index = self.last_key_instance_index[-1]
        if self.perform_projection:
            projected_last_instance = self.project_proposal_instance(frame_index)
        elif self.perform_projection:
            projected_last_instance = self.proposal_instances[last_key_instance_index]

        projection_dim = projected_last_instance.pred_boxes.tensor[:, 2:] - projected_last_instance.pred_boxes.tensor[:, :2]
        proposals_dim = proposal_instances.pred_boxes.tensor[:, 2:] - proposal_instances.pred_boxes.tensor[:, :2]

        if self.max_dimension_change_ratio is not None:
            dim_ratios = proposals_dim / projection_dim
            selection_mask = torch.all((dim_ratios > 1 - self.max_dimension_change_ratio) & (dim_ratios < 1 + self.max_dimension_change_ratio), dim=1)
            proposal_instances = proposal_instances[selection_mask]

        if self.max_dimension_change_abs is not None:
            dim_diffs = torch.abs(proposals_dim - projection_dim)
            selection_mask = torch.all(dim_diffs < self.max_dimension_change_abs, dim=1)
            proposal_instances = proposal_instances[selection_mask]

        if len(proposal_instances) == 0:
            return False

        iou_per_proposal = pairwise_iou(proposal_instances.pred_boxes, projected_last_instance.pred_boxes)

        if self.extend_class_only:
            # Ignore all proposals of different classes
            class_proposals = (proposal_instances.pred_classes != self.class_to_detect)
            iou_per_proposal[class_proposals] = 0

        extension_candidate_proposal_index = int(iou_per_proposal.argmax())

        last_index = self.frame_ids[-1]
        if iou_per_proposal[extension_candidate_proposal_index] < self.iou_threshold:
            # This tubelet will not be extended in this frame
            return False

        for j in range(last_index + 1, frame_index):
            self.frame_ids.append(j)
            self.proposal_instances.append(None)
            self.proposal_instances_ious.append(0)
            self.last_key_instance_index.append(last_key_instance_index)
            self.projected_proposal_instances.append(projected_last_instance)

        pi = proposal_instances[extension_candidate_proposal_index]
        pi.generation_process = ["T+"] if frame_index > 0 else ["T-"]

        self.frame_ids.append(frame_index)
        self.proposal_instances.append(pi)
        self.proposal_instances_ious.append(iou_per_proposal[extension_candidate_proposal_index])
        self.last_key_instance_index.append(len(self.last_key_instance_index))
        self.projected_proposal_instances.append(projected_last_instance)
        return True

    def extend_before(self, frame_index, proposal_instances):
        def restore_last_key_instance_index(proposal_instances):
            result = []
            last_key_index = 0
            for i in range(len(proposal_instances)):
                if proposal_instances[i] is not None:
                    last_key_index = i
                result.append(last_key_index)
            return result

        if frame_index >= self.frame_ids[0]:
            return

        # We simply turn around the time and use extend_after
        self.proposal_instances.reverse()
        self.proposal_instances_ious.reverse()
        self.projected_proposal_instances.reverse()
        self.frame_ids = list(-fid for fid in reversed(self.frame_ids))
        self.last_key_instance_index = restore_last_key_instance_index(self.proposal_instances)

        was_extended = self.extend_after(-frame_index, proposal_instances)

        self.proposal_instances.reverse()
        self.proposal_instances_ious.reverse()
        self.projected_proposal_instances.reverse()
        self.frame_ids = list(-fid for fid in reversed(self.frame_ids))
        self.last_key_instance_index = restore_last_key_instance_index(self.proposal_instances)
        return was_extended


    def collides_with(self, frame_id, proposal_instance, use_iou=False):
        """last_index = self.frame_ids[-1]
        if last_index != frame_index:
            return False
        last_instance = self.proposal_instances[-1]"""

        tubelet_instance = self.get_instance(frame_id)
        if tubelet_instance is None:
            return torch.zeros(len(proposal_instances), dtype=torch.bool)
        if use_iou:
            iou = pairwise_iou(tubelet_instance.pred_boxes, proposal_instance.pred_boxes)
            return iou > self.iou_threshold
        else:
            last_instance_box = Boxes(tubelet_instance.pred_boxes.tensor)
            proposal_instance_box = Boxes(proposal_instance.pred_boxes.tensor)
            last_instance_area = last_instance_box.area()
            proposal_instance_area = proposal_instance_box.area()
            intersection = pairwise_intersection(tubelet_instance.pred_boxes, proposal_instance.pred_boxes)
            return intersection / last_instance_area > 0.8 or intersection / proposal_instance_area > 0.8

    def collides_with_mask(self, frame_id, proposal_instances, use_iou=False):
        """last_index = self.frame_ids[-1]
        if last_index != frame_index:
            return torch.zeros(len(proposal_instances), dtype=torch.bool)
        last_instance = self.proposal_instances[-1]"""

        tubelet_instance = self.get_instance(frame_id)
        if tubelet_instance is None:
            return torch.zeros(len(proposal_instances), dtype=torch.bool)
        if use_iou:
            iou = pairwise_iou(tubelet_instance.pred_boxes, proposal_instances.pred_boxes)
            return (iou > self.iou_threshold)[0,:]
        else:
            last_instance_box = Boxes(tubelet_instance.pred_boxes.tensor)
            proposal_instance_boxes = Boxes(proposal_instances.pred_boxes.tensor)
            last_instance_area = last_instance_box.area()
            proposal_instance_area = proposal_instance_boxes.area()
            intersections = pairwise_intersection(tubelet_instance.pred_boxes, proposal_instances.pred_boxes)
            return ((intersections / last_instance_area > 0.8) | (intersections / proposal_instance_area > 0.8))[0, :]

    def __len__(self):
        return len(self.frame_ids)

    def is_active(self, frame_id):
        return self.frame_ids[-1] == frame_id

    def get_instance(self, frame_id):
        if frame_id not in self.frame_ids:
            return None
        else:
            i = self.frame_ids.index(frame_id)
            if self.proposal_instances[i] is not None:
                return self.proposal_instances[i]
            else:
                # This has been a skipped frame ... interpolate box from the neighboring instances
                index_before = index_after = i
                while self.proposal_instances[index_before] is None and index_before > 0:
                    index_before -= 1
                while self.proposal_instances[index_after] is None and index_after < len(self.proposal_instances):
                    index_after += 1
                instance_before = self.proposal_instances[index_before]
                instance_after = self.proposal_instances[index_after]

                interpolation_factor = (i - index_before) / (index_after - index_before)

                interpolated_instance = Instances(instance_before.image_size)
                interpolated_instance.pred_boxes = Boxes(instance_before.pred_boxes.tensor + interpolation_factor * (instance_after.pred_boxes.tensor - instance_before.pred_boxes.tensor))
                interpolated_instance.scores = torch.tensor([0])
                interpolated_instance.pred_classes = instance_before.pred_classes
                interpolated_instance.class_distributions = instance_before.class_distributions
                interpolated_instance.generation_process = ["I"]
                return interpolated_instance

    def get_projected_instance(self, frame_id):
        if frame_id not in self.frame_ids:
            return None
        else:
            i = self.frame_ids.index(frame_id)
            return self.projected_proposal_instances[i]



def generate_tubelets(
        args,
        proposals_dict,
        score_threshold=0.7,
        start_frame=0,
        class_index_to_detect=0,
        tubelet_iou_threshold=0.2,
        nms_iou_threshold=0.4,
        extend_class_only=True,
        num_skippable_frames=5,
        max_dimension_change_ratio=None,
        max_dimension_change_abs=15,
        perform_projection=True
):
    tubelets = []
    if args.method in {"all", "threshold", "nms"}:
        with tqdm.tqdm(total=len(proposals_dict)) as pbar:
            for i, frame_path in enumerate(sorted(proposals_dict.keys())):
                pbar.update(1)
                proposal_instances = proposals_dict[frame_path]
                if args.method == "all":
                    surviving_instances = proposal_instances
                elif args.method == "threshold":
                    surviving_instances = proposal_instances[proposal_instances.class_distributions[:, class_index_to_detect] > score_threshold]
                elif args.method == "nms":
                    surviving_instances = proposal_instances[proposal_instances.class_distributions[:, class_index_to_detect] > score_threshold]
                    surviving_indices = nms(surviving_instances.pred_boxes.tensor, surviving_instances.class_distributions[:, class_index_to_detect], nms_iou_threshold)
                    surviving_instances = surviving_instances[surviving_indices]

                # Generate one tubelet for each proposal
                for proposal_index in range(len(surviving_instances)):
                    tubelets.append(Tubelet(
                            start_index = i+start_frame,
                            proposal_instance = surviving_instances[proposal_index],
                            class_to_detect = class_index_to_detect
                        ))
    elif args.method == "tubelet":
        print(f"Tubelet forward pass through {len(proposals_dict)} frames ...")
        proposals_dict_after_nms = {}
        with tqdm.tqdm(total=len(proposals_dict)) as pbar:
            for i, frame_path in enumerate(sorted(proposals_dict.keys())):
                pbar.update(1)
                proposal_instances = proposals_dict[frame_path]
                proposal_instances_after_nms = proposal_instances[nms(proposal_instances.pred_boxes.tensor, proposal_instances.class_distributions[:, class_index_to_detect], nms_iou_threshold)]
                proposals_dict_after_nms[frame_path] = proposal_instances_after_nms
                non_colliding_proposal_mask = torch.zeros(len(proposal_instances_after_nms), dtype=torch.bool)
                for tubelet_id, tubelet in enumerate(tubelets):
                    tubelet.extend_after(start_frame + i, proposal_instances_after_nms[torch.nonzero(~non_colliding_proposal_mask)[:,0]])
                    non_colliding_proposal_mask = non_colliding_proposal_mask | tubelet.collides_with_mask(start_frame + i, proposal_instances_after_nms)
                key_proposal_indices = torch.nonzero((proposal_instances_after_nms.scores > score_threshold) & (proposal_instances_after_nms.pred_classes == class_index_to_detect) & ~non_colliding_proposal_mask)
                for key_proposal_index in key_proposal_indices:
                    tubelets.append(Tubelet(
                        start_index = i+start_frame,
                        proposal_instance = proposal_instances_after_nms[key_proposal_index],
                        class_to_detect = class_index_to_detect,
                        iou_threshold = tubelet_iou_threshold,
                        extend_class_only = extend_class_only,
                        num_skippable_frames = num_skippable_frames,
                        max_dimension_change_ratio = max_dimension_change_ratio,
                        max_dimension_change_abs = max_dimension_change_abs,
                        perform_projection = perform_projection
                    ))
        print(f"Tubelet backward pass through {len(proposals_dict)} frames ...")
        with tqdm.tqdm(total=len(proposals_dict_after_nms)) as pbar:
            for i, frame_path in enumerate(reversed(sorted(proposals_dict_after_nms.keys()))):
                pbar.update(1)
                proposal_instances_after_nms = proposals_dict_after_nms[frame_path]
                colliding_proposal_mask = torch.zeros(len(proposal_instances_after_nms), dtype=torch.bool)
                for tubelet in tubelets:
                    tubelet_colliding_proposal_mask = tubelet.collides_with_mask(start_frame + len(proposals_dict_after_nms) - i - 1, proposal_instances_after_nms)
                    colliding_proposal_mask = colliding_proposal_mask | tubelet_colliding_proposal_mask
                for tubelet in tubelets:
                    if tubelet.extend_before(start_frame + len(proposals_dict_after_nms) - i - 1, proposal_instances_after_nms[torch.nonzero(~colliding_proposal_mask)[:,0]]):
                        tubelet_colliding_proposal_mask = tubelet.collides_with_mask(start_frame + len(proposals_dict_after_nms) - i - 1, proposal_instances_after_nms)
                        colliding_proposal_mask = colliding_proposal_mask | tubelet_colliding_proposal_mask
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print()
    print(f"Tubelet statistics:")
    print(f"    - Overall: {len(tubelets)}, avergae length: {sum(len(t) for t in tubelets) / len(tubelets) if tubelets else 0}")
    return tubelets


def draw_instance_predictions(visualizer, tubelet_ids, tubelet_instances, tubelet_instance_projections, draw_projections=False):
    def get_color(i):
        colors = "bgrcmykw"
        return colors[i % len(colors)]

    if not any(tubelet_instances):
        return visualizer.output

    tubelet_instance_ids = [i for i, inst in zip(tubelet_ids, tubelet_instances) if inst is not None]
    tubelet_instances = Instances.cat([inst for inst in tubelet_instances if inst is not None])

    tubelet_instance_projection_ids = [i for i, inst in zip(tubelet_ids, tubelet_instance_projections) if inst is not None]
    tubelet_instance_projections = Instances.cat([inst for inst in tubelet_instance_projections if inst is not None])

    labels = _create_text_labels(tubelet_instances.pred_classes, tubelet_instances.scores, visualizer.metadata.get("thing_classes", None))
    for i, tubelet_id in enumerate(tubelet_instance_ids):
        labels[i] = f"{labels[i]} ({tubelet_instances.generation_process[i]}, #{tubelet_id})"

    colors = [get_color(i) for i in tubelet_instance_ids]
    visualizer.overlay_instances(
        boxes=tubelet_instances.pred_boxes,
        labels=labels,
        assigned_colors=colors,
        alpha=0.5,
    )

    if draw_projections:
        colors = [get_color(i) for i in tubelet_instance_projection_ids]
        labels = [f"Pred. #{i}" for i in tubelet_instance_projection_ids]
        visualizer.overlay_instances(
            boxes=tubelet_instance_projections.pred_boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=0.1,
        )
    return visualizer.output


def render_frame(prms):
    frame_id, frame_path, tubelets, rendered_frames_dir, cfg = prms

    rendered_frame_path = rendered_frames_dir / frame_path.name
    if rendered_frame_path.exists():
        return

    tubelet_ids = []
    tubelet_instances = []
    tubelet_instance_projections = []
    for tubelet_id, tubelet in enumerate(tubelets):
        tubelet_ids.append(tubelet_id)
        tubelet_instances.append(tubelet.get_instance(frame_id))
        tubelet_instance_projections.append(tubelet.get_projected_instance(frame_id))

    frame = cv2.imread(str(frame_path))
    if tubelet_ids:
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = draw_instance_predictions(v, tubelet_ids, tubelet_instances, tubelet_instance_projections)
        cv2.imwrite(str(rendered_frame_path), v.get_image()[:, :, ::-1])
    else:
        cv2.imwrite(str(rendered_frame_path), frame)


def render_video(args):
    cfg, _ = load_model(config_only=True)

    proposals_dir = args.working_dir / args.video_id / "proposals"
    frames_dir = args.working_dir / args.video_id / "frames"
    rendered_frames_dir = args.working_dir / args.video_id / f"rendered_frames__{args.method}"

    if not rendered_frames_dir.exists():
        rendered_frames_dir.mkdir()

    frame_list = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"Loading {len(frame_list)} frames and proposals")
    proposals_dict = {}
    with tqdm.tqdm(total=len(frame_list)) as pbar:
        for i, frame_path in enumerate(frame_list):
            pbar.update(1)
            frame_id = int(frame_path.name[frame_path.name.find("_")+1:frame_path.name.find(".")])
            if args.frame_start is not None and frame_id < args.frame_start:
                continue
            if args.frame_stop is not None and frame_id > args.frame_stop:
                continue

            proposals_path = proposals_dir / frame_path.with_suffix(".pickle").name
            if not proposals_path.exists():
                continue
            with proposals_path.open("rb") as f_in:
                proposals = pickle.load(f_in)

            proposals_dict[frame_path] = proposals

    print(f"Generating tubelets for {args.video_id}")
    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    class_index_to_detect = classes.index(args.class_to_detect)
    tubelets = generate_tubelets(
        args,
        proposals_dict,
        start_frame=args.frame_start if args.frame_start is not None else 0,
        class_index_to_detect=class_index_to_detect
    )

    rendering_prms = []
    for i, frame_path in enumerate(frame_list):
        frame_id = int(frame_path.name[frame_path.name.find("_") + 1:frame_path.name.find(".")])
        if args.frame_start is not None and frame_id < args.frame_start:
            continue
        if args.frame_stop is not None and frame_id > args.frame_stop:
            continue
        rendering_prms.append((i, frame_path, tubelets, rendered_frames_dir, cfg))

    print(f"Rendering {len(rendering_prms)} frames for {args.video_id}.")
    with ThreadPoolExecutor(max_workers=30) as executor:
        result = list(tqdm.tqdm(executor.map(render_frame, rendering_prms), total=len(rendering_prms)))


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
        pred_classes=predictions["instances"].pred_classes[instances_to_keep],
        class_distributions=predictions["instances"].class_distributions[instances_to_keep]
    )}


def generate_poposals(images, model, score_threshold=0):
    inputs = [{
        "image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)),
        "height": image.shape[0],
        "width": image.shape[1]
    } for image in images]

    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)
        proposals_scores, proposals_deltas = model.roi_heads.box_predictor(box_features)

        boxes_tensors = model.roi_heads.box_predictor.predict_boxes((proposals_scores, proposals_deltas), proposals)
        scores = model.roi_heads.box_predictor.predict_probs((proposals_scores, proposals_deltas), proposals)

        result = []
        for i in range(len(inputs)):
            image_size = proposals[i].image_size
            num_bbox_reg_classes = boxes_tensors[i].shape[1] // 4
            boxes = Boxes(boxes_tensors[i].reshape(-1, 4))
            boxes.clip(image_size)
            boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)

            img_scores = scores[i][:, :-1]
            max_scores, pred_classes = torch.max(img_scores, dim=1)

            keep_mask = max_scores > score_threshold
            filtered_scores = img_scores[keep_mask, :]
            filtered_max_scores = max_scores[keep_mask]
            filtered_pred_classes = pred_classes[keep_mask]
            boxes = boxes[keep_mask, filtered_pred_classes, :]

            result_instance = Instances(image_size)
            result_instance.pred_boxes = Boxes(boxes)
            result_instance.scores = filtered_max_scores
            result_instance.pred_classes = filtered_pred_classes
            result_instance.class_distributions = filtered_scores
            result.append(result_instance)

    return result


def load_model(config_only=False, architecture="faster_rcnn_50"):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"

    if architecture == "faster_rcnn_101":
        cfg.merge_from_file(Path(detectron2.__file__).resolve().parent / "model_zoo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    elif architecture == "faster_rcnn_50":
        cfg.merge_from_file(Path(detectron2.__file__).resolve().parent / "model_zoo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"

    if config_only:
        return cfg, None

    print("Loading model.")
    predictor = DefaultPredictor(cfg)
    return cfg, predictor.model


def assemble_results(args):
    source_path = args.working_dir / args.video_id / f"rendered_frames__{args.method}" / "frame_%04d.jpg"
    target_path = args.working_dir / args.video_id / f"video__{args.method}.mp4"
    if target_path.exists():
        print(f"Target video {target_path} already exists")
    else:
        print(f"Assembling video {target_path}")
        #subprocess.run(["ffmpeg", "-r", "25", "-start_number", "1500", "-i", str(source_path), "-y", str(target_path)])
        subprocess.run(["ffmpeg", "-r", "25", "-i", str(source_path), "-y", str(target_path)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="One of: preprocess_video, ...")
    parser.add_argument("--working_dir", default="/tmp/video-object-detection/")
    parser.add_argument("--video_id", default="eKKdRy20HXI")
    parser.add_argument("--class_to_detect", default="car")
    parser.add_argument("--frame_start", default=None)
    parser.add_argument("--frame_stop", default=None)
    parser.add_argument("--method", default="tubelet", help="One of: all, threshold, nms, tubelet")
    args = parser.parse_args()

    args.working_dir = Path(args.working_dir)
    args.frame_start = int(args.frame_start) if args.frame_start is not None else None
    args.frame_stop = int(args.frame_stop) if args.frame_stop is not None else None

    if args.action == "preprocess_video":
        preprocess_video(args)
    elif args.action == "render_video":
        render_video(args)
    elif args.action == "assemble_results":
        assemble_results(args)
    else:
        raise NotImplementedError(f"Unknown action: {args.action}")