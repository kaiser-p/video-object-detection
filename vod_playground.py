import argparse
import pytube
import pytube.cli
import cv2
import tqdm
from pathlib import Path


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


def detect_objects(args):
    pass


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
    else:
        raise NotImplementedError(f"Unknown action: {args.action}")