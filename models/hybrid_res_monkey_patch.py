# hybrid-res_monkey_patch.py

from functools import lru_cache
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import logging
import sys

from qwen_vl_utils.vision_process import (
    smart_resize,
    VIDEO_MIN_PIXELS,
    VIDEO_MAX_PIXELS,
    VIDEO_TOTAL_PIXELS,
    FRAME_FACTOR,
    IMAGE_FACTOR,
    get_video_reader_backend,
    VIDEO_READER_BACKENDS,
)

logger = logging.getLogger(__name__)


def patched_fetch_video(
    ele: dict, image_factor: int = IMAGE_FACTOR
) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        max_pixels = ele.get("max_pixels", max_pixels)
        max_low_res_pixels = ele.get("max_low_res_pixels", max_pixels)
        group_size = ele.get("group_size", 0)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if "low_res_height" in ele and "low_res_width" in ele:
            low_res_height, low_res_width = smart_resize(
                ele["low_res_height"],
                ele["low_res_width"],
                factor=image_factor,
            )
        else:
            low_res_height, low_res_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_low_res_pixels,
            )

        if group_size > 0:
            logger.info("start to smart resize video")
            logger.info(f"group_size: {group_size}")
            resized_frames = []
            for i in range(0, nframes, group_size):
                group = video[i : i + group_size]
                group_resized = []
                for j, frame in enumerate(group):
                    if j == 0:
                        frame_resized = transforms.functional.resize(
                            frame,
                            [resized_height, resized_width],
                            interpolation=InterpolationMode.BICUBIC,
                            antialias=True,
                        )
                    else:
                        frame_resized = transforms.functional.resize(
                            frame,
                            [low_res_height, low_res_width],
                            interpolation=InterpolationMode.BICUBIC,
                            antialias=True,
                        )
                    group_resized.append(transforms.ToPILImage()(frame_resized))
                resized_frames.extend(group_resized)
            return resized_frames
        else:
            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return video.float()
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image(
                {"image": video_element, **process_info}, size_factor=image_factor
            )
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def enable_hybrid_res():
    
    import qwen_vl_utils.vision_process as vp

    vp.fetch_video = patched_fetch_video
    print("Applied hybrid res monkey patch", file=sys.stderr)
