import multiprocessing
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import FaceWrapper

VIDEO_LENGTH = 3.0  # seconds
FPS = 30
VIDEO_FRAME_COUNT = int(VIDEO_LENGTH * FPS)


def create_face_morph(
    face1_path: Union[Path, str],
    face2_path: Union[Path, str],
    morph_3d: bool = False,
    align: bool = False,
    add_background_points: bool = True,
    frame_count: int = VIDEO_FRAME_COUNT,
    fps: int = FPS,
    before_and_after: int = -1,
    display: bool = True,
    pad_frames: int = int(1 * FPS),
    output_path: Path = Path("output/videos"),
) -> Tuple[Path, List]:
    # Ensure paths are Path objects
    face1_path = Path(face1_path)
    face2_path = Path(face2_path)

    # Get number from filename
    pattern = re.compile(r"(04\d+)")
    match1 = pattern.search(str(face1_path))
    match2 = pattern.search(str(face2_path))
    num1 = match1.group(1) if match1 else "unknown"
    num2 = match2.group(1) if match2 else "unknown"

    print(face1_path, face2_path)

    parameters = {
        "face1": face1_path,
        "face2": face2_path,
        "align": align,
        "addBackgroundPoints": add_background_points,
    }

    sequence = FaceWrapper.ParappaTheFaceWrappa(parameters)
    sequence.setup_faces()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Face 1")
    # plt.axis("off")
    # plt.imshow(sequence.face1.imc1)
    # plt.subplot(1, 2, 2)
    # plt.title("Face 2")
    # plt.axis("off")
    # plt.imshow(sequence.face2.imc1)
    # plt.show()

    # Create morph with head interpolation
    images = []
    if morph_3d:
        images = sequence.typical_interpolate_morph(
            before_and_after=before_and_after, timelapse=frame_count
        )
    else:
        images = sequence.typical_morph(timelapse=frame_count)

    morph_name_parts = [
        "morph",
        "3d" if morph_3d else "2d",
        "aligned" if align else "noalign",
        num1,
        num2,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ]
    morph_name = "_".join(morph_name_parts)

    # Generate output filename with timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{morph_name}.mp4"
    sequence.convert_to_video(
        images,
        fps,
        str(output_file),
        pad_beginning_frames=pad_frames,
        pad_end_frames=pad_frames,
    )

    # if display:
    #     display_images(images)

    return output_file, images


def __inner(args):
    face1_path, face2_path, do_3d, do_align, bg_remove = args

    print(f"Processing {face1_path} and {face2_path} with 3D: {do_3d}, align: {do_align}, bg_remove: {bg_remove}")

    f, ims = create_face_morph(
        face1_path=face1_path,
        face2_path=face2_path,
        morph_3d=do_3d,
        align=do_align,
        add_background_points=bg_remove,
        display=False,
    )


if __name__ == "__main__":
    # Get sample images for CNN model
    test_im_files = sorted(
        Path("data/prashantarorat/facial-key-point-data/images").glob("04*.png")
    )

    # Define image pairs to process
    image_pairs = [
        # (04000.png, 04001.png)
        (
            "data/prashantarorat/facial-key-point-data/images/04000.png",
            "data/prashantarorat/facial-key-point-data/images/04001.png",
        ),
        # # (04254.png, 04983.png)
        # ("data/prashantarorat/facial-key-point-data/images/04254.png",
        #  "data/prashantarorat/facial-key-point-data/images/04983.png"),
        # (04175.png, 04964.png)
        (
            "data/prashantarorat/facial-key-point-data/images/04175.png",
            "data/prashantarorat/facial-key-point-data/images/04964.png",
        ),
        # (04712.png, 04746.png)
        (
            "data/prashantarorat/facial-key-point-data/images/04712.png",
            "data/prashantarorat/facial-key-point-data/images/04746.png",
        ),
    ]

    # Create tasks list with image pairs and processing parameters
    tasks = []
    for face1_path, face2_path in image_pairs:
        for do_3d, do_align, bg_remove in [
            (False, False, True),
            (False, True, True),
            (True, False, True),
            (True, True, True),
            (True, False, False),
            (True, True, False),
        ]:
            tasks.append((face1_path, face2_path, do_3d, do_align, bg_remove))

    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(__inner, tasks)

    pool.close()
    pool.join()
