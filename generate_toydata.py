import cv2
import click
import numpy as np
from pathlib import Path


def generate_labeled_circles(image_width: int, image_height: str):
    assert image_width >= image_height
    color_image = np.zeros((image_height, image_width, 3), np.uint8)
    label_image = np.zeros((image_height, image_width), np.uint8)
    circle_radius = int(image_height * 0.33)
    circle_center_left = (circle_radius, image_height // 2)
    circle_center_right = (image_width - circle_radius, image_height // 2)

    # draw left circle
    cv2.circle(color_image, circle_center_left, circle_radius, (255, 0, 0), cv2.FILLED)
    cv2.circle(label_image, circle_center_left, circle_radius, (1), cv2.FILLED)

    # draw right circle
    cv2.circle(color_image, circle_center_right, circle_radius, (0, 0, 255), cv2.FILLED)
    cv2.circle(label_image, circle_center_right, circle_radius, (2), cv2.FILLED)
    return color_image, label_image


def save_image(title: str, image: np.ndarray) -> None:
    cv2.imwrite(title, image)
    cv2.waitKey(10)


@click.command()
@click.option("--output-dir", "-o", default="data")
@click.option("--color-image-name", "-c", default="color.png")
@click.option("--label-image-name", "-l", default="label.png")
@click.option("--image-width", default=512)
@click.option("--image-height", default=384)
def main(output_dir, color_image_name, label_image_name, image_width, image_height):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    output_color_image_name = str(output_dir_path.joinpath(color_image_name))
    output_label_image_name = str(output_dir_path.joinpath(label_image_name))
    color_image, label_image = generate_labeled_circles(image_width, image_height)
    save_image(output_color_image_name, color_image)
    save_image(output_label_image_name, label_image)


if __name__ == "__main__":
    main()