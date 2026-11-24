"""
Sample Dataset Generator for CNN Image Classification.
Generates synthetic geometric shape images for class_1 (circles), class_2 (squares),
and class_3 (triangles) to allow out-of-the-box training and testing.
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter


def create_sample_dataset(base_dir="dataset", samples_per_class=40, img_size=(150, 150)):
    classes = {
        "class_1": "circle",
        "class_2": "square",
        "class_3": "triangle"
    }

    colors = [
        (230, 57, 70),   # Red / Coral
        (69, 123, 157),  # Steel Blue
        (42, 157, 143),  # Teal Green
        (244, 162, 97),  # Sandy Orange
        (155, 93, 229),  # Purple
        (0, 180, 216),   # Sky Blue
    ]

    for class_folder, shape_type in classes.items():
        folder_path = os.path.join(base_dir, class_folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Generating {samples_per_class} images for {class_folder} ({shape_type})...")

        for i in range(1, samples_per_class + 1):
            # Background with slight variation
            bg_color = (
                random.randint(235, 255),
                random.randint(235, 255),
                random.randint(235, 255)
            )
            img = Image.new("RGB", img_size, color=bg_color)
            draw = ImageDraw.Draw(img)

            # Random shape properties
            color = random.choice(colors)
            w, h = img_size
            radius = random.randint(25, 45)
            cx = random.randint(radius + 15, w - radius - 15)
            cy = random.randint(radius + 15, h - radius - 15)

            if shape_type == "circle":
                bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
                draw.ellipse(bbox, fill=color, outline=(20, 20, 20), width=random.randint(1, 3))
            elif shape_type == "square":
                bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
                # Optional rotation simulation
                draw.rectangle(bbox, fill=color, outline=(20, 20, 20), width=random.randint(1, 3))
            elif shape_type == "triangle":
                angle_offset = random.uniform(-0.3, 0.3)
                points = []
                for p_idx in range(3):
                    angle = angle_offset + p_idx * (2 * math.pi / 3) - math.pi / 2
                    px = cx + radius * math.cos(angle)
                    py = cy + radius * math.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=color, outline=(20, 20, 20), width=random.randint(1, 3))

            # Add gentle anti-aliasing / blur
            if random.random() > 0.5:
                img = img.filter(ImageFilter.SMOOTH_MORE)

            filename = f"image_{i:03d}.jpg"
            img_path = os.path.join(folder_path, filename)
            img.save(img_path, format="JPEG", quality=90)

    print(f"\n[OK] Sample dataset successfully created in '{base_dir}/'.")


if __name__ == "__main__":
    create_sample_dataset()
