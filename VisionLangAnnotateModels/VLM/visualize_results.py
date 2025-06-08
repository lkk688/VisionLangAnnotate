from PIL import Image, ImageDraw, ImageFont
import os

def draw_results(image_path, results, output_path=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for r in results:
        x1, y1, x2, y2 = map(int, r["bbox"])
        label = r["vlm_description"].strip().capitalize()

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red", font=font)

    if output_path:
        image.save(output_path)
        print(f"Saved: {output_path}")
    else:
        image.show()


def visualize_batch(results_list, image_paths, output_dir="visualized"):
    os.makedirs(output_dir, exist_ok=True)
    for results, image_path in zip(results_list, image_paths):
        filename = os.path.basename(image_path)
        out_path = os.path.join(output_dir, filename)
        draw_results(image_path, results, out_path)


from inference_runner import infer_from_folder
from visualize_results import visualize_batch

images = ["data/image1.jpg", "data/image2.jpg"]
results_list = [run_pipeline(img, setup_detectors()) for img in images]

visualize_batch(results_list, images, output_dir="vis_output")