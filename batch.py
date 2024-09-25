import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Configuration options
PRINT_CAPTIONS = False  # Print captions to the console during inference
PRINT_CAPTIONING_STATUS = False  # Print captioning file status to the console
OVERWRITE = False  # Allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
STRIP_LINEBREAKS = True  # Remove line breaks from generated captions before saving
DEFAULT_SAVE_FORMAT = ".txt"  # Default format for saving captions

# Image resizing options
MAX_WIDTH = 1024  # Set to 0 or less to ignore
MAX_HEIGHT = 1024  # Set to 0 or less to ignore

# Generation parameters
REPETITION_PENALTY = 1.3  # Penalty for repeating phrases, float ~1.5
TEMPERATURE = 0.7  # Sampling temperature to control randomness
TOP_K = 50  # Top-k sampling to limit number of potential next tokens

# Default values for input folder, output folder, prompt, and save format
DEFAULT_INPUT_FOLDER = Path(__file__).parent / "input"
DEFAULT_OUTPUT_FOLDER = DEFAULT_INPUT_FOLDER
DEFAULT_PROMPT = "Describe this image."

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and generate captions using Qwen model.")
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT_FOLDER, help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Path to the output folder for saving captions.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for generating the caption.")
    parser.add_argument("--save_format", type=str, default=DEFAULT_SAVE_FORMAT, help="Format for saving captions (e.g., .txt, .md, .json).")
    parser.add_argument("--max_width", type=int, default=None, help="Maximum width for resizing images (default: no resizing).")
    parser.add_argument("--max_height", type=int, default=None, help="Maximum height for resizing images (default: no resizing).")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, help="Penalty for repetition during caption generation (default: 1.10).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature for generation (default: 0.7).")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k sampling during generation (default: 50).")
    return parser.parse_args()

# Function to ignore images that don't have output files yet
def filter_images_without_output(input_folder, save_format):
    images_to_caption = []
    skipped_images = 0
    total_images = 0

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                total_images += 1
                image_path = os.path.join(root, file)
                output_path = os.path.splitext(image_path)[0] + save_format
                if not OVERWRITE and os.path.exists(output_path):
                    skipped_images += 1
                else:
                    images_to_caption.append(image_path)

    return images_to_caption, total_images, skipped_images

# Function to save caption to a file
def save_caption_to_file(image_path, caption, save_format):
    txt_file_path = os.path.splitext(image_path)[0] + save_format  # Same name, but with chosen save format
    caption = PREPEND_STRING + caption + APPEND_STRING  # Apply prepend/append strings

    with open(txt_file_path, "w") as txt_file:
        txt_file.write(caption)

    if PRINT_CAPTIONING_STATUS:
        print(f"Caption for {os.path.abspath(image_path)} saved in {save_format} format.")

# Function to process all images recursively in a folder
def process_images_in_folder(images_to_caption, prompt, save_format, max_width=MAX_WIDTH, max_height=MAX_HEIGHT, repetition_penalty=REPETITION_PENALTY, temperature=TEMPERATURE, top_k=TOP_K):
    
    for image_path in tqdm(images_to_caption, desc="Processing Images"):
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize the image if necessary
            image = resize_image_proportionally(image, max_width, max_height)
            
            caption = qwen_caption(image, prompt, repetition_penalty, temperature, top_k)
            save_caption_to_file(image_path, caption, save_format)

            if PRINT_CAPTIONS:
                print(f"Caption for {os.path.abspath(image_path)}: {caption}")

        except Exception as e:
            print(f"Error processing {os.path.abspath(image_path)}: {str(e)}")

        torch.cuda.empty_cache()

# Resize the image proportionally based on max width and/or max height.
def resize_image_proportionally(image, max_width=None, max_height=None):
    """
    If both max_width and max_height are provided, the image is resized to fit within both dimensions,
    keeping the aspect ratio intact. If only one dimension is provided, the image is resized based on that dimension.
    """
    if (max_width is None or max_width <= 0) and (max_height is None or max_height <= 0):
        return image  # No resizing if both dimensions are not provided or set to 0 or less

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # Determine the new dimensions
    if max_width and not max_height:
        # Resize based on width
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    elif max_height and not max_width:
        # Resize based on height
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    else:
        # Resize based on both width and height, keeping the aspect ratio
        new_width = max_width
        new_height = max_height

        # Adjust the dimensions proportionally to the aspect ratio
        if new_width / aspect_ratio > new_height:
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int(new_width / aspect_ratio)

    # Resize the image using LANCZOS (equivalent to ANTIALIAS in older versions)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

# Generate a caption for the provided image using the Ertugrul/Qwen2-VL-7B-Captioner-Relaxed model
def qwen_caption(image, prompt, repetition_penalty=REPETITION_PENALTY, temperature=TEMPERATURE, top_k=TOP_K):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.uint8(image))
        
    # Prepare the conversation content, which includes the image and the text prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply the chat template to format the message for processing
    text_prompt = qwen_processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )

    # Prepare the inputs for the model, padding as necessary and converting to tensors
    inputs = qwen_processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

    # Trim the generated IDs to remove the input part from the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    # Decode the trimmed output into text, skipping special tokens
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Strip line breaks if the option is enabled
    if STRIP_LINEBREAKS:
        output_text[0] = output_text[0].replace('\n', ' ')

    return output_text[0]

# Run the script
if __name__ == "__main__":
    args = parse_arguments()
    input_folder = args.input_folder
    output_folder = args.output_folder
    prompt = args.prompt
    save_format = args.save_format
    max_width = args.max_width
    max_height = args.max_height
    repetition_penalty = args.repetition_penalty
    temperature = args.temperature
    top_k = args.top_k

    # Define model_id
    model_id = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"

    # Filter images before loading the model
    images_to_caption, total_images, skipped_images = filter_images_without_output(input_folder, save_format)

    # Print summary of found, skipped, and to-be-processed images
    print(f"\nFound {total_images} image{'s' if total_images != 1 else ''}.")
    if not OVERWRITE:
        print(f"{skipped_images} image{'s' if skipped_images != 1 else ''} already have captions with format {save_format}, skipping.")
    print(f"\nCaptioning {len(images_to_caption)} image{'s' if len(images_to_caption) != 1 else ''}.\n\n")

    # Only load the model if there are images to caption
    if len(images_to_caption) == 0:
        print("No images to process. Exiting.\n\n")
    else:
        # Initialize the Ertugrul/Qwen2-VL-7B-Captioner-Relaxed model
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        qwen_processor = AutoProcessor.from_pretrained(model_id)

        # Process the images with optional resizing and caption generation
        process_images_in_folder(
            images_to_caption,
            prompt,
            save_format,
            max_width=max_width,
            max_height=max_height,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k
        )
