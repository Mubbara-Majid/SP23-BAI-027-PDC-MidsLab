import os
import time
import cv2
import numpy as np

INPUT_DIR = 'data_set'
OUTPUT_DIR = 'output_seq'
TARGET_SIZE = (128, 128)
WATERMARK_TEXT = "Processed"

SUBFOLDERS = ['cars', 'Cat', 'dogs', 'Flowers']
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

def process_image(input_path, output_path):
    
    try:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            print(f"Warning: Could not read {input_path}. Skipping.")
            return

        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (255, 255, 255) 
        
        (text_width, text_height), baseline = cv2.getTextSize(WATERMARK_TEXT, font, font_scale, font_thickness)
        
        x = TARGET_SIZE[0] - text_width - 10
        y = TARGET_SIZE[1] - 10
        
        cv2.putText(img_resized, WATERMARK_TEXT, (x, y), font, font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
        cv2.putText(img_resized, WATERMARK_TEXT, (x, y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cv2.imwrite(output_path, img_resized)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    
    print("Starting sequential processing (OpenCV)...")
    
    start_time = time.perf_counter()

    for folder_name in SUBFOLDERS:
        input_subfolder = os.path.join(INPUT_DIR, folder_name)
        output_subfolder = os.path.join(OUTPUT_DIR, folder_name)
        
        if not os.path.isdir(input_subfolder):
            print(f"Warning: Input directory not found: {input_subfolder}. Skipping.")
            continue
            
        os.makedirs(output_subfolder, exist_ok=True)

        try:
            for file in os.listdir(input_subfolder):
                if file.lower().endswith(ALLOWED_EXTENSIONS):
                    
                    input_path = os.path.join(input_subfolder, file)
                    
                    base, ext = os.path.splitext(file)
                    output_path = os.path.join(output_subfolder, f"{base}.png")
                    
                    process_image(input_path, output_path)
        except Exception as e:
            print(f"Error listing files in {input_subfolder}: {e}")


    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"\nSequential Processing Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()