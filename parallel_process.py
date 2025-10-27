import os
import time
import cv2
import numpy as np
import multiprocessing

INPUT_DIR = 'data_set'
OUTPUT_DIR = 'output_parallel' 
TARGET_SIZE = (128, 128) 
WATERMARK_TEXT = "Processed"
SUBFOLDERS = ['cars', 'Cat', 'dogs', 'Flowers']
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

WORKER_COUNTS = [1, 2, 4, 8]

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

        cv2.imwrite(output_path, img_resized)

    except Exception as e:
        print(f"Error in worker processing {input_path}: {e}")

def main():
    print("Starting parallel processing test...")
    
    
    tasks = [] 
    output_dirs_to_create = set()
    for folder_name in SUBFOLDERS:
        input_subfolder = os.path.join(INPUT_DIR, folder_name)
        output_subfolder = os.path.join(OUTPUT_DIR, folder_name)
        
        output_dirs_to_create.add(output_subfolder)
        
        if not os.path.isdir(input_subfolder):
            print(f"Warning: Input directory not found: {input_subfolder}. Skipping.")
            continue

        try:
            for file in os.listdir(input_subfolder):
                if file.lower().endswith(ALLOWED_EXTENSIONS):
                    input_path = os.path.join(input_subfolder, file)
                    base, ext = os.path.splitext(file)
                    output_path = os.path.join(output_subfolder, f"{base}.png")
                    
                    tasks.append((input_path, output_path))
        except Exception as e:
            print(f"Error listing files in {input_subfolder}: {e}")

    if not tasks:
        print("No images found. Exiting.")
        return
        
    print(f"Found {len(tasks)} images to process.")

    for dir_path in output_dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        
    results = [] 
    
    for count in WORKER_COUNTS:
        print(f"\nProcessing with {count} worker(s)...")
        start_time = time.perf_counter()

        with multiprocessing.Pool(processes=count) as pool:
            pool.starmap(process_image, tasks)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print(f"Time with {count} worker(s): {total_time:.2f} seconds")
        results.append((count, total_time))

    print("\n--- Speedup Table ---")
    print(f"{'Workers':<8} | {'Time (s)':<10} | {'Speedup':<8}")
    print("-" * 30)
    
    baseline_time = results[0][1] 
    
    for workers, time_taken in results:
        speedup = baseline_time / time_taken
        print(f"{workers:<8} | {time_taken:<10.2f} | {speedup:<8.2f}x")

if __name__ == "__main__":
    main()