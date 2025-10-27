import os
import time
import cv2
import numpy as np
import multiprocessing

INPUT_DIR = 'data_set'
OUTPUT_DIR = 'output_distrib' 
SEQ_OUTPUT_DIR = 'output_seq_baseline' 
TARGET_SIZE = (128, 128) 
WATERMARK_TEXT = "Processed"
SUBFOLDERS = ['cars', 'Cat', 'dogs', 'Flowers']
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
NUM_NODES = 2

def process_image(input_path, output_path):

    try:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
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
        print(f"Error in worker processing {input_path}: {e}")

def get_sequential_time(tasks_list):

    print("Running sequential baseline to get reference time...")
    start_time = time.perf_counter()
    
    for in_path, _ in tasks_list: 
        relative_path = os.path.relpath(in_path, INPUT_DIR)
        base, ext = os.path.splitext(relative_path)
        output_path = os.path.join(SEQ_OUTPUT_DIR, f"{base}.png")
        
        process_image(in_path, output_path)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Sequential baseline time: {total_time:.2f} seconds.")
    return total_time

def worker_node(node_id, tasks_chunk, result_queue):
    
    print(f"[{node_id}] starting, assigned {len(tasks_chunk)} images.")
    
    node_start_time = time.perf_counter()
    
    for in_path, out_path in tasks_chunk:
        process_image(in_path, out_path)
        
    node_end_time = time.perf_counter()
    node_total_time = node_end_time - node_start_time
    
    result_queue.put( (node_id, node_total_time, len(tasks_chunk)) )
    print(f"[{node_id}] finished in {node_total_time:.2f}s.")

def get_all_tasks():
    
    tasks = []
    for folder_name in SUBFOLDERS:
        input_subfolder = os.path.join(INPUT_DIR, folder_name)
        output_subfolder = os.path.join(OUTPUT_DIR, folder_name)
        
        if not os.path.isdir(input_subfolder):
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
    return tasks

def main():
    
    tasks = get_all_tasks()
    if not tasks:
        print("No images found. Exiting.")
        return
    print(f"Found {len(tasks)} total images to process.")

    sequential_time = get_sequential_time(tasks)
    
    print(f"\nStarting distributed simulation ({NUM_NODES} Nodes)...")
    
    result_queue = multiprocessing.Queue()
    
    chunk_size = len(tasks) // NUM_NODES
    task_chunks = []
    for i in range(NUM_NODES):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < NUM_NODES - 1 else len(tasks)
        task_chunks.append(tasks[start:end])

    processes = []
    
    distrib_start_time = time.perf_counter() 
    
    for i in range(NUM_NODES):
        node_id = f"Node {i+1}"
        chunk = task_chunks[i]
        p = multiprocessing.Process(target=worker_node, args=(node_id, chunk, result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    distrib_end_time = time.perf_counter() 
    
    total_distrib_time = distrib_end_time - distrib_start_time
    
    print("\n--- Distributed Simulation Summary ---")
    
    node_reports = []
    for _ in range(NUM_NODES):
        node_reports.append(result_queue.get())
        
    node_reports.sort() 
    
    for node_id, node_time, image_count in node_reports:
        print(f"{node_id} processed {image_count} images in {node_time:.1f}s")
        
    print(f"Total distributed time: {total_distrib_time:.1f}s")
    
    efficiency = sequential_time / total_distrib_time
    print(f"Efficiency: {efficiency:.2f}x over sequential")
    

if __name__ == "__main__":
    main()