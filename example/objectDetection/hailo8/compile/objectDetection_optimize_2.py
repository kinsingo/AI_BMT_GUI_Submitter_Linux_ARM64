#Compiler version v3.31

import os
import re
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext

images_path = "ObjectDetection_Calibration_Images"
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]
calib_dataset = np.zeros((len(images_list), 640, 640, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    assert img.shape == (640, 640, 3), f"{img_name} has unexpected shape {img.shape}"
    calib_dataset[idx] = img

# Create ObjectDetection_quantized_hars folder if it doesn't exist
os.makedirs("ObjectDetection_quantized_hars", exist_ok=True)

# Get all HAR files from ObjectDetection_hars folder
har_files = [f for f in os.listdir("ObjectDetection_hars") if f.endswith(".har")]

for har_file in har_files:
    model_name = har_file.replace("_hailo_model.har", "")
    hailo_model_har_name = f"ObjectDetection_hars/{har_file}"
    
    print(f"[INFO] Processing {model_name}...")
    
    try:

        quantized_model_har_path = f"ObjectDetection_quantized_hars/{model_name}_bgr2rgb_normalized_quantized_model.har"
        if os.path.exists(quantized_model_har_path):
            print(f"[INFO]{quantized_model_har_path} is existing.. continue..")
            continue

        runner = ClientRunner(har=hailo_model_har_name)
        opt_level = 2
        alls = (
            f"model_optimization_flavor(optimization_level={opt_level},compression_level=1,batch_size=2)\n"
            "color_convert = input_conversion(bgr_to_rgb, emulator_support=True)\n"
            "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
        )

        # Auto-detect class prediction Conv layers and add change_output_activation(sigmoid).
        # For models with 6 Conv end nodes (yolov5u/v8/v9/v10/yolo11/yolo12),
        # class outputs (cv3) have more channels (80 for COCO) than box outputs (cv2, 64 channels).
        # Skip for yolov6 (class end nodes are already Sigmoid) and yolov7/7x (Sigmoid end nodes).
        if not re.search(r"yolov[67]", model_name.lower()):
            hn = runner.get_hn()
            layers = hn.get('layers', {})
            output_convs = []
            for name in sorted(layers.keys()):
                layer = layers[name]
                if layer.get('type') == 'output_layer':
                    shapes = layer.get('output_shapes', [[]])
                    inputs = layer.get('input', [])
                    if inputs and shapes and len(shapes[0]) >= 4:
                        conv_name = inputs[0].split('/')[-1]
                        channels = shapes[0][3]
                        output_convs.append((conv_name, channels))

            if len(output_convs) >= 2:
                min_ch = min(ch for _, ch in output_convs)
                class_convs = [n for n, ch in output_convs if ch > min_ch]
                for conv_name in class_convs:
                    alls += f"change_output_activation({conv_name}, sigmoid)\n"
                if class_convs:
                    print(f"[INFO] Added change_output_activation(sigmoid) for: {', '.join(class_convs)}")

        runner.load_model_script(alls)
        # Call Optimize to perform the optimization process
        runner.optimize(calib_dataset) 
        # Save the result state to a Quantized HAR file
        runner.save_har(quantized_model_har_path)
        print(f"[SUCCESS] {model_name} optimized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to optimize {model_name}: {str(e)}")
        continue

# Create ObjectDetection_quantized_hars folder if
