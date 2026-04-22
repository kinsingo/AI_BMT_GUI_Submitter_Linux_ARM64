# Compiler Version v3.31

# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

# Segmentation
# for onnx_model_name in ["deeplabv3_mobilenet_v3_large_opset12", "deeplabv3_resnet50_opset12", "deeplabv3_resnet101_opset12", "fcn_resnet50_opset12", "fcn_resnet101_opset12"]:
#     onnx_path = f"Segmentation_onnxs/{onnx_model_name}.onnx"
#     runner = ClientRunner(hw_arch="hailo8")
#     hn, npz = runner.translate_onnx_model(
#         onnx_path,
#         onnx_model_name,
#         start_node_names=["input"],
#         end_node_names=["output"],   
#         net_input_shapes={"input": [1, 3, 520, 520]},
#     )
#     hailo_model_har_name = f"{onnx_model_name}_hailo_model.har"
#     runner.save_har(hailo_model_har_name)

# Object Detection 
import os
import re

# Create ObjectDetection_hars folder if it doesn't exist
os.makedirs("ObjectDetection_hars", exist_ok=True)

onnx_files = sorted([f.replace(".onnx", "") for f in os.listdir("ObjectDetection_onnxs") if f.endswith(".onnx")])
runner = ClientRunner(hw_arch="hailo8")

# Define end node names for different model types.
# List of (regex_pattern, end_nodes) tuples — checked in order, first match wins.
# None means no end_node_names (auto-detect by Hailo SDK).
#
# Key insight: DFL-based models (yolov5u/v8/v9/v10/yolo11/yolo12) must use
# 6 separate Conv end nodes (3x cv2 for box regression + 3x cv3 for class prediction)
# instead of a single Concat output. The single Concat creates a tensor [1,1,8400,144]
# that exceeds Hailo-8's 128 memory unit limit (needs 150 units).
# Using 6 separate smaller outputs avoids this memory bottleneck.
#
# For yolov6: 6 end nodes = 3x reg_preds Conv + 3x Sigmoid (cls after sigmoid).
# For yolov7/7x: 3 Sigmoid end nodes (anchor-based, already separate outputs).
end_nodes_config = [
    # YOLOv5u variants (DFL-based, /model.24/ detect head, 6 Conv end nodes)
    (r"yolov5[mns]u", [
        "/model.24/cv2.0/cv2.0.2/Conv", "/model.24/cv3.0/cv3.0.2/Conv",
        "/model.24/cv2.1/cv2.1.2/Conv", "/model.24/cv3.1/cv3.1.2/Conv",
        "/model.24/cv2.2/cv2.2.2/Conv", "/model.24/cv3.2/cv3.2.2/Conv",
    ]),
    # YOLOv6 n/s/m — 6 end nodes: Sigmoid (cls) + reg_preds Conv (box) per scale
    (r"yolov6", [
        "/detect/Sigmoid",          "/detect/reg_preds.0/Conv",
        "/detect/Sigmoid_1",        "/detect/reg_preds.1/Conv",
        "/detect/Sigmoid_2",        "/detect/reg_preds.2/Conv",
    ]),
    # YOLOv7x — anchor-based, 3 Sigmoid detection heads in /model.121/
    (r"yolov7x", ["/model.121/Sigmoid", "/model.121/Sigmoid_1", "/model.121/Sigmoid_2"]),
    # YOLOv7 base — anchor-based, 3 Sigmoid detection heads in /model.105/
    (r"yolov7",  ["/model.105/Sigmoid", "/model.105/Sigmoid_1", "/model.105/Sigmoid_2"]),
    # YOLOv8 (DFL-based, /model.22/ detect head, 6 Conv end nodes)
    (r"yolov8", [
        "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv",
    ]),
    # YOLOv9 (same head structure as YOLOv8, /model.22/)
    (r"yolov9", [
        "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv",
    ]),
    # YOLOv10 (one2one detect head, /model.23/, 6 Conv end nodes)
    (r"yolov10", [
        "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv", "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
        "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv", "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
        "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv", "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",
    ]),
    # YOLO11 (DFL-based, /model.23/ detect head, 6 Conv end nodes)
    (r"yolo11", [
        "/model.23/cv2.0/cv2.0.2/Conv", "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv", "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv", "/model.23/cv3.2/cv3.2.2/Conv",
    ]),
    # YOLO12 (DFL-based, /model.21/ detect head, 6 Conv end nodes)
    (r"yolo12", [
        "/model.21/cv2.0/cv2.0.2/Conv", "/model.21/cv3.0/cv3.0.2/Conv",
        "/model.21/cv2.1/cv2.1.2/Conv", "/model.21/cv3.1/cv3.1.2/Conv",
        "/model.21/cv2.2/cv2.2.2/Conv", "/model.21/cv3.2/cv3.2.2/Conv",
    ]),
]

for onnx_model_name in onnx_files:
    onnx_path = f"ObjectDetection_onnxs/{onnx_model_name}.onnx"
    
    # Determine end nodes based on model name (first regex match wins)
    end_nodes = "NO_MATCH"
    for pattern, nodes in end_nodes_config:
        if re.search(pattern, onnx_model_name.lower()):
            end_nodes = nodes
            break
    
    if end_nodes == "NO_MATCH":
        print(f"[SKIP] {onnx_model_name}: no matching config in end_nodes_config")
        continue

    hailo_model_har_name = f"ObjectDetection_hars/{onnx_model_name}_hailo_model.har"
    if os.path.exists(hailo_model_har_name):
        print(f"[SKIP] {hailo_model_har_name} already exists")
        continue

    try:
        if end_nodes is not None:
            hn, npz = runner.translate_onnx_model(
                onnx_path,
                onnx_model_name,
                end_node_names=end_nodes,
                net_input_shapes={"images": [1, 3, 640, 640]},
            )
        else:
            hn, npz = runner.translate_onnx_model(
                onnx_path,
                onnx_model_name,
                net_input_shapes={"images": [1, 3, 640, 640]},
            )
        runner.save_har(hailo_model_har_name)
        print(f"[SUCCESS] {onnx_model_name} converted successfully")
    except Exception as e:
        print(f"[ERROR] Failed to convert {onnx_model_name}: {str(e)}")
        # Extract recommended end node from error message if available
        if "Please try to parse the model again, using these end node names:" in str(e):
            recommended_node = str(e).split("using these end node names:")[-1].strip()
            print(f"[RECOMMENDATION] Try using end_node_names=['{recommended_node}']")
        continue

