# Compiler Version v3.31

from hailo_sdk_client import ClientRunner
import os

os.makedirs("ObjectDetection_compiled_hars", exist_ok=True)

for quantized_model_har_name in os.listdir("ObjectDetection_quantized_hars"):
    if not quantized_model_har_name.endswith(".har"):
        continue

    har_path = os.path.join("ObjectDetection_quantized_hars", quantized_model_har_name)
    base_name = os.path.splitext(quantized_model_har_name)[0]
    out_path = f"ObjectDetection_compiled_hars/{base_name}_compiled.hef"

    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
        continue

    print(f"[INFO] Compiling {har_path} ...")

    try:
        runner = ClientRunner(har=har_path)
        # Apply max optimization level to help fit models into Hailo8 resources
        runner.load_model_script("performance_param(compiler_optimization_level=max)\n")
        hef = runner.compile()
    except Exception as e:
        print(f"[ERROR] Failed to compile {quantized_model_har_name}: {e}")
        continue

    with open(out_path, "wb") as f:
        f.write(hef)

    print(f"[SUCCESS] Saved {out_path}")
