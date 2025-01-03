from ultralytics import YOLO
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="./save_dir",
                                            help="Checkpoint of Model")
# parser.add_argument("--save_dir", type=str, default="./save_dir",
#                                             help="Folder to save the checkpoints and training log")
# parser.add_argument("--resume", type=str, const=True, default="", nargs="?",
#                                                     help="resume from the save_dir or checkpoint")
parser.add_argument("--cfg_path", type=str, default="./configs/inpainting_debug.yaml",
                                                                    help="Configs of yaml file")
parser.add_argument("--device", type=int, default="./configs/inpainting_debug.yaml",
                                                                    help="Device GPU to train")
args = parser.parse_args()
configs={}
for key in vars(args):
    configs[key] = getattr(args, key)

print(configs)
# Load a model
model = YOLO(configs["ckpt_path"])


# Train the model
train_results = model.train(
    data=configs["cfg_path"],  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=configs["device"],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export()  # return path to exported model
print(path)