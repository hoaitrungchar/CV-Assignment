import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from Inpainting.CV2_Inpaint import process_video_frame
from VideoObjectSegmentation.segmentvideo import segmentation_each_frame
def parse_arguments():
    """
    Parse command line arguments for the video inpainting pipeline.
    """
    parser = argparse.ArgumentParser("", add_help=True)
    
    # Integrated arguments from second parser
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--replacement_image', type=str,
                       help='Path to image used for replacing masked areas.')
    parser.add_argument('--output', type=str,
                       help='Path to folder to save output video')
    return parser.parse_args()

class VideoPipeline:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.replacement_image = None
        if args.replacement_image:
            self.replacement_image = cv2.imread(args.replacement_image)
            if self.replacement_image is None:
                raise ValueError(f"Could not load replacement image: {args.replacement_image}")
        self.load_models()

    def load_models(self):
        """
        Load all required models based on command line arguments.
        """
        try:
            from ultralytics import YOLO
            self.detector = YOLO('ObjectDetection\OD_best.pt')
        except ImportError:
            raise ImportError("YOLO not installed. Install with: pip install ultralytics")
        

        try:
            from ultralytics import YOLO
            self.segmentation = YOLO('ObjectSegmentation\OS_best.pt')
        except ImportError:
            raise ImportError("YOLO not installed. Install with: pip install ultralytics")
       

    def detect_objects(self, frame):
        """
        Detect objects using selected detection model.
        """
        results = self.detector(frame)
        boxes= results[0].boxes.xyxy.cpu().numpy()
        return boxes if boxes.size>0 else None

    def segmentation_frame(self, frame):
        """
        Detect objects using selected detection model.
        """
        results = self.segmentation(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for result in results:
            if result.masks is not None:
                for segment, class_id in zip(result.masks.xy, result.boxes.cls):
                    # Check if the class is billboard
                    # Note: Update class_id based on your model's class mapping
                    if result.names[int(class_id)].lower() == "billboard":
                        # Convert segment points to integer
                        segment = segment.astype(np.int32)
                        # Fill the mask
                        cv2.fillPoly(mask, [segment], 255)
        return mask


    def process_video(self):
        """
        Main video processing pipeline.
        """
        cap = cv2.VideoCapture(self.args.input)
        if not cap.isOpened():
            raise IOError(f"Cannot open input video: {self.args.input}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(
            self.args.output,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            detection = self.detect_objects(frame)
            
            if detection is None:
                continue
            mask=self.segmentation_frame(frame)
            list_frame=[]
            while True:
                ret, frame=cap.read()
                if not ret:
                    break
                list_frame.append(frame)

            list_mask=segmentation_each_frame(list_frame,mask)
            list_mask = [mask]+list_mask
            for frameprocessed,maskprocessed in list(zip(list_frame,list_mask)):
                inpainted_frame = process_video_frame(frameprocessed, self.replacement_image, maskprocessed)
            out.write(inpainted_frame)

        cap.release()
        out.release()
        print(f"Processing complete. Output saved to {self.args.output}")

def main():
    args = parse_arguments()
    pipeline = VideoPipeline(args)
    pipeline.process_video()

if __name__ == "__main__":
    main()