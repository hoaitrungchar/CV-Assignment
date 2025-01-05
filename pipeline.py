import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from Inpainting.CV2_Inpaint import process_video_frame
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
        return results[0].boxes.xyxy.cpu().numpy()
    

    def segmentation_frame(self, frame):
        """
        Detect objects using selected detection model.
        """
        results = self.segmentation(frame)


    def inpaint_frame(self, frame, mask):
        """
        Inpaint frame using either replacement image or OpenCV inpainting.
        """
        if self.replacement_image is not None:
            # Resize replacement image to match frame size if necessary
            if self.replacement_image.shape != frame.shape:
                self.replacement_image = cv2.resize(
                    self.replacement_image, 
                    (frame.shape[1], frame.shape[0])
                )
            
            # Create inverted mask for original frame
            inv_mask = cv2.bitwise_not(mask)
            
            # Extract masked regions from replacement image and original frame
            replacement_masked = cv2.bitwise_and(
                self.replacement_image, 
                self.replacement_image, 
                mask=mask
            )
            frame_masked = cv2.bitwise_and(
                frame, 
                frame, 
                mask=inv_mask
            )
            
            # Combine the images
            return cv2.add(frame_masked, replacement_masked)
        else:
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

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

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            detections = self.detect_objects(frame)
            self.segmentation_frame(frame)
            
            # Create mask from detections
            masks = []
            for box in detections:
                mask, _, _ = self.segmentation.predict(
                    box=box,
                    multimask_output=False
                )
                masks.append(mask)

            # Combine masks if multiple objects detected
            if masks:
                combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            else:
                combined_mask = np.zeros((height, width), dtype=np.uint8)

            # Inpaint frame
            inpainted_frame = process_video_frame(frame, )
            out.write(inpainted_frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames")

        cap.release()
        out.release()
        print(f"Processing complete. Output saved to {self.args.output}")

def main():
    args = parse_arguments()
    pipeline = VideoPipeline(args)
    pipeline.process_video()

if __name__ == "__main__":
    main()