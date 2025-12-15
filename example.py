"""
Example script demonstrating Intel RealSense camera integration.

This script shows how to:
1. Import the Intel RealSense library
2. Initialize and configure a RealSense pipeline
3. Capture frames from a connected Intel RealSense camera
4. Display frame dimensions as a simple demo
"""

import pyrealsense2 as rs
import numpy as np


def main():
    """Main function to demonstrate Intel RealSense camera usage."""
    
    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config object
    config = rs.config()
    
    # Configure the pipeline to stream color and depth
    # Color stream: 640x480 resolution at 30 FPS
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Depth stream: 640x480 resolution at 30 FPS
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # Start streaming
        print("Starting Intel RealSense camera...")
        pipeline.start(config)
        print("Camera initialized successfully!")
        
        # Capture a few frames to allow auto-exposure to stabilize
        for _ in range(5):
            pipeline.wait_for_frames()
        
        # Get frames
        frames = pipeline.wait_for_frames()
        
        # Get color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            raise RuntimeError("Could not get frames from camera")
        
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Print frame dimensions
        print("\n=== Frame Information ===")
        print(f"Color frame dimensions: {color_image.shape}")
        print(f"Depth frame dimensions: {depth_image.shape}")
        print(f"Color frame size: {color_image.shape[1]}x{color_image.shape[0]}")
        print(f"Depth frame size: {depth_image.shape[1]}x{depth_image.shape[0]}")
        
        # Get depth scale for distance calculations
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")
        
        # Sample depth at center of image
        center_x = depth_image.shape[1] // 2
        center_y = depth_image.shape[0] // 2
        center_depth = depth_image[center_y, center_x] * depth_scale
        print(f"Distance at center point: {center_depth:.3f} meters")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Intel RealSense camera is connected")
        print("2. pyrealsense2 library is installed (pip install pyrealsense2)")
        print("3. You have necessary permissions to access the camera")
        
    finally:
        # Stop streaming
        pipeline.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
