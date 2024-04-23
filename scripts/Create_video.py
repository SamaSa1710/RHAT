import cv2
import os

def create_video_from_images(folder_path, output_video_path, fps=60):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The specified folder '{folder_path}' does not exist.")
        return

    # Get the list of image files in the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

    # Sort the images by filename
    images.sort()

    # Determine the size of the images (assumes all images have the same size)
    first_image_path = os.path.join(folder_path, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape
    temp_name = str(width)+"x"+str(height)
    output_video_path = output_video_path + temp_name +".mp4" 
    print(output_video_path)
    # Initialize video writer
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Write each image to the video
    for image in images:
        image_path = os.path.join(folder_path, image)
        img = cv2.imread(image_path)
        video_writer.write(img)

    # Release the video writer
    video_writer.release()

def display_video(video_path):
    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video is successfully opened
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Display the video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_folder_path' with the path to your folder containing images
input_folder_path = r'/home/rtx3090ti-12f-1-vtrg/Desktop/bubble/RethinkVSRAlignment/data/vimeo90k/GT/00095/0889'
name = input_folder_path.split("\\")[-1]
# Replace 'output_video.mp4' with the desired name for the output video file

output_video_path = r"/home/rtx3090ti-12f-1-vtrg/Desktop/bubble/RethinkVSRAlignment/results"
output_video_path = os.path.join(output_video_path, name)
# Create video from images
create_video_from_images(input_folder_path, output_video_path, 2)
print("completete")
# # Display the created video
# display_video(output_video_path)