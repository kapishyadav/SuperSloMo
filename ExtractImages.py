import cv2
import time
import os

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if count == 200:
            cap.release()
            break
        if ret == False:
            cap.release()
            break
        # Write the results back to output location.
        frame = cv2.resize(frame, (512,512))
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1




input_path = "DeepVideoDeblurring_Dataset_Original_High_FPS_Videos/original_high_fps_videos/"
output_path = "ExtractedImages/"
filenames = os.listdir(input_path)

for video in filenames:
	print(video)
	input_loc = input_path+video
	if not os.path.exists("ExtractedImages/"+video):
		os.makedirs("ExtractedImages/"+video)
	output_loc = "ExtractedImages/"+video
	video_to_frames(input_loc, output_loc)