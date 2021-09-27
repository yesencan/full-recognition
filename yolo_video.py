import numpy as np
import time
from scipy import spatial
import cv2
from input_retrieval import *
import os
import json

#import libraries for car classification
import tensorflow.keras.backend as K
import scipy.io
from utils import load_model

begin = time.time()

list_of_vehicles = ["bicycle","car","motorbike","bus","truck", "train"]

FRAMES_BEFORE_CURRENT = 10  
inputWidth, inputHeight = 416, 416

LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU = parseCommandLineArguments()

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, #Image
		'Detected Vehicles: ' + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0xFF, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		os.system('clear')
		print("FPS:", num_frames)
		num_frames = 0
		start_time = current_time
	return start_time, num_frames


def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			#extract class name and confidence
			
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]

			#draws the rectangle
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])			
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			#Draw a green dot in the middle of the box
			cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)

def drawDetectionBoxes2(idxs, boxes, classIDs, confidences, frame, results):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			#extract class name and confidence
			if i in results:
				class_name = results[i][0]
				prob = results[i][1]
				
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]

				#draws the rectangle
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				# text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				# 	confidences[i])
				text = "{}: {}".format(class_name, prob)
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				#Draw a green dot in the middle of the box
				cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (0, 0xFF, 0), thickness=2)


def crop_cars(idxs, boxes, frame, classIDs, num_frames):
	# ensure at least one detection exists
	directory = "cars/frame{}".format(num_frames)
	os.mkdir(directory)
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			if x > 0 and y > 0 and LABELS[classIDs[i]] == "car":
				cropped_frame = frame[y:y+h, x:x+w].copy()
				cv2.imwrite("{}/car{}.jpg".format(directory,i), cropped_frame)

def predict(idxs, boxes, frame, classIDs):
	img_width, img_height = 224, 224
	model = load_model()

	cars_meta = scipy.io.loadmat('devkit/cars_meta2')
	class_names = cars_meta['class_names']  # shape=(1, 196)
	class_names = np.transpose(class_names)
	class_names2 = np.array([[i[0][0]] for i in class_names])
	best_results = {}
	results = {}
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			
			if x > 0 and y > 0:
				bgr_img = frame[y:y+h, x:x+w].copy()
				bgr_img = cv2.resize(bgr_img, (img_width, img_height), cv2.INTER_CUBIC)
				rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
				rgb_img = np.expand_dims(rgb_img, 0)
				preds = model.predict(rgb_img)
				prob = np.max(preds)
				class_id = np.argmax(preds)
				temp = np.concatenate((class_names2, preds.T), axis=1)
				temp = temp[np.argsort(temp[:,1].astype(np.float32))][::-1]
				
				results[i] = [list(temp[:,0]),list(temp[:,1])]
				best_results[i]= [class_names[class_id][0][0], '{:.4}'.format(prob)]
				

	return results, best_results
				
			
def initializeVideoWriter(video_width, video_height, videoStream):
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,(video_width, video_height), True)


def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame, results, all_results, frame_count):
	
	current_detections = {}
	vehicle_count_part2 = vehicle_count
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			if i in results:
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				
				centerX = x + (w//2)
				centerY = y+ (h//2)

				if (LABELS[classIDs[i]] in list_of_vehicles):
					current_detections[(centerX, centerY)] = vehicle_count 
					if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
						vehicle_count += 1

					ID = current_detections.get((centerX, centerY))
					# If there are two detections having the same ID due to being too close, 
					# then assign a new ID to current detection.
					if (list(current_detections.values()).count(ID) > 1):
						current_detections[(centerX, centerY)] = vehicle_count
						vehicle_count += 1 

					#Display the ID at the center of the box
					cv2.putText(frame, str(ID), (centerX, centerY),\
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
					
					temp = results[i]
					temp.append(frame_count)
					if ID in all_results:
						all_results[ID].append(temp)
					else:
						all_results[ID] = [temp]

	return vehicle_count_part2,vehicle_count, current_detections, all_results


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")


if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
#ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initalize a results list
all_results = {}
# Specifying coordinates for a default line 
x1_line = 0
y1_line = video_height//2
x2_line = video_width
y2_line = video_height//2

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count, frame_count = 0, 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
# loop over frames from the video file stream
while True:
	print("================NEW FRAME================")
	num_frames+= 1
	frame_count += 1
	print("FRAME:\t", num_frames)
	# Initialization for each iteration
	boxes, confidences, classIDs = [], [], []

	#Calculating fps each second
	start_time, num_frames = displayFPS(start_time, num_frames)
	# read the next frame from the file
	(grabbed, frame) = videoStream.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	# layerOutputs contains Numpy ndarray which can be used to plot the boxes
	layerOutputs = net.forward(ln)
	end = time.time()

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		# print(output)
		for i, detection in enumerate(output):
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			# print(detection)
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > preDefinedConfidence:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	# Draw detection box 
	results, best_results = predict(idxs, boxes, frame, classIDs)
	drawDetectionBoxes2(idxs, boxes, classIDs, confidences, frame, best_results)
	vehicle_count_part2, vehicle_count, current_detections, all_results = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections,
																						frame, results, all_results, frame_count)
	print("vehicles in frame    :   ")
	print(vehicle_count-vehicle_count_part2)
	# Display Vehicle Count if a vehicle has passed the line 
	displayVehicleCount(frame, vehicle_count)

    # write the output frame to disk
	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	
	# Updating with the current frame detections
	previous_frame_detections.pop(0) #Removing the first frame from the list
	# previous_frame_detections.append(spatial.KDTree(current_detections))
	previous_frame_detections.append(current_detections)

# release the file pointers
print("[INFO] cleaning up...")
print("No of cars identified : ", vehicle_count)
writer.release()
videoStream.release()

end = time.time()
print(f"Total runtime of the program is {end - begin}")


try:
    json.dump(all_results, open("results.txt", "w"))
  
except:
    print("Something went wrong")