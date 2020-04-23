import glob
import cv2
import numpy as np
#from goto import with_goto

def get_dynamic_image(frames, normalized=True):
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    num_channels = frames[0].shape[2] 
    # to find number of components : A R G B. shape gives the details of the image returns 3 here in avenue dataset
    channel_frames = _get_channel_frames(frames, num_channels) 
    # splits the frame by channels
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images)) 
    # opposite of split. merges the different channels
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def _get_channel_frames(iter_frames, num_channels):
    """ Takes a list of frames and returns a list of frame lists split by channel. """
    frames = [[] for channel in range(num_channels)] 
    #here 3 []

    for frame in iter_frames:
        for channel_frames, channel in zip(frames, cv2.split(frame)): 
	    # zip used for mapping. example (1,2,3,4) and (4,5,6,7) to [(1,4),(2,5),(3,6),(4,7)]
            channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
    for i in range(len(frames)):
        frames[i] = np.array(frames[i])
    return frames


def _compute_dynamic_image(frames):
    """ For computing dynamic image"""
    num_frames, h, w, depth = frames.shape# number =20

    # Compute the coefficients for the frames.
    coefficients = np.zeros(num_frames)
    #print(num_frames)
    for n in range(num_frames):
        cumulative_indices = np.array(range(n, num_frames)) + 1
        coefficients[n] = np.sum(((2*cumulative_indices) - num_frames-1) / cumulative_indices) #1/t B_t

    # Multiply frames by the coefficients and sum the result.
    x1 = np.expand_dims(frames, axis=0) #v_t
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
    result = x1 * x2 #B_t * V_t
    return np.sum(result[0], axis=0).squeeze()


""" driver program for making frames to get dynamic image"""
pic=cv2.VideoCapture('/home/aj/ab/Avenue_Dataset/Avenue Dataset/training_videos/16.avi') # 0 for taking pictures from camera
break_flag = False 
counter=762
# for exiting from the loop in case of end of the video
while(pic.isOpened()):
	i=0
	for i in range(20):
	        #  create frames from video source
		ret,frame=pic.read()
		if ret==False:
			break_flag = True
			break
			# goto .br
		cv2.imwrite(str(i)+'.jpg',frame)
		i+=1
	frames = glob.glob('*.jpg')
	frames = [cv2.imread(f) for f in frames]

	dyn_image = get_dynamic_image(frames, normalized=True) # to get dynamic image from the current number of extracted frames.
	cv2.imwrite('dynamic_image'+str(counter)+'.png', dyn_image)
	counter=counter+1
	cv2.imshow('', dyn_image)
	cv2.waitKey()
	if break_flag:
		break

#label .br
pic.release()
cv2.destroyAllWindows()
