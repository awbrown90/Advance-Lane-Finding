# Created By Aaron Brown
# January 20, 2017
# Udacity Self-Driving Car Nanodegree

# Main class to generate video output

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
from LineTracker import Tracker

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "camera_cal/calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# (Selected_Video) Preset Video parameters (there are 3 videos, project_video: 1, challenge_video:2, and harder_challenge_video:3)
# WHICH VIDEO DO YOU WANT TO WORK ON?
Selected_Video = 3

if Selected_Video == 1:
	# set up thresholding for the appropriate video to get best pixels of interest
	# For these videos we are focusing on using x/y gradients and hsv and hls color spaces, which seemed most useful for finding important pixels
	gradx_thresh = (25,255) # gradient x threshold
	grady_thresh = (10,255) # gradient y threshold
	schannel_thresh = (100,255) # gradient s channel threshold
	vchannel_thresh = (200,255) # gradient v channel threshold

	# preset percentages to transform road image, want road lines to be as parallel as possible for LineTracker
	bot_width = .76 # percent of bottom trapizoid height #.76
	mid_width = .1 # percent of middle trapizoid height .1 # .3
	height_pct = .63 # percent for trapizoid height .63
	bottom_trim = .935 # percent from top to bottom to avoid car hood 

	# Set up the overall class to do all the tracking
	curve_centers = Tracker(Mycenter_dis = .25*1280, Mywindow_width = 25, Mywindow_height = 40, Mypadding = 25, Myslide_res = 5, Myframe_ps = 25, Mycapture_height = 720, \
					   	    My_ym = 30/720, My_xm = 4/384, Myline_dist = 10)

	# set output and input video names
	Output_video = 'output1_tracked.mp4'
	Input_video = 'project_video.mp4'

elif Selected_Video == 2:
	# set up thresholding for the appropriate video to get best pixels of interest
	# For these videos we are focusing on using x/y gradients and hsv and hls color spaces, which seemed most useful for finding important pixels
	gradx_thresh = (20,255) # gradient x threshold
	grady_thresh = (20,255) # gradient y threshold
	schannel_thresh = (8,255) # gradient s channel threshold
	vchannel_thresh = (160,255) # gradient v channel threshold

	# preset percentages to transform road image, want road lines to be as parallel as possible for LineTracker
	bot_width = .76 # percent of bottom trapizoid height #.76
	mid_width = .25 # percent of middle trapizoid height .1 # .3
	height_pct = .7 # percent for trapizoid height .63
	bottom_trim = .935 # percent from top to bottom to avoid car hood 

	# Set up the overall class to do all the tracking
	curve_centers = Tracker(Mycenter_dis = .25*1280, Mywindow_width = 25, Mywindow_height = 40, Mypadding = 25, Myslide_res = 5, Myframe_ps = 25, Mycapture_height = 720, \
					   	    My_ym = 15/720, My_xm = 4/384, Myline_dist = 3)

	# set output and input video names
	Output_video = 'output2_tracked.mp4'
	Input_video = 'challenge_video.mp4'

elif Selected_Video == 3:

	# set up thresholding for the appropriate video to get best pixels of interest
	# For these videos we are focusing on using x/y gradients and hsv and hls color spaces, which seemed most useful for finding important pixels
	gradx_thresh = (20,255) # gradient x threshold
	grady_thresh = (20,255) # gradient y threshold
	schannel_thresh = (240,255) # gradient s channel threshold
	vchannel_thresh = (240,255) # gradient v channel threshold

	# preset percentages to transform road image, want road lines to be as parallel as possible for LineTracker
	bot_width = .76 # percent of bottom trapizoid height #.76
	mid_width = .3 # percent of middle trapizoid height .1 # .3
	height_pct = .73 # percent for trapizoid height .63
	bottom_trim = .935 # percent from top to bottom to avoid car hood 

	# Set up the overall class to do all the tracking
	curve_centers = Tracker(Mycenter_dis = .25*1280, Mywindow_width = 25, Mywindow_height = 40, Mypadding = 25, Myslide_res = 5, Myframe_ps = 25, Mycapture_height = 720, \
					   	    My_ym = 10/720, My_xm = 4/384, Myline_dist = 3)

	# set output and input video names
	Output_video = 'output3_tracked.mp4'
	Input_video = 'harder_challenge_video.mp4'


# Useful functions for producing the binary pixel of interest images to feed into the LaneTracker Algorithm
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # Apply threshold
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output =  np.zeros_like(absgraddir)
        # Apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])  ] = 1

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hsv[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])  ] = 1

	output = np.zeros_like(s_channel)
	output[(s_binary == 1) & (v_binary == 1)] = 1
	return output

def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	if(((center-width/2) < img_ref.shape[1]) & ((center+width/2) > 0)):
		output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(img_ref.shape[1],int(center+width))] = 1
	return output

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)   

def process_image(image):

	# undistort the image 
	img = cv2.undistort(image,mtx,dist,None,mtx)

	# process image and generate binary pixel of interests
	preprocessImage = np.zeros_like(img[:,:,0])
	gradx = abs_sobel_thresh(img, orient='x', thresh=gradx_thresh) 
	grady = abs_sobel_thresh(img, orient='y', thresh=grady_thresh) 
	c_binary = color_threshold(img, sthresh=schannel_thresh, vthresh=vchannel_thresh)
	preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] = 255
	
	# work on defining perspective transformation area
	img_size = (img.shape[1],img.shape[0])
	src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
	offset = img_size[0]*.33
	dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset ,img_size[1]]])

	# perform the transform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)

	# for debugging just to get the binary image
	#warped = np.array(cv2.merge((warped,warped,warped)),np.uint8)
	#return warped

	# set up the analysis for speed measurement
	spd_offset = img_size[0]*(1.0-bot_width)/2
	spd_dst = np.float32([[spd_offset, 0], [img_size[0]-spd_offset, 0],[img_size[0]-spd_offset, img_size[1]], [spd_offset ,img_size[1]]])
	spd_M = cv2.getPerspectiveTransform(src,spd_dst)
	# generate gray scale image to use for tracking speed in LineTracker, using template matching
	warped_mono = cv2.warpPerspective(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),spd_M,img_size,flags=cv2.INTER_LINEAR)
	curve_centers.speed_track(warped_mono)

	# find the best line centers based on the binary pixel of interest input
	frame_centers = curve_centers.track_line(warped)
	# need these parameters to draw the graphic overlay illustraing the window convolution matching
	window_width = curve_centers.window_width 
	window_height = curve_centers.window_height
	# points used for graphic overlay 
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)

	# points used to find the left and right lanes
	rightx = []
	leftx = []

	res_yvals = np.arange(warped.shape[0]-(window_height+window_height/2),0,-window_height)
	
	for level in range(1,len(frame_centers)):
		l_mask = window_mask(window_width,window_height,warped,frame_centers[level][0],level)
		r_mask = window_mask(window_width,window_height,warped,frame_centers[level][1],level)
		# add center value found in frame to the list of lane points per left,right
		leftx.append(frame_centers[level][0])
		rightx.append(frame_centers[level][1])
		# fill in graphic points here if pixels fit inside the specificed window from l/r mask
		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

	# drawing the graphic overlay to represents the results found for tracking window centers
	template = np.array(r_points+l_points,np.uint8)
	zero_channel = np.zeros_like(template)
	template = np.array(cv2.merge((zero_channel,zero_channel,template)),np.uint8)
	warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
	graphic_measure = cv2.addWeighted(warpage, 0.2, template, 0.75, 0.0)
	

	# fit the lane boundaries to the left,right center positions found
	yvals = range(0,warped.shape[0])

	left_fit = np.polyfit(res_yvals, leftx, 3)
	left_fitx = left_fit[0]*yvals*yvals*yvals + left_fit[1]*yvals*yvals + left_fit[2]*yvals+left_fit[3]
	left_fitx = np.array(left_fitx,np.int32)
	
	right_fit = np.polyfit(res_yvals, rightx, 3)
	right_fitx = right_fit[0]*yvals*yvals*yvals + right_fit[1]*yvals*yvals + right_fit[2]*yvals+right_fit[3]
	right_fitx = np.array(right_fitx,np.int32)

	# used to find center curve
	curve_xpts = [(right_fitx[0]+left_fitx[0])/2,(right_fitx[len(right_fitx)/2]+left_fitx[len(left_fitx)/2])/2,(right_fitx[-1]+left_fitx[-1])/2]
	curve_ypts = [yvals[0],yvals[(int)(len(yvals)/2)],yvals[-1]]
	curve_fit = np.polyfit(curve_ypts, curve_xpts, 2)
	curve_fitx = curve_fit[0]*yvals*yvals + curve_fit[1]*yvals+ curve_fit[2]

	# used to format everything so its ready for cv2 draw functions
	left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	curve_pts = np.array([list(zip(curve_fitx,yvals))],np.int32)

	# draw lane lines, middle curve, road background on two different blank overlays
	road = np.zeros_like(template)
	road_bkg = np.zeros_like(template)
	cv2.fillPoly(road,[left_lane],color=[9, 67, 109])
	cv2.fillPoly(road,[right_lane],color=[9, 67, 109])
	cv2.polylines(road,[curve_pts],isClosed=False, color=[5, 176, 249], thickness=3)
	for horz_line_y in curve_centers.horz_lines:
		cv2.line(road,(left_fitx[(int)(horz_line_y)],(int)(horz_line_y)),(right_fitx[(int)(horz_line_y)],(int)(horz_line_y)),color=[5, 176, 249], thickness=3)

	cv2.fillPoly(road_bkg,[inner_lane],color=[38, 133, 197])

	# after done drawing all the marking effects, warp back image to its orginal perspective.
	# Note for the two different overlays, just seperating road_warped and road_warped_bkg to get two different alpha values, its just for astetics...
	road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

	# merging all the different overlays, basically make things look pretty!
	lane_template = np.array(cv2.merge((road_warped[:,:,2],road_warped[:,:,2],road_warped[:,:,2])),np.uint8)
	bkg_template = np.array(cv2.merge((road_warped_bkg[:,:,2],road_warped_bkg[:,:,2],road_warped_bkg[:,:,2])),np.uint8)
	base = cv2.addWeighted(img, 1.0, bkg_template, -0.6, 0.0)
	base = cv2.addWeighted(base, 1.0, road_warped_bkg, 0.6, 0.0)
	base = cv2.addWeighted(base, 1.0, lane_template, -1.8, 0.0)
	result = cv2.addWeighted(base, 1.0, road_warped, 0.9, 0.0)
	#return result
	
	# calcuate the middle line curvature
	ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dimension
	xm_per_pix = curve_centers.xm_per_pix # meteres per pixel in x dimension
	curve_fit_cr = np.polyfit(np.array(curve_ypts,np.float32)*ym_per_pix, np.array(curve_xpts,np.float32)*xm_per_pix, 2)
	curverad = ((1 + (2*curve_fit_cr[0]*curve_ypts[1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])
	curve_centers.curvatures.append(curverad)
	curverad = curve_centers.SmoothCurve()

	# calculate the speed of the car
	speed = curve_centers.CalculateSpeed(metric_mode = False)
	
	# calculate the offset of the car on the road
	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
	side_pos = 'left'
	if center_diff <= 0:
		side_pos = 'right'
	
	# add text backdrop
	txt_bkg = np.zeros_like(result)
	txt_bkg_pts = np.array([(0,0),((int)(result.shape[1]*.5),0),((int)(result.shape[1]*.5),(int)(result.shape[0]*.25)),(0,(int)(result.shape[0]*.25))])
	cv2.fillPoly(txt_bkg,[txt_bkg_pts],color=[200, 200, 200])
	result = cv2.addWeighted(result, 1.0, txt_bkg, -1.0, 0.0)
	# draw the text showing curvature, offset, and speed
	cv2.putText(result,'Radius of Curvature = '+str(round(curverad,3))+'(m)',(50,50) , cv2.FONT_HERSHEY_SIMPLEX, 1,(5, 176, 249),2)
	cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100) , cv2.FONT_HERSHEY_SIMPLEX, 1,(5, 176, 249),2)
	cv2.putText(result,'Estimated Speed is '+str(abs(round(speed,3)))+' MPH',(50,150) , cv2.FONT_HERSHEY_SIMPLEX, 1,(5, 176, 249),2)

	# insert graphic overlay map
	graphic_bkg = np.zeros_like(result)
	# scale the graphic measure that we generated near the start for finding window centers, by some constant factor in both axis
	g_scale = 0.4
	graphic_overlay = cv2.resize(graphic_measure, (0,0), fx=g_scale, fy=g_scale) 
	g_xoffset = result.shape[1]-graphic_overlay.shape[1]
	# overlay the graphic measure in the result image at the top right corner
	result[:graphic_overlay.shape[0], g_xoffset:g_xoffset+graphic_overlay.shape[1]] = graphic_overlay

	return result

video_output = Output_video
clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
video_clip.write_videofile(video_output, audio=False)