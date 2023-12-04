global model
global conf

model = something
conf = something_else

def _display_detected_frames( image, is_display_tracking=None, tracker=None):
    global conf
    global model
    image = cv2.resize(image, (720, int(720*(9/16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
    # Return the processed frame to callback
    res_plotted = res[0].plot()
    return res_plotted

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    is_display_tracker, tracker = display_tracker_options()
    image = _display_detected_frames(img, is_display_tracker, tracker)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)
