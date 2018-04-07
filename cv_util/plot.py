import cv2

def plot_frame(img, name='test', wait_time=10):
    cv2.imshow(name, img)
    cv2.waitKey(wait_time)

def plot_bb(img, bb_list, color=(0,255,0)):
    for bb in bb_list:
        if bb.is_track:
            cv2.rectangle(img, (bb.xmin,bb.ymin), (bb.xmax,bb.ymax), color, 2)
            cv2.rectangle(img, (bb.xmin,bb.ymin-20), (bb.xmax,bb.ymin), (125,125,125), -1)
            cv2.putText(img,bb.label + ' : %.2f' % bb.confidence, (bb.xmin+5,bb.ymin-7), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)
