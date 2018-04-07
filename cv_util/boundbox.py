class BB:
	def __init__(self, xmin, ymin, xmax, ymax, label, confidence, obj_id='na'):
		self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
		self.label, self.confidence, self.obj_id = label, confidence, obj_id
		#self.pt = None
		self.is_track = True

	def is_contain(self, pt):
		if pt[0] < self.xmin or pt[0] > self.xmax or pt[1] < self.ymin or pt[1] > self.ymax:
			return False
		else:
			return True

def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA.xmin, boxB.xmin)
	yA = max(boxA.ymin, boxB.ymin)
	xB = min(boxA.xmax, boxB.xmax)
	yB = min(boxA.ymax, boxB.ymax)

	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)

	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA.xmax - boxA.xmin + 1) * (boxA.ymax - boxA.ymin + 1)
	boxBArea = (boxB.xmax - boxB.xmin + 1) * (boxB.ymax - boxB.ymin + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

# TODO: penalize bb_list from tracking
def merge_bb_lists(bb_list_old, bb_list_new, iou_thrd):
	bb_list = bb_list_old + bb_list_new
	bb_list.sort(key = lambda bb: bb.confidence, reverse=True)
	i = 0
	while i < len(bb_list)-1:
		bb0 = bb_list[i]
		j = i + 1
		while j < len(bb_list):
			bb1 = bb_list[j]
			iou = bb_iou(bb0, bb1)
			if iou >= iou_thrd:
				del bb_list[j]
			j += 1
		i += 1
	return bb_list

def combine_all_bb_list(all_bb_list_map):
	all_bb_list_label_map = {"vehicle": [], "person": []}
	for detector_name, bb_list in all_bb_list_map.iteritems():
		for bb in bb_list:
			if bb.label == 'car' or bb.label == 'truck' or  bb.label == 'bus':
				bb.label = 'vehicle'
				all_bb_list_label_map[bb.label].append(bb)
			elif bb.label == 'pedestrian' or bb.label == 'cyclist,' or  bb.label == 'person':
				bb.label = 'person'
				all_bb_list_label_map[bb.label].append(bb)
	# NMS
	for label, bb_list in all_bb_list_label_map.iteritems():
		all_bb_list_label_map[label] = confidence_nms(bb_list)

	# Fuse bb_list
	bb_list_combined = []
	for label in all_bb_list_label_map:
		bb_list_combined += all_bb_list_label_map[label]

	return bb_list_combined

def confidence_nms(bb_list):
	bb_list.sort(key=lambda bb: bb.confidence, reverse=True)
	survivor_list = []
	while len(bb_list) > 0:
		bb_survivor = bb_list[0]
		survivor_list.append(bb_survivor)
		del bb_list[0]
		for i in range(len(bb_list)-1, -1, -1):
			bb = bb_list[i]
			iou = bb_iou(bb_survivor, bb)
			if iou > 0.3:
				print 'delete: label/{} conf/{} ({},{},{},{}) // label/{} conf/{} ({},{},{},{}) // iou{}'.format(
					bb_list[i].label, bb_list[i].confidence, bb_list[i].xmin, bb_list[i].xmax, bb_list[i].ymin, bb_list[i].ymax, 
					bb_survivor.label, bb_survivor.confidence, bb_survivor.xmin, bb_survivor.xmax, bb_survivor.ymin, bb_survivor.ymax, iou)
				del bb_list[i]
	return survivor_list
