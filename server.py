import grpc
import numpy as np
import cv2
import logging, sys, time

from proto import server_pb2
#from darknet import detector as darknet
from cv_util import plot
from cv_util import message
from cv_util import boundbox

sys.path.insert(0, './lib')
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.visualize import infer_image, load_model
from datasets.factory import get_imdb
from datasets.viz import draw_scene_graph, viz_scene_graph

import tensorflow as tf
#from multiprocessing import Queue, Process

import pickle

ENDPOINT = "[::]:9010"
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# Scene-graph
NETWORK_NAME = "dual_graph_vrd_final"
WEIGHT_NAME = "model/new_model.ckpt"
NR_CLASSES = 151
NR_PREDICATES = 51

CFG_FILE_SCENE = 'experiments/cfgs/sparse_graph.yml'
cfg.TEST.INFERENCE_ITER = 2
cfg.GPU_ID = 0
cfg_from_file(CFG_FILE_SCENE)

config = tf.ConfigProto()
config.allow_soft_placement = True
get_imdb("/home/ubuntu/scene-graph-TF-release/data/vg/mini_VG-SGG", "/home/ubuntu/scene-graph-TF-release/data/vg/mini_imdb_1024.h5", "/home/ubuntu/scene-graph-TF-release/data/vg/mini_proposals.h5")

QUEUE_SIZE = 10

class WorkerServicer(server_pb2.BetaWorkerServicer):
    def __init__(self):
        # Detection engine
        #self.query_queue_d, self.result_queue_d = Queue(QUEUE_SIZE), Queue(QUEUE_SIZE)
        #self.p_detect = Process(target=darknet.worker, args=(self.query_queue_d, self.result_queue_d, ))
        #self.p_detect.start()
        time.sleep(10)
        (self.sess, self.net, self.inputs) = load_model(NETWORK_NAME, WEIGHT_NAME, NR_CLASSES, NR_PREDICATES)

        saver = tf.train.Saver()
        saver.save(self.sess, 'model/my_test_model', global_step=1000)

    def RunTask(self, request, context):
        print 'Recv task {}'.format(request.seqId)
        
        img_array = np.asarray(bytearray(request.rawImage), dtype=np.uint8)
        #cvmat = cv2.imdecode(img_array, cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img = cv2.resize(img, (720, 720))

        #self.query_queue_d.put(message.Query(img))
        #bb_list = self.result_queue_d.get().bb_list

        bb_list = []        
        for bb_p in request.bb_list:
            bb_list.append(boundbox.BB(bb_p.xmin, bb_p.ymin, bb_p.xmax, bb_p.ymax, bb_p.label, bb_p.confidence, bb_p.object_uuid))
        
        #bb_list_str = pickle.dumps(bb_list)

        #bb_list = self.detector.detect(img)

        #plot.plot_bb(img, bb_list)
        #cv2.imshow('image', img)
        #cv2.waitKey(0)

        # bb_list to proposal
        box_proposals = []
        for bb in bb_list:
            box_proposals.append([bb.xmin, bb.ymin, bb.xmax, bb.ymax])
        box_proposals = np.asarray(box_proposals)
        print box_proposals

        relations = []
        for i in range(box_proposals.shape[0]):
            for j in range(box_proposals.shape[0]):
                if i != j:
                    relations.append([i, j])
        #relations = [[0, 1], [1, 0], [0, 2], [2, 0]]

        start = time.time()
        out_dict, inference_iter = infer_image(img, box_proposals, relations, 
            self.sess, self.net, self.inputs)
        scene_delay = time.time() - start
        print 'scene graph takes {}'.format(scene_delay)

        #print 'process output'
        #sg_entry = out_dict[inference_iter]
        #boxes, cls_score, rel_score = sg_entry['boxes'], sg_entry['scores'], sg_entry['relations']
        
        #cls_pred = np.argmax(cls_score, 1)
        #rel_pred_mat = np.argmax(rel_score, 2)
        #rel_pred, all_rels = [], []

        #for i in xrange(rel_pred_mat.shape[0]):
        #    for j in xrange(rel_pred_mat.shape[1]):
        #        if i == j or [i, j] in all_rels:
        #            continue
        #        rel_pred.append([i, j, rel_pred_mat[i, j], 1])
        #        all_rels.append([i, j])

        #rel_pred = np.array(rel_pred)
        #if rel_pred.size > 0:
        #    pred_inds = rel_pred[:, :2].ravel()
        #    img = img[:, :, (2,1,0)]
        #    viz_scene_graph(img, boxes, cls_pred, pred_inds, rel_pred, preprocess=False)

        print 'send back result'
        out_str = pickle.dumps(out_dict)
		
        outs = server_pb2.Result(seqId=request.seqId, execTime=0, scene_data=out_str)
        return outs


def main():
    FMT = '%(asctime)s %(name)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FMT)
    logger = logging.getLogger(0)
    
    # Start server
    workerServer = WorkerServicer()
    server = server_pb2.beta_create_Worker_server(workerServer)
    server.add_insecure_port(ENDPOINT)
	
    logger.info("Worker {} listening on {}".format(0, ENDPOINT))
    server.start()

    # Block
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    main()
