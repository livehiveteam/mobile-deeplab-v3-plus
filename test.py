import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile

class Model(object):
    def __init__(self, model_filepath):
        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph = self.graph)

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        # nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        # for node in nodes:
        #     print(node)

        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        print(names)

        # Define input tensor
        self.input = tf.placeholder(np.float32, shape = [None, 513, 513, 3], name='Input')

        tf.import_graph_def(graph_def, {'Input': self.input})
        print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)

        '''
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        '''

    def predict(self, data):
        output_tensor = self.graph.get_tensor_by_name("import/Output:0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: data})

        return output


tf.reset_default_graph()
#MODEL_PATH = "weights/deeplab_v3_plus_mnv2_decoder_513.pb"
MODEL_PATH = "weights/frozen_model.pb"
model = Model(model_filepath=MODEL_PATH)

TEST_DIR = 'datasets/picsart/test/'
#TEST_DIR = 'datasets/people_segmentation/images/'

RESULT_DIR = 'results/'
INPUT_SIZE = 513
#MEAN_UINT_PIXEL = [127.5, 127.5, 127.5]
test_imgs = os.listdir(TEST_DIR)

for i, img_path in enumerate(test_imgs):
    img_name = img_path
    img_name = img_name.split('.')[0]
    img_path = os.path.join(TEST_DIR, img_path)

    test_img = cv2.imread(img_path)
    test_img = test_img[:,:,::-1]
    test_img = cv2.resize(test_img, (INPUT_SIZE, INPUT_SIZE))
    test_img = test_img[np.newaxis, ...]

    test_img = (2.0 / 255.0) * test_img - 1.0

    output = model.predict(test_img)[0]
    output = np.squeeze(output, axis=-1)

    output[output > 0.5] = 255

    plt.imshow(test_img[0])
    plt.imshow(output, alpha=0.4)
    plt.show()

    plt.savefig(RESULT_DIR + img_name + '.png')

    if i > 9:
        break
