import tensorflow as tf
import utils
import os
import numpy as np

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)
graph1 = tf.Graph()

with graph1.as_default():
    with tf.Session() as sess1:
        sess1.run(tf.global_variables_initializer())
        tf.saved_model.loader.load(sess1, [tf.saved_model.tag_constants.SERVING], "/tmp/model/vgg193/")
        # img_placeholder = graph1.get_tensor_by_name('img_out_placeholder:0')
        vgg_model = graph1.get_tensor_by_name('vgg_model:0')
        prob = sess1.run(vgg_model, {
            "input_image": batch
        })

        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        prob = sess1.run(vgg.prob, feed_dict=feed_dict)
        print(prob)

        utils.print_prob(prob[0], './synset.txt')
        utils.print_prob(prob[1], './synset.txt')