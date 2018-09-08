import tensorflow as tf

import scipy.misc
import numpy as np
try:
    imread = scipy.misc.imread
except AttributeError:
    from imageio import imread

path_to_checkpoint_folder = 'checkpoint/chpt_model_360_640/'

print ('Reading sample image...')
frame_in = np.zeros(shape=(1,360,640,3),dtype=np.float32)
sample_frame = np.expand_dims(imread('sample_inference/reality_360_640.jpg'),0)
sample_frame = np.asarray(sample_frame,dtype=np.float32)/127.5 - 1.0
frame_in[0,:,:,:] = sample_frame
print ('Shape of double_frame_in is: '+str(np.shape(frame_in))+' .')

with tf.Session() as session:
    print ('Loading meta graph of cycleGAN...')
    saver = tf.train.import_meta_graph(meta_graph_or_file=path_to_checkpoint_folder+'cyclegan.model-84002.meta')
    print ('Finished loading meta graph of cycleGAN !')
    print ('Restoring latest checkpoint weights into cycleGAN...')
    saver = saver.restore(session,tf.train.latest_checkpoint(checkpoint_dir=path_to_checkpoint_folder))
    print ('Finished restoring latest checkpoint weights into cycleGAN !')

    print ('Restoring input and output tensor operators...')
    graph = tf.get_default_graph()
    # frame_inp = graph.get_operation_by_name(name='infer_inp')
    # frame_out = graph.get_operation_by_name(name='infer_out')
    frame_inp = graph.get_tensor_by_name(name='test_A:0')
    frame_out = graph.get_tensor_by_name(name='infer_out:0')
    print ('Finished restoring input and output tensor operators !')

    print ('Starting inference of single frame...')
    domain_transfered_frame = session.run(frame_out,feed_dict={frame_inp: frame_in})
    print ('Infered single frame !')

    print ('Saving infered frame...')
    scipy.misc.imsave(name='sample_inference/reality_360_640_transfered.jpg',arr=domain_transfered_frame[0,:,:,0:3])
    print ('Saved infered frame !')