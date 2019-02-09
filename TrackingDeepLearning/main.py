from __future__ import absolute_import, division, print_function
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from exercice1 import create_dataset,create_Sdbdataset, generate_real_dataset, generate_batch_from_dataset
from exercice2 import cnn_model_fn, loss_batch
from exercice3 import plot_histo, val_histo, KNN
import re
import random

tf.logging.set_verbosity(tf.logging.INFO)

proj_dir="D:/LouiseLoesch/telecom_paristech/3A/tracking and detection for CV/Projet 3"
coarse_dir = proj_dir+"/dataset/coarse/"; # Enter Directory of all images
fine_dir = proj_dir+"/dataset/fine/"; # Enter Directory of all images
real_dir =proj_dir+"/dataset/real/"; # Enter Directory of all images
checkpoint_dir = "D:/LouiseLoesch/telecom_paristech/3A/tracking and detection for CV/Projet 3/tmp/proj3/"

"""
Stack all the elements in some lists -> in increasing order of labels index.
"""

Sdb_labelsName,Sdb_feats,Sdb_quats,Sdb_labels=create_Sdbdataset(coarse_dir)
#print(Sdb_labelsName.shape,Sdb_feats.shape, Sdb_quats.shape, Sdb_labels.shape)

Strain_labelsName,Strain_feats,Strain_quats,Strain_labels,Stest_feats,Stest_quats,Stest_labels=create_dataset(fine_dir,real_dir)
#print(Strain_labelsName.shape, Strain_feats.shape, Strain_quats.shape, Strain_labels.shape, Stest_feats.shape, Stest_quats.shape, Stest_labels.shape)

# Identify former projects
print("Here are the projects already existing : ")
dirList = [x[0] for x in os.walk("summaries/")]
projectsName=[]
for i in range(len(dirList)):
    path=dirList[i]
    m = re.search('summaries/(.+)', path)
    if m:
        found = m.group(1)
        if found!="":
            projectsName.append(found)
            print("* : "+found)

# Ask the user if he wants to load a previous project
newProject=input("Choose a name for the project : if already existing, continue the project ; else create it : ")
if newProject in projectsName:
    reload=True
    files = [f for f in os.listdir(checkpoint_dir+newProject) if os.path.isfile(os.path.join(checkpoint_dir+newProject, f))]
    maximumStep=-1
    for j in range(len(files)):
        fileName=files[j]
        m = re.search('model_(.*).ckpt', fileName)
        if m:
            found = m.group(1)
            if found!="" and int(found)>maximumStep:
                maximumStep=int(found)
    if maximumStep==-1:
        reload=False
        print("Existing project but valid checkpoint not found")
    else:
        reload_path=checkpoint_dir+newProject+"/model_"+str(maximumStep)+".ckpt"
        startStep=maximumStep
        print("Existing project and valid checkpoint found (step "+str(startStep)+")")
else:
    reload=False
    startStep=-1
    print("Project not existing yet")

# Directory for Tensorboard reports
summaries_dir=newProject

# Parameters of the dataset
num_labels=Sdb_labelsName.shape[0]
dataset_size=Strain_labels.shape[0]
testingset_size=Stest_labels.shape[0]

# Parameters of the training
epochs = 1000
epochs_save = 10
num_samples = 50
batch_size = num_samples*3
rate = [0.001,0.0005,0.0001] # nb iter / 3 for each
steps_per_rate=int(epochs/len(rate))
nb_batch = int(dataset_size/ num_samples)
nb_batch_test = int(testingset_size/ num_samples)
histo_save = 40

# Computation of the descriptor for each batch training
x = tf.placeholder(tf.float32, [None, 64,64,3])
output = cnn_model_fn(x)
loss = loss_batch(output,batch_size)
learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8).minimize(loss)

# Constitution of the real dataset
real_dataset = generate_real_dataset(Strain_feats, Strain_quats, Strain_labels, Sdb_feats, Sdb_quats,Sdb_labels, num_labels)
test_dataset = generate_real_dataset(Stest_feats, Stest_quats, Stest_labels, Sdb_feats, Sdb_quats,Sdb_labels, num_labels)

# Precompute batches
batch_list=[generate_batch_from_dataset(real_dataset,i*batch_size,batch_size,dataset_size) for i in range(nb_batch)]
batch_test_list=[generate_batch_from_dataset(test_dataset,i*batch_size,batch_size,testingset_size) for i in range(nb_batch_test)]
batch_index=[i for i in range(nb_batch)]
batch_test_index=[i for i in range(nb_batch_test)]
random.shuffle(batch_index)

with tf.Session() as sess:

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
    if not os.path.exists(os.path.join('summaries',summaries_dir)):
        os.mkdir(os.path.join('summaries',summaries_dir))

    summ_writer = tf.summary.FileWriter(os.path.join('summaries',summaries_dir), sess.graph)

    with tf.name_scope('performance'):
        tf_loss_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary')
        tf_loss_summary = tf.summary.scalar('loss_training', tf_loss_ph)
        tf_loss_test_ph = tf.placeholder(tf.float32,shape=None,name='loss_summary_test')
        tf_loss_summary_test = tf.summary.scalar('loss_test', tf_loss_test_ph)
        performance_summaries = tf.summary.merge([tf_loss_summary,tf_loss_summary_test])

    with tf.name_scope("accuracy"):
        tf_acc_ph = tf.placeholder(tf.float32,shape=None,name='acc_summary')
        tf_acc_summary = tf.summary.scalar('accuracy_validation', tf_acc_ph)

    # Initialize networks
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
   
    sess.run(init_op) #execute init_op

    if reload==True:
        saver.restore(sess, reload_path)
        print("Model loaded")
    
    #print(np.max(tf.get_default_graph().get_tensor_by_name("conv2d/kernel:0").eval(session=sess)))
    for epoch in range(startStep+1,epochs+startStep):
        avg_loss = 0
        for i in range(len(batch_index)):
            batch_x = batch_list[batch_index[i]]
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, learning_rate: rate[min([len(rate)-1,int(epoch/steps_per_rate)])]})
            #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))# -> get all the layers labels
            avg_loss += c
            #print(np.mean(tf.get_default_graph().get_tensor_by_name("conv2d/kernel:0").eval(session=sess)))

        avg_loss = (avg_loss*1000)/dataset_size
        random.shuffle(batch_index)
        print("Epoch training :", (epoch + 1), "loss =", "{:.3f}".format(avg_loss))

        if epoch%epochs_save==0:

            avg_loss_test = 0
            for i in batch_test_index:
                c_test = sess.run(loss, feed_dict={x: batch_test_list[i]})
                avg_loss_test += c_test
            avg_loss_test = (avg_loss_test*1000)/testingset_size
            print("Testing  :", (epoch + 1), "loss =", "{:.3f}".format(avg_loss_test))
            summ = sess.run(performance_summaries, feed_dict={tf_loss_ph:avg_loss,tf_loss_test_ph:avg_loss_test})
            summ_writer.add_summary(summ, epoch)
            save_path = saver.save(sess, checkpoint_dir+newProject+"/model_"+str(epoch)+".ckpt")
            print("Model saved in path: %s" % save_path)

        if epoch%histo_save==0:
            # X=descriptors.eval()
            # KNN initialization
            neigh = KNeighborsClassifier(n_neighbors=1,weights='distance',p=2)
            X = sess.run(output, feed_dict={x: Sdb_feats})
            y=Sdb_labels
            neigh.fit(X, y)
            X2 = sess.run(output, feed_dict={x: Stest_feats})
            k_neighbors=neigh.kneighbors(X2,return_distance=False)
            result_array=KNN(Sdb_labels,Stest_labels,k_neighbors,Sdb_quats, Stest_quats)
            print("Results for epoch "+str(epoch)+" : ",result_array)
            plot_histo(result_array,"histo"+str(epoch),newProject)
            accuracy=result_array[3]*100
            summ_acc = sess.run(tf_acc_summary, feed_dict={tf_acc_ph:accuracy})
            summ_writer.add_summary(summ_acc, epoch)

    save_path = saver.save(sess, checkpoint_dir+newProject+"/model_"+str(epoch)+".ckpt")
    print("Model saved in path: %s" % save_path)

    print("Training completed")
    # KNN initialization
    neigh = KNeighborsClassifier(n_neighbors=1,weights='distance',p=2)
    
    X = sess.run(output, feed_dict={x: Sdb_feats})
    y=Sdb_labels
    neigh.fit(X, y)
    X2 = sess.run(output, feed_dict={x: Stest_feats})
    k_neighbors=neigh.kneighbors(X2,return_distance=False)
    result_array=KNN(Sdb_labels,Stest_labels,k_neighbors,Sdb_quats, Stest_quats)
    print("final results : ",result_array)
    plot_histo(result_array,"histo_final")
    accuracy=result_array[3]*100
    summ_acc = sess.run(tf_acc_summary, feed_dict={tf_acc_ph:accuracy})
    summ_acc_writer.add_summary(summ_acc, epoch)
