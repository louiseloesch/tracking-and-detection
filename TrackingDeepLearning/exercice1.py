import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

def create_Sdbdataset(coarse_dir):
        """create Sdb dataset by storing the pose in Sdb_quats, the images in Sdb_feats and the labels in Sdb_labels"""
        list_folders_coarse = [name for name in os.listdir(coarse_dir) if os.path.isdir(os.path.join(coarse_dir, name))]
        #print(list_folders_coarse)
        Sdb_feats=[]
        Sdb_quats=[]
        Sdb_labels=[]
        Sdb_labelsName=list_folders_coarse
        for index_folder in range(len(list_folders_coarse)):
                folder = list_folders_coarse[index_folder]
                lines = open(coarse_dir+folder+"/poses.txt").read().split("# ")
                for content in lines:
                        x=content.split("\n")
                        if (len(x)>2):
                                filename=coarse_dir+folder+'/'+x[0]
                                #Load the image
                                image = mpimg.imread(filename)
                                pose=x[1].split(' ')
                                pose = list(map(float, pose))
                                #obj.append([image,pose])
                                Sdb_feats.append(image)
                                Sdb_quats.append(pose)
                                Sdb_labels.append(index_folder)
        return np.array(Sdb_labelsName),np.array(Sdb_feats),np.array(Sdb_quats),np.array(Sdb_labels)

def create_dataset(fine_dir,real_dir):
        """ create Train dataset and test dataset the same way we created the Sdb dataset"""
        list_folders_fine=[name for name in os.listdir(fine_dir) if os.path.isdir(os.path.join(fine_dir, name))]
        list_folders_real = [name for name in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, name))]

        # Training dataset
        Strain_feats=[]
        Strain_quats=[]
        Strain_labels=[]
        Strain_labelsName=list_folders_fine

        # Test dataset : we assume the label are collected in the same way
        Stest_feats=[]
        Stest_quats=[]
        Stest_labels=[]
        
        for index_folder in range(len(list_folders_fine)):
                folder = list_folders_fine[index_folder]
                lines = open(fine_dir+folder+"/poses.txt").read().split("# ")
                for content in lines:
                        x=content.split("\n")
                        if len(x)>2:
                                filename=fine_dir+folder+'/'+x[0]
                                # Load the image
                                image = mpimg.imread(filename)
                                pose=x[1].split(' ')
                                pose = list(map(float, pose))
                                Strain_feats.append(image)
                                Strain_quats.append(pose)
                                Strain_labels.append(index_folder)
                                
        train = open(real_dir+"training_split.txt").read().split(", ")
        train = list(map(int, train))
        line=0
        for index_folder in range(len(list_folders_real)):
                folder = list_folders_real[index_folder]
                cpt=0
                lines = open(real_dir+folder+"/poses.txt").read().split("# ")
                for content in lines:
                        x=content.split("\n")
                        if len(x)>2:
                                filename=real_dir+folder+'/'+x[0]
                                # Load the image
                                image = mpimg.imread(filename)
                                pose=x[1].split(' ')
                                pose = list(map(float, pose))
                                if (cpt in train):
                                        Strain_feats.append(image)
                                        Strain_quats.append(pose)
                                        Strain_labels.append(index_folder)
                                else:
                                        Stest_feats.append(image)
                                        Stest_quats.append(pose)
                                        Stest_labels.append(index_folder)
                                cpt+=1
                line+=1
        return np.array(Strain_labelsName),np.array(Strain_feats),np.array(Strain_quats),np.array(Strain_labels),np.array(Stest_feats),np.array(Stest_quats),np.array(Stest_labels)

def generate_real_dataset(Strain_feats, Strain_quats, Strain_labels, Sdb_feats, Sdb_quats,Sdb_labels, num_labels):
        """ generate the triplet for each image of the training dataset  and we shuffle the training set"""
        size_dataset=int((Strain_labels).shape[0])
        size_data_per_labels=int(size_dataset/num_labels)
        size_sdb=int((Sdb_labels).shape[0])
        size_sdb_per_labels=int(size_sdb/num_labels)
        Nsamples=size_dataset

        Lindices = np.array([i for i in range(Nsamples)]) # indices of the samples to consider : for num_samples==-1, we just do that to shuffle the dataset
        np.random.shuffle(Lindices)

        batch_feats=[]
        for index in Lindices:
                anchor_feats=Strain_feats[index]
                anchor_quats=Strain_quats[index]
                anchor_label=int(Strain_labels[index])
                min_theta=np.pi
                min_idx=-1
                for idx in range(anchor_label*size_sdb_per_labels,(anchor_label+1)*size_sdb_per_labels):
                        candidate_quat=Sdb_quats[idx]
                        denom=np.linalg.norm(anchor_quats)*np.linalg.norm(candidate_quat)
                        epsilon=1e-6
                        if denom<epsilon:
                                denom=epsilon
                        arg=abs((np.dot(anchor_quats,candidate_quat)/denom))
                        if arg>=1:
                                arg=1
                        theta=2*np.arccos(arg)
                        if theta<min_theta and index!=idx:
                                min_theta=theta
                                min_idx=idx
                puller_feats=Sdb_feats[min_idx]
                typePusher=random.randint(0,1)
                if typePusher==0:
                        other=random.randint(anchor_label*size_sdb_per_labels,(anchor_label+1)*size_sdb_per_labels-1)
                        if other!=min_idx:
                                pusher_feats = Sdb_feats[other]
                        else:
                                while other==min_idx:
                                        other=random.randint(0,size_sdb-1)
                                pusher_feats=Sdb_feats[other]
                else:
                        other=random.randint(0,size_sdb-1)
                        while other==min_idx:
                                other=random.randint(0,size_sdb-1)
                        pusher_feats=Sdb_feats[other]

                batch_feats.append(anchor_feats)
                batch_feats.append(puller_feats)
                batch_feats.append(pusher_feats)
                
        return np.array(batch_feats)

def generate_batch_from_dataset(real_dataset,start,batch_size,dataset_size):
        """ generate batch from the dataset """
        if start+batch_size >= dataset_size*3:
                return real_dataset[start:]
        else:
                return real_dataset[start:start+batch_size]

