import matplotlib.pyplot as plt
import numpy as np

def val_histo(angle):
        """compute the values for the histogram"""
        angle_10=0
        angle_20=0
        angle_40=0
        angle_180=0
        for i in range(len(angle)):
                if angle[i]<10:
                        angle_10+=1
                if angle[i]<20:
                        angle_20+=1
                if angle[i]<40:
                        angle_40+=1
                if angle[i]<180:
                        angle_180+=1
        return [angle_10,angle_20,angle_40,angle_180]

def KNN(Sdb_labels,Stest_labels,k_neighbors,Sdb_quats, Stest_quats):
        """ find the nearest neighbor from the test dataset in the Sdb dataset"""
        angle=[]
        for ind in range(len(k_neighbors)):
                if Sdb_labels[k_neighbors[ind]]==Stest_labels[ind]:
                        angular_diff=2*np.arccos(abs((np.dot(Sdb_quats[k_neighbors[ind]],Stest_quats[ind])/(np.linalg.norm(Sdb_quats[k_neighbors[ind]])*np.linalg.norm(Stest_quats[ind])))))
                        angle.append(180*angular_diff/np.pi)
                        #print(180*angular_diff/np.pi)
        resultArray=np.array(val_histo(angle))
        resultArray=resultArray/len(k_neighbors)
        return resultArray

def plot_histo(resultArray,name,projectName):
        angle_10,angle_20,angle_40,angle_180=resultArray[0],resultArray[1],resultArray[2],resultArray[3]
	
        fig, ax = plt.subplots()
        ind = np.arange(1, 5)

        p10, p20, p40,p180 = plt.bar(ind, angle_180)
        p10.set_facecolor('r')
        p20.set_facecolor('g')
        p40.set_facecolor('b')
        p10.set_height(angle_10)
        p20.set_height(angle_20)
        p40.set_height(angle_40)
        p180.set_height(angle_180)
        ax.set_xticks(ind)
        ax.set_xticklabels(['<10°', '<20°', '<40°','<180°'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('pourcentage of images')
        ax.set_title('Histogram')
        plt.savefig(name+"_"+projectName+".png")
        # show the figure
        #plt.show()
