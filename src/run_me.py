import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import sys
from sklearn.cluster import KMeans

################################################################################
##@function- displays the top K rank approximation of input image
##@param image- input image to be approximated
##@param vectors- the eigenvectors of image, in order of decreasing eigenvaules
##@return- void, simply displays the approximations of the image along with
##           the MSE of each approximation
################################################################################

def PCA(image, vectors):
    k = [6, 12, 18, 24, 30, 36, 42, 50]#ranks to be approximated
    for j in k:
        proj = []
        for i in range(j):#getting the top k eigenvectors 
            proj.append(vectors[i])
        proj = np.array(proj)
        proj = proj.transpose()
        reduced = (image@proj)@proj.transpose()#projecting onto the top K rank approx subspace
        #visualize(image, reduced, j)
        plt.imshow(reduced)#showing the image
        plt.title('Rank '+ str(j)+ " approximation")
        plt.show()
        print('error for ', j, np.square(np.subtract(image, reduced)).mean())

        
        
####################################################################
##@function- calculates the average MSE of PCA on all 100 images
####################################################################
def PCAmse():
    k = [6, 12, 18, 24, 30, 36, 42, 50]
    output=[0,0,0,0,0,0,0,0]
    for l in range(100):
        base = '../../Data/Faces/face_'#reading in each picture one at a time
        f = Image.open(base + str(l) + '.png')
        image = np.array(f)
        image=image/np.linalg.norm(image)#normalizing image matrix
        comatrix = image @ image.transpose()#creating covariance matrix
        values, vectors = np.linalg.eigh(comatrix)#get eigenvectors 
        vectors = np.flip(vectors)#order eigenvectors such that the one corresponding to the highest eigenvalue is first
        #Perform PCA for each image
        for j in range(len(k)):
            proj = []
            for i in range(k[j]):
                proj.append(vectors[i])
            proj = np.array(proj)
            proj = proj.transpose()
            reduced = (image @ proj) @ proj.transpose()
            output[j]+=np.square(np.subtract(image, reduced)).mean()#calculate MSE for each image
    output = np.array(output)
    output = output/100
    #plotting MSE by K
    plt.plot([6, 12, 18, 24, 30, 36, 42, 50], output)
    plt.xlabel('k')
    plt.ylabel('avg MSE')
    plt.title('Avg MSE for each K for all 100 faces')
    plt.show()


# grab the image
PCAmse()

original_image = np.array(Image.open('../../Data/face.png')) #read in example image
test = original_image.astype('uint8')#showing original image
plt.imshow(test)
plt.show()
original_image = original_image/np.linalg.norm(original_image)#normalizing image matrix
comatrix = original_image@original_image.transpose()#creating covariance matrix
values, vectors = np.linalg.eigh(comatrix)#getting eigenvectors
vectors = np.flip(vectors)#reordering eigenvectors in decreasing order
plt.plot(list(range(0,50)), np.flip(values))#plotting eigenvalues in decreasing order, this tells you how well a matrix can be approximated. A steep drop off early on is desired
plt.xlabel('k')
plt.ylabel('avg MSE')
plt.title('Avg MSE for each K for all 100 faces')
plt.show()

PCA(original_image, vectors)
