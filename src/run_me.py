import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import sys
from sklearn.cluster import KMeans



def visualize(im1, im2, k):
    # displays two images
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.title('Original')
    f.add_subplot(1, 2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.title('Cluster: ' + str(k))
    plt.savefig('k_means_' + str(k) + '.jpg')
    plt.show()
    return None


def MSE(Im1, Im2):
    # computes error
    Diff_Im = Im2 - Im1
    Diff_Im = np.power(Diff_Im, 2)
    Diff_Im = np.sum(Diff_Im)
    Diff_Im = np.sqrt(Diff_Im)
    sum_diff = np.sum(np.sum(Diff_Im))
    avg_error = sum_diff / float(Im1.shape[0] * Im2.shape[1])
    return avg_error


def question3(data, original):
    k = [2,5,10,25,50,100,200]
    for j in k:
        guesser = KMeans(n_clusters=j)
        temp = (guesser.fit_predict(data))
        changed = []
        for i in temp:
            changed.append([guesser.cluster_centers_[i][0], guesser.cluster_centers_[i][1], guesser.cluster_centers_[i][2]])

        changed = np.array(changed)
        changed = changed.reshape(309, 393, 3)

        original_image = changed
        visualize(original, original_image,j)
        # original_image = original_image.astype('uint8')
        # plt.imshow(original_image)
        # plt.show()
        print("error for ", j, MSE(original, changed))

def PCA(image, vectors):
    k = [6, 12, 18, 24, 30, 36, 42, 50]
    for j in k:
        proj = []
        for i in range(j):
            proj.append(vectors[i])
        proj = np.array(proj)
        proj = proj.transpose()
        reduced = (image@proj)@proj.transpose()
        #visualize(image, reduced, j)
        plt.imshow(reduced)
        plt.title('Rank '+ str(j)+ " approximation")
        plt.show()
        print('error for ', j, np.square(np.subtract(image, reduced)).mean())

def PCAmse():
    k = [6, 12, 18, 24, 30, 36, 42, 50]
    output=[0,0,0,0,0,0,0,0]
    for l in range(100):
        base = '../../Data/Faces/face_'
        f = Image.open(base + str(l) + '.png')
        image = np.array(f)
        image=image/np.linalg.norm(image)
        comatrix = image @ image.transpose()
        values, vectors = np.linalg.eigh(comatrix)
        #print(values[0])
        vectors = np.flip(vectors)
        for j in range(len(k)):
            proj = []
            for i in range(k[j]):
                proj.append(vectors[i])
            proj = np.array(proj)
            proj = proj.transpose()
            reduced = (image @ proj) @ proj.transpose()
            # visualize(image, reduced,j)
            # print('error for ', j, MSE(image, reduced))
            output[j]+=np.square(np.subtract(image, reduced)).mean()
    #output.insert(0,100)
    output = np.array(output)
    output = output/100
    # plt.plot([6, 12, 18, 24, 30, 36, 42, 50], output)
    # plt.xlabel('k')
    # plt.ylabel('avg MSE')
    # plt.title('Avg MSE for each K for all 100 faces')
    # plt.show()


# grab the image
PCAmse()

original_image = np.array(Image.open('../../Data/face.png'))
test = original_image.astype('uint8')
plt.imshow(test)
plt.show()
# #original_image = original_image.reshape(2500,1)
# # print(np.shape(original_image))
# # print(original_image)
# data = []
# y=[]
original_image = original_image/np.linalg.norm(original_image)
test = original_image.astype('uint8')
plt.imshow(test)
plt.show()
comatrix = original_image@original_image.transpose()
# # print(np.shape(comatrix))
values, vectors = np.linalg.eigh(comatrix)
vectors = np.flip(vectors)
#print(np.flip(values))
plt.plot(list(range(0,50)), np.flip(values))
plt.xlabel('k')
plt.ylabel('avg MSE')
plt.title('Avg MSE for each K for all 100 faces')
plt.show()

PCA(original_image, vectors)


# original_image = original_image.astype('uint8')
# plt.imshow(original_image)
# plt.show()
# print(np.shape(original_image))
# for i in range(309):
#     for j in range(393):
#         data.append(original_image[i][j])

#test = np.array(data).reshape(309,393,3)
#question3(data, original_image)

