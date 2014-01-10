'''
Created on Jan 10, 2014

@author: c3h3
'''


from sklearn.datasets import fetch_mldata
import cv2


mnist = fetch_mldata('MNIST original', data_home="/home/c3h3/c3h3Works/MLDM_Projects/learning_sklearn/data")

one_img = mnist["data"][0,:]

cv2.imshow('Display Window',mnist["data"][0,:].reshape(28,28))
cv2.waitKey(0)
cv2.destroyAllWindows()




if __name__ == '__main__':
    pass