
# ORCHARD, M. T., AND BOUMAN, C. A. 1991. Color Quantiza-tion of Images.
# IEEE Transactions on Signal Processing 39, 12,2677�2690.
import cv2
import numpy as np
import numpy.ma as ma
from numpy.linalg import eig


# TODO: Check eigen vectors and values for accuracy
# TODO: Evaluate flattening of points indices

class Cluster:
    def __init__(self, img, RMp=None) -> None:
        self.img = img
        self.channels = self._split_image_channels

        if RMp is None:
            self.R, = self._initialize_r
            self.M = self._initialize_m
            # flattened indices for pixels of image array of each channel
            self.point_arr = np.zeros((self.img.shape[0], self.img.shape[1]))
            self.point = list(range(0, img.shape[0] * img.shape[1]))
            self.N = self.img.shape[0] * self.img.shape[1]
        elif isinstance(RMp, tuple):
            self.R, self.M, self.point = RMp
            self.N = len(self.point)
        else:
            raise TypeError(f"RMp must be of type Tuple, not {type(RMp)}")

        R1_bar = self.R - ((self.M * self.M.H) / self.N)
        eig_vals, eig_vect = eig(R1_bar)
        self.eig_vect = eig_vect[:, 0]
        self.eig_val = abs(eig_vals[0, 0])

    def split_cluster(self):
        # check this function
        index = list(self.eig_val).index(max(self.eig_val))
        vector = self.eig_vect[index]
        point = self.point[index]
        point_arr = self.point_arr[index]
        M = self.M[index]
        N = self.N[index]
        Q = M / N
        R = self.R[index]

        a = vector[0, 0] * self.channels[0][point] + vector[1, 0] * self.channels[1][point] + vector[2, 0] * \
            self.channels[2][point]
        # a = vector[0, 0] * ma.masked_where(point_arr == np.nan, self.channels[0]) + \
        #     vector[1, 0] * ma.masked_where(point_arr == np.nan, self.channels[1]) + \
        #     vector[2, 0] * ma.masked_where(point_arr == np.nan, self.channels[2])

        point1 = point[np.where(a <= vector * Q)]
        point2 = point[np.where(a > vector * Q)]

        point_arr1 = ma.masked_where(~a <= vector * Q, point_arr)
        point_arr2 = ma.masked_where(~a > vector * Q, point_arr)

        # construct R and M from first cluster
        R1 = self._construct_r(point1)
        M1 = self._construct_m(point1)

        # get R and M from second cluster
        R2 = R - R1
        M2 = M - M1

        # return parameters of clusters
        return (R1, M1, point1), (R2, M2, point2)

    def create_new_clusters(self):
        cluster1_params, cluster2_params = self.split_cluster
        cluster1 = Cluster(self.img, RMp=cluster1_params)
        cluster2 = Cluster(self.img, RMp=cluster2_params)

        return cluster1, cluster2

    def _split_image_channels(self):
        R_Comp = self.img[:, :, 0]
        G_Comp = self.img[:, :, 1]
        B_Comp = self.img[:, :, 2]
        channels = [R_Comp, G_Comp, B_Comp]
        return channels

    def _initialize_r(self):
        R1 = np.zeros(2, 2)
        for i in range(3):
            for j in range(3):
                R1[i, j] = (self.channels[i] * self.channels[j]).sum()
        return R1

    def _initialize_m(self):
        # check M
        M = self.img.sum()
        M1 = np.zeros(0, 2)
        M1[0, 0] = M[:, :, 0]
        M1[0, 1] = M[:, :, 1]
        M1[0, 2] = M[:, :, 2]
        return M1

    def _construct_r(self, points):
        R1 = np.zeros(2, 2)
        for i in range(3):
            for j in range(3):
                # calling with raveled points as index?
                R1[i, j] = (self.channels[i][points] * self.channels[j][points].H).sum()
        return R1

    def _construct_m(self, points):
        M = self.img.sum()
        M1 = np.zeros(0, 2)
        # calling with raveled points as index?
        M1[0, 0] = M[:, :, 0][points]
        M1[0, 1] = M[:, :, 1][points]
        M1[0, 2] = M[:, :, 2][points]
        return M1


class OrchardBouman:
    def __init__(self, image, k) -> None:
        self.image = image
        self.k = k
        self.nodes = None

    def construct_image(self):
        for i in range(0, self.k - 1):
            temp = self.nodes[i].point
            I1 = np.zeros(self.image.shape[0], self.image.shape[1])
            # what?
            # y_p, x_p = ind2sub([size(I,1) size(I,2)],temp)
            y_p, x_p = np.unravel_index(temp, (self.image.shape[0], self.image.shape[1]))
            for t in range(0, len(y_p)):
                y = y_p(t)
                x = x_p(t)
                I1[y, x, 0] = self.image[y, x, 0]
                I1[y, x, 1] = self.image[y, x, 1]
                I1[y, x, 2] = self.image[y, x, 2]

    def orchard_bouman(self):
        nodes = []
        initial_cluster = Cluster(self.image)
        nodes.append(initial_cluster)
        for i in range(0, self.k - 1):
            for j in range(0, len(nodes)):
                node = nodes[j]
                cluster1, cluster2 = node.create_new_clusters
                nodes[j] = (cluster1, cluster2)
            # combine each cluster into new nodes list
            nodes = [i for tup in nodes for i in tup]
        self.nodes = nodes
        clustered_image = self.construct_image

        return clustered_image
