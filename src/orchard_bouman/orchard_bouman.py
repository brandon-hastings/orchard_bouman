# Python implementation of Orchard Bouman clustering
# Author: Brandon Hastings
# Original research paper:
# ORCHARD, M. T., AND BOUMAN, C. A. 1991. Color Quantiza-tion of Images.
# IEEE Transactions on Signal Processing 39, 12,2677�2690.

import numpy as np
import numpy.ma as ma
from numpy.linalg import eig


class Cluster:
    """
    Class Cluster finds initial clusters and splits child nodes based on orchard bouman clustering given an image and
    optional tuple of 3x3 R matrix, 1x3 M matrix, and points array of shape image, where 0 values represent pixels belonging
     to that cluster.
     To perform the initial node split, Cluster only requires the image and creates the initial R, M and points array before splitting the
    initial node into two nodes. Specifically, the split_cluster function returns R, M, and points array for each child
    node. Create_new_cluster uses the above function to create and return two new Cluster objects, each representing one of
    the child nodes.

    Note: It is unlikely that this class will be accessed directly. For the purposes of performing clustering on an image
    Cluster is accessed by the OrchardBouman class.

    Inputs:
    Image: 3-dimensional numpy array of RGB image
    RMp: If None, initializes RMp from image. Else, RMp should be a tuple of 3x3 R matrix, 1x3 M matrix, and points array
        NOTE: RMp is called internally in method create_new_cluster. RMp should not be supplied for an initial node split.

    Attributes:
    img: Original 3-dimensional numpy array of RGB image
    RMp: tuple of 3x3 R matrix, 1x3 M matrix, and points array for Cluster object
    channels: list of 2-dimensional numpy arrays, with each element representing the R, G, and B color channel
    R: 3x3 matrix representing the sum of squares of pixel values belonging to that node in each RGB color channel
    M: 1x3 matrix representing the sum of pixel values belonging to that node in each RGB color channel
    point_arr: 2-dimensional numpy zeros array of size image. Zeros represent pixel locations assigned to given node, else
        the value is np.nan
    N: total number of pixels in node
    Q: 1x3 matrix representing the average pixel value for each RGB color channel. Q = M / N
    lmbda: maximum eigenvalue
    e: principle eigenvector
    """

    def __init__(self, image, RMp=None) -> None:
        self.img = image
        self.RMp = RMp

        # check input parameters
        self._check_params()

        # set initial values to None, call functions to set if needed so a function is not being called when referenced
        self.channels = None
        self.R = None
        self.M = None

        # set image channels
        self._split_image_channels()

        if RMp is None:
            self._initialize_r()
            self._initialize_m()
            self.point_arr = np.zeros((self.img.shape[0], self.img.shape[1]))
            # total number of pixels in image
            self.N = self.img.shape[0] * self.img.shape[1]
        else:
            # get RMp from tuple and use to set N
            self.R, self.M, self.point_arr = RMp
            # get number of pixels in cluster
            self.N = len(np.argwhere(~np.isnan(self.point_arr)))

        # cluster covariance
        R1_bar = self.R - ((self.M * self.M.H) / self.N)
        eig_vals, eig_vect = eig(R1_bar)
        # determine the maximum eigenvalue (lmbda) and principle eigenvector (e) of cluster
        self.lmbda = np.max(np.abs(eig_vals))
        self.e = eig_vect[np.argmax(np.abs(eig_vals))]
        self.Q = self.M / self.N

    # main method to determine each pixel's cluster assignment and return RMp for two child nodes from given node
    def split_cluster(self):

        # a is expression to determine cluster assignment for a pixel
        # masking each channel return channel pixels where points array is not nan, and is thus part of cluster
        a = self.e[0, 0] * ma.masked_where(self.point_arr == np.nan, self.channels[0]).filled(np.nan) + self.e[0, 1] * \
            ma.masked_where(self.point_arr == np.nan, self.channels[1]).filled(np.nan) + self.e[0, 2] * \
            ma.masked_where(self.point_arr == np.nan, self.channels[2]).filled(np.nan)

        point_arr1 = ma.masked_where(a > self.e * self.Q.T, self.point_arr).filled(np.nan)
        point_arr2 = ma.masked_where(a <= self.e * self.Q.T, self.point_arr).filled(np.nan)

        # construct R and M for first child node
        R1 = self._construct_r(point_arr1)
        M1 = self._construct_m(point_arr1)

        # determine R and M for second child node
        R2 = self.R - R1
        M2 = self.M - M1

        # return parameters of clusters
        return (R1, M1, point_arr1), (R2, M2, point_arr2)

    # create Cluster objects for created child nodes and return
    def create_new_clusters(self):
        cluster1_params, cluster2_params = self.split_cluster()
        cluster1 = Cluster(self.img, RMp=cluster1_params)
        cluster2 = Cluster(self.img, RMp=cluster2_params)

        return cluster1, cluster2

    # sum of square pixel values for pixels in a given cluster, using points_arr to determine which pixels in img should
    # be used
    def _construct_r(self, points_arr):
        R1 = np.zeros((3, 3))
        for i in range(R1.shape[0]):
            for j in range(R1.shape[1]):
                # R is 3x3 array representing sum of squares between each color channel
                R1[i, j] = (np.matrix(ma.masked_where(points_arr == np.nan, self.channels[i]).filled(np.nan)) *
                            np.matrix(ma.masked_where(points_arr == np.nan, self.channels[j]).filled(np.nan)).H).sum()
        return R1

    # sum pixel values for pixels in a given cluster, using points_arr to determine which pixels in img should be used
    def _construct_m(self, points_arr):
        # turn points_arr into same 3D shape as image array by repetition
        temp = np.repeat(points_arr[:, :, np.newaxis], 3, axis=2)
        # sum RGB values of original image ignoring nans in points array, return 1D matrix of length 3
        # image divided by 255 within mask to avoid error filling with nan in int64 dtype, then mask is multiplied back
        mask = ma.masked_where(np.isnan(temp), (self.img/255)).filled(np.nan) * 255
        return np.matrix(np.nansum(np.nansum(mask, axis=1), axis=0))

    # initialization method to split image color channels
    def _split_image_channels(self):
        R_channel = self.img[:, :, 0]
        G_channel = self.img[:, :, 1]
        B_channel = self.img[:, :, 2]
        channels = [R_channel, G_channel, B_channel]
        self.channels = channels

    # initialization method to build R matrix for an initial cluster
    def _initialize_r(self):
        R1 = np.zeros((3, 3))
        for i in range(R1.shape[0]):
            for j in range(R1.shape[1]):
                R1[i, j] = (self.channels[i] * self.channels[j]).sum()
        self.R = R1

    # initialization method to build M matrix for an initial cluster
    def _initialize_m(self):
        # sum RGB values, return 1D matrix of length 3. Image is not necessarily always weighted, only if weights are
        # supplied
        self.M = np.matrix(self.img.sum(axis=1).sum(axis=0))

    # method to check inputs before completing initialization
    def _check_params(self):
        if isinstance(self.img, np.ndarray) is False:
            raise TypeError(f"Expected image as numpy ndarray, not {type(self.img)}.")

        # check the value of RMp if supplied
        if self.RMp is not None:
            if isinstance(self.RMp, tuple) | isinstance(self.RMp, list):
                if len(self.RMp) != 3:
                    raise IndexError(f"RMp should be of length 3")
            else:
                raise TypeError(f"If RMp is supplied it should be a tuple or list, not {type(self.RMp)}.")


class OrchardBouman:
    """
    Class OrchardBouman performs orchard bouman clustering to split an image into 2^k number of child nodes, where each
    child node is an instance of Cluster. This is accomplished by method orchard_bouman. The resulting clustered color image
    can be returned with method construct_image.

    Inputs:
    Image: 3-dimensional numpy array of RGB image
    k: number of times to split nodes. Total number of resulting child nodes == 2^k

    Attributes:
    nodes: List of Cluster objects for each child node of length 2^k
    """

    def __init__(self, image, k) -> None:
        self.image = image
        self.k = k

        # check input parameters
        self._check_params()

        self.nodes = None

        # run on initialization to set the value of nodes
        self.orchard_bouman()

    # method that clusters an image in k^2 child nodes. Runs on intialization of the class to set the value of nodes
    def orchard_bouman(self):
        nodes = []
        # create initial Cluster object of the image and add to nodes list
        initial_cluster = Cluster(self.image)
        nodes.append(initial_cluster)
        print("Initial cluster created.")
        # for k number of times, split each cluster object in node list and replace the cluster object with the
        # corresponding cluster objects of it's child nodes

        for i in range(0, self.k):
            for j in range(0, len(nodes)):
                print("Splitting nodes.")
                # get cluster object
                node = nodes[j]
                # split it
                cluster1, cluster2 = node.create_new_clusters()
                # replace in list with tuple of child nodes
                nodes[j] = (cluster1, cluster2)
            # combine the tuples in node list before repeating the process
            nodes = [i for tup in nodes for i in tup]
            print(f"New nodes created. Total number of nodes = {len(nodes)}. Desired number of nodes = {2**self.k}.")
        # assign resulting nodes list after splitting is complete to nodes attribute
        self.nodes = nodes
        print("All nodes created.")

    # construct a color image
    # color image assigns each pixel to that channels average pixel value taken from Q
    def construct_image(self, segmentation_mask=False):
        # construct image template made of nan values
        image_temp = np.empty((self.image.shape[0], self.image.shape[1], 3))
        image_temp[:] = np.nan

        for i in range(0, len(self.nodes)):
            # convert point array to match shape of image by copying point array into 3rd dimension
            temp = np.repeat(self.nodes[i].point_arr[:, :, np.newaxis], 3, axis=2)
            # loop through each color channel of the image
            for j in range(0,3):
                # find pixels that belong to the current cluster
                indices = np.nonzero(~np.isnan(temp[:, :, j]))
                # # for every pixel assigned to cluster, set color equal to Q attribute, avg pixel value
                image_temp[:, :, j][indices] = self.nodes[i].Q[0, j]
        # return clustered color image
        return image_temp

    # method to check input parameters on initialization
    def _check_params(self):
        if isinstance(self.image, np.ndarray) is False:
            raise TypeError(f"Expected image as numpy ndarray, not {type(self.image)}.")
        if isinstance(self.k, int) is False:
            raise TypeError(f"Expected k as int, not {type(self.k)}.")
