from numba import cuda
from numba import njit
from numba import prange
import imageio
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import convolve2d
import timeit


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    num_threads = 100#image.shape[1]
    num_blocks = 100#image.shape[0]
    batchsize = (math.ceil(image.shape[0] // num_blocks) + 1, math.ceil(image.shape[1] // num_threads) + 1)
    y = int(kernel.shape[0] / 2)
    x = int(kernel.shape[1] / 2)


    ker_image = cuda.to_device(image)
    ker_kernel = cuda.to_device(kernel)
    result = np.zeros((image.shape[0], image.shape[1]))
    result = cuda.to_device(result)

    correlation_kernel[num_blocks, num_threads](ker_kernel, ker_image, result, batchsize[0], batchsize[1], y , x)
    result = result.copy_to_host()
    return result



@cuda.jit
def correlation_kernel(kernel, image, result, ybatch, xbatch, y_move, x_move):
    i = cuda.blockIdx.x
    j = cuda.threadIdx.x
    #we will have permanent 1000 threads on 1000 blocks
    for y in range(i*ybatch, (i+1) * ybatch):
        if y < image.shape[0]:
            for x in range(j*xbatch, (j+1) * xbatch):
                if  x < image.shape[1]:
                    for q in range(kernel.shape[0]):
                        for k in range(kernel.shape[1]):
                            index = (y - y_move + q, x - x_move + k)
                            if 0 <= index[0] < image.shape[0] and 0 <= index[1] < image.shape[1]:
                                cuda.atomic.add(result[y], x, kernel[q][k] * image[y - y_move + q][x - x_move + k])
                                #result[y][x] += kernel[q][k] * image[i - y + q][j - x + k]


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    size = image.shape
    y = int(kernel.shape[0] / 2)
    x = int(kernel.shape[1] / 2)

    res = np.zeros(size)
    for i in prange(size[0]):
        for j in prange(size[1]):
            val = .0
            for q in range(kernel.shape[0]):
                for k in range(kernel.shape[1]):
                    index = (i - y + q, j - x + k)
                    if 0 <= index[0] < size[0] and 0 <= index[1] < size[1]:
                        val += kernel[q][k] * image[i - y + q][j - x + k]
            res[i][j] = val
    return res




def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    fil = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    image = load_image()
    Gx = correlation_numba(fil, image)
    Gy = correlation_numba(np.transpose(fil), image)
    return (Gx**2 + Gy**2) ** 0.5
    


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()

def sobel_test(flag):
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    image = load_image()
    if flag == 'CPU':
        flipped_kernel = np.rot90(kernel, 2)
        Gx = convolve2d(flipped_kernel, image)
        Gy = convolve2d(np.transpose(flipped_kernel), image)
        return (Gx**2 + Gy**2) ** 0.5
    elif flag == 'NUMBA':
        Gx = correlation_numba(kernel, image)
        Gy = correlation_numba(np.transpose(kernel), image)
        return (Gx**2 + Gy**2) ** 0.5
    elif flag == 'GPU':
        Gx = correlation_gpu(kernel, image)
        Gy = correlation_gpu(np.transpose(kernel), image)
        return (Gx**2 + Gy**2) ** 0.5

# Note use image show on your local computer to view the results 
def compare_sobel():
    '''run sobel_operator with different correlation functions (CPU, numba, GPU)
        '''
    def timer(f, flag):
        return min(timeit.Timer(lambda: f(flag)).repeat(2, 1))
    
    pic = load_image
    res = pic
    print(f"CPU:{timer(sobel_test, 'CPU')}")
    print(f"Numba:{timer(sobel_test, 'NUMBA')}")
    print(f"GPU:{timer(sobel_test, 'GPU')}")
    
    # your implementation
    # show_image(res)
