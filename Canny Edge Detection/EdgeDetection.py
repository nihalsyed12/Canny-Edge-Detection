from math import ceil
import string
import cv2 
import numpy as np
#Nihal Syed



#Function that generates Gaussian kernel. 1 input (sigma)
def gaussianKernel(sigma):
    #size is set based on sigma 
    size = 2 * ceil(3*sigma) + 1
    #make new kernel based on new size  
    newKernel = np.linspace(-size/2, size/2, size)
    
    #apply gausian distribution to each kernel element 
    for i in range(size):
        newKernel[i] = 1 / (np.sqrt(2*np.pi) * sigma) * np.e ** (-np.power((newKernel[i]-np.mean(newKernel[i])) / sigma, 2) /2)
        
    #make kernel matrix out of kernel vectors
    gauss2d = np.outer(newKernel.T, newKernel.T)
    #normailse to make sure center value is always 1
    gauss2d *= 1.0/gauss2d.max()

    #return kernel matrix 
    return gauss2d

#Function 2 parameters (image, kernel) applies kernel convolution over image
def convolution(img, kernel):

    #keep track of image and kernel rows and columns
    kern = kernel.shape
    imageRow, imageCol = img.shape
    kernRow, kernCol = kernel.shape

    #new output matrix to fill 
    outImage = np.zeros(img.shape)

    #Create new image with padding and set image values in new paddedImage
    padding = int(kernRow / 2)
    #new matrix with padding
    newImage = np.zeros((imageRow + (2 * padding), imageCol + (2 * padding)))
    newImage[padding: newImage.shape[0] - padding, padding: newImage.shape[1] - padding] = img     

    #applies convolution 
    for i in range(imageRow):
        for j in range(imageCol):
            outImage[i, j] = np.sum(kernel * newImage[i: i + kernRow, j: j + kernCol])
            #divide output pixel by total number of pixels
            outImage[i, j] /= kern[0] * kern[-1]
     
    outImage = outImage.astype(np.uint8)
    return outImage

#Sobel function takes image as input parameter and applies sobel filter, returns gradianMagnitude and gradiantDirection
def sobelFunction(img):
    #Sobel Filters for X and Y pass
    sobelFilterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelFilterY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    #convolve filters for both X and Y filters
    sobelX = convolution(img, sobelFilterX)
    sobelY = convolution(img, sobelFilterY)

    #calculate gradiant magnitude image 
    gradMagnitudeImage = np.sqrt(np.square(sobelX) + np.square(sobelY))
    gradMagnitudeImage *= 255/gradMagnitudeImage.max()
    gradMagnitudeImage = gradMagnitudeImage.astype(np.uint8)

    #calculate gradient direction image 
    gradientDirectionImage = np.arctan2(sobelX, sobelY)
    gradientDirectionImage = np.degrees(gradientDirectionImage) + 180
    gradientDirectionImage = gradientDirectionImage.astype(np.uint8)

    return gradMagnitudeImage, gradientDirectionImage

#Function that takes gradient Magnitude and gradient direction and applies non maxima suppression
def non_maximaSuppression(gradMagnitudeImage, gradientDirectionImage):

    #create new zero matrix to update 
    outImage = np.zeros(gradMagnitudeImage.shape)
    #Variables keep track of columns and rows
    imageRow, imageCol = gradMagnitudeImage.shape
    
    #nested for loop checks each pixel gradient, mapping gradient angle to 4 different cases 
    for row in range(0, imageRow - 1):
        for col in range(0, imageCol - 1):
            #map neibours magnitude base of gradient direction
            #angle between 0<x<45
            if (0 <= np.absolute(gradientDirectionImage[row, col]) < 45):
                leftNeighbour = gradMagnitudeImage[row, col -1]
                rightNeighbour = gradMagnitudeImage[row, col + 1]
            #angle between 45 < x < 90
            elif ( 45 <= np.absolute(gradientDirectionImage[row, col]) < 90):
                    leftNeighbour = gradMagnitudeImage[row - 1, col]
                    rightNeighbour = gradMagnitudeImage[row + 1, col]
            #angle between 45 < x < 135     
            elif (90 <= np.absolute(gradientDirectionImage[row, col]) < 135):
                leftNeighbour = gradMagnitudeImage[row + 1, col - 1]
                rightNeighbour = gradMagnitudeImage[row - 1, col + 1]
            else:
                leftNeighbour = gradMagnitudeImage[row - 1, col - 1]
                rightNeighbour = gradMagnitudeImage[row + 1, col + 1]

            #apply non-maxima suppression
            if gradMagnitudeImage[row, col] > leftNeighbour or gradMagnitudeImage[row, col] < rightNeighbour:
                outImage[row, col] = 0
            else: 
                outImage[row, col] = gradMagnitudeImage[row, col]
            
    outImage = outImage.astype(np.uint8)


    return outImage

if __name__ == '__main__':
    img = cv2.imread('cat2.jpg', 0)
    #save nonmaxsuppress image, grad mag image, grad orientation image
    gaussKernel = gaussianKernel(1)
    blurredImage = convolution(img, gaussKernel)
    gradMag, gradDir = sobelFunction(blurredImage)
    nMax = non_maximaSuppression(gradMag, gradDir)

    #images are created and saved 
    cv2.imwrite('gradOrien.jpg', gradDir)
    cv2.imwrite('gradMag.jpg', gradMag)
    cv2.imwrite('non-maxSupression.jpg', nMax)
   