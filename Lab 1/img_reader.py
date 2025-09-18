import cv2

def show(filename, flag, window_flag):
    img = cv2.imread(filename, flag)
    cv2.namedWindow(filename, window_flag)
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
show('/Users/asyachz/DigitalMultimediaProcessingAlgorithms/Lab 1/src/cat.jpeg', cv2.IMREAD_GRAYSCALE,  cv2.WINDOW_FULLSCREEN)
show('/Users/asyachz/DigitalMultimediaProcessingAlgorithms/Lab 1/src/cat.png', cv2.IMREAD_COLOR, cv2.WINDOW_FREERATIO)
show('/Users/asyachz/DigitalMultimediaProcessingAlgorithms/Lab 1/src/cat.gif', cv2.IMREAD_REDUCED_COLOR_8, cv2.WINDOW_AUTOSIZE)
