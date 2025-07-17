# color-detection-using-python

COLOR DETECTION USING PYTHON

ABSTRACT:

Color detection is necessary to recognize objects, it is also used as a tool in various image editing and drawing apps. It is the process of detecting the name of any color. Well, for humans this is an extremely easy task but for computers, it is not straightforward. Human eyes and brains work together to translate light in to color. Light receptors that are present in our eyes transmit the signal to the brain. Our brain then recognizes the color. Hence the problem that arises in front of us is how to make a computer understand or recognize colors, so we are going to solve this problem So basically in this project using python we need 3 different parts to be used. Python code which will be recognizing color, Image that will be used for testing the color recognition, a.csv file that will be containing the colors as dataset. Hence the above 3 modules will help us in achieving our aim that is detecting the colors in an image using python.

INTRODUCTION:

Color detection using Python is a popular and practical application of computer vision and image processing. It involves identifying and recognizing colors in images or video streams. This can be useful in various fields such as robotics, industrial automation, image editing, and more. In this introduction, I'll provide you with an overview of how to perform color detection using Python.

Prerequisites: Before diving into color detection, you should have a basic understanding of Python programming and some familiarity with libraries such as OpenCV (Open Source Computer Vision Library) and NumPy for image processing and manipulation.

Steps for Color Detection:

Installation of Required Libraries:Ensure that you have OpenCV and NumPy installed. You can install them using pip: Copy code.

pip install opencv-python pip install numpy.

Capture or Load an Image/Video:You can either capture an image using your device's camera or load an existing image or video. OpenCV provides functions to do both.

Convert the Image to the Desired Color Space:Images are often represented in the BGR (Blue, Green, Red) color space in OpenCV. Depending on your application, you may need to convert the image to another color space, such as HSV (Hue, Saturation, Value), which is useful for color-based segmentation.

Define the Color Range:Specify the range of colors you want to detect in the target color space. For example, if you want to detect red, you might define a range of hues that correspond to the red color.

Thresholding and Filtering:Apply thresholding techniques to isolate the pixels that fall within the defined color range. This will create a binary mask where the detected color is highlighted.

Find and Analyze Contours: Use contour detection algorithms in OpenCV to find and identify the contours of the detected color regions. Contours can help you determine the location and shape of the color areas.

Draw Boundaries or Perform Actions: Once you've identified the color regions, you can draw bounding boxes, circles, or other shapes around them. Alternatively, you can perform specific actions based on the detected color, depending on your application.

Display the Result: Finally, display the processed image or video with the detected colors and any annotations you added.

EXISTING SYSTEM:

Existing color detection systems often use computer vision libraries like OpenCV to process images or video frames. These systems typically start by converting the image into a suitable color space, such as HSV, to simplify color analysis. They then apply color thresholding, defining specific color ranges of interest. Once the desired colors are identified, techniques like blob detection may be used to locate and analyse color regions. In more advanced systems, machine learning models can aid in color recognition. These systems can be designed for various applications, from image processing to object sorting or tracking, and may include user interfaces for interaction and calibration for accurate detection under varying lighting conditions.

DISADVANTAGES OF EXISTING SYSTEM:

RGB COLOR MODEL: 1.sensitive to lighting.

2.Inability to Handle illuminant change.

COLOR THRESHOLDING:

1.color variability.

2.Manual thresholding.

HSV COLOR:

1.Storage and processing overhead.

2.loss of spatial information.

CUSTOM ALOGRITHMS:

1.Development and maintenance.

2.Resource –intensive.

3.Limited Generalization.

PROPOSED SYSTEM: The proposed color detection system utilizes advanced computer vision techniques and machine learning to accurately identify and track colors in real-time. This system integrates seamlessly with cameras or image sources, making it versatile for applications such as quality control, object tracking, and robotics. Its user-friendly interface allows users to select specific colors of interest and fine-tune detection parameters. With adaptive color thresholding and the ability to handle varying lighting conditions, this system offers precise and reliable color recognition. Its scalability and compatibility with other hardware make it a valuable tool for a wide range of industries and applications, ensuring consistent and efficient color analysis.

ADVANTAGES OF PROPOSED SYSTEM: The advantages of adding color detection to an proposed system depend on the specific use case and the goals of the system. By integrating color detection capabilities, you can improve performance, efficiency, and decision-making while also expanding the system's functionality and versatility.

1.Encganced object identification 2. Quality control and inspection 3.safety and security 4.data analysis 5.error reduction. MODULE WISE DESCRIPITION OR MODULES:

MODULES: OpenCV (Open Source Computer Vision Library): OpenCV is one of the most widely used libraries for computer vision tasks, including color detection. It provides functions for image acquisition, color space conversion, thresholding, contour detection, and more.

• Installation: You can install OpenCV using pip.

Copy code pip install opencv-python.

NumPy: NumPy is a fundamental library for numerical operations in Python. It's often used in combination with OpenCV for efficient image manipulation.

• Installation: You can install NumPy using pip.

Copy code

pip install numpy

• Documentation: https://numpy.org/doc/stable/.

Matplotlib: Matplotlib is a popular library for creating visualizations and displaying images. It's helpful for visualizing the results of color detection.

• Installation: You can install Matplotlib using pip.

Copy code

pip install matplotlib

• Documentation: https://matplotlib.org/stable/contents.html.

Pillow (PIL Fork): Pillow is a library for image processing tasks, including color manipulation and conversion. It's useful when you need to work with images in different formats.

• Installation: You can install Pillow using pip.

Copy code

pip install Pillow

• Documentation: https://pillow.readthedocs.io/en/stable/index.html.

Scikit-Image: Scikit-Image is another library for image processing in Python. It provides various algorithms and functions for color manipulation and analysis.

• Installation: You can install Scikit-Image using pip.

Arduino copy.

Code pip install.

Scikit –image.

• Documentation: https://scikit-image.org/docs/stable/index.html.

Python Colorama: If you want to add color to your terminal output for a command-line

interface application, you can use Python Colorama.

There are various software development approaches defined and designed which are used/employed during development process of software, these approaches are also referred as "Software Development Process Models". Each process model follows a particular life cycle in order to ensure success in process of software development.

KNOWLEDGE REQUIRED TO DEVELOPE THE PROJECT:

Color can be identified from the sensoryoptic nerves of the eyes. Colors only be seen or identified when a source of light is applied to an object. Color blindness can be termed as inability of the differentiation between colors. It is incurable disease that can be termed as lifelong disease. Edges can be very helpful in color differentiation boundary.

Color Detection can be used in agriculture industry to find the weeds the along with the crops. Via and destroyed and the crop scan be saved. It can be also used in medical industries to detect the disease and other disorders especially in face and other internal diseases like cancers.

Colour detection is the process of detecting the name of any colour. Simple, isn’t it? Well, for humans this is an extremely easy task but for computers, it is not straightforward.

Human eyes and brains work together to translate light into color. Light receptors that are present in our eyes transmit the signal to the brain. Our brain then recognizes the color.

· Define Your Scope: Determine the specific aspects of color detection you want to explore, such as object recognition, image processing, or color matching.

· Search Databases: Use academic databases like IEEE Xplore, PubMed, Google Scholar, and research journals to search for relevant papers. Keywords like "color detection," "Python," and "image processing" can help narrow your search.

· Filter and Review: Go through the search results and filter out papers that are most relevant to your topic. Read the abstracts and conclusions to decide which ones to include in your survey.

· Categorize and Summarize: Group the selected papers into categories based on their focus or methodology. Summarize the key findings, methodologies, and contributions of each paper.

· Identify Trends and Gaps: Analyze the common trends, tools, and techniques used in the papers. Also, identify any research gaps or areas where further investigation is needed.

SOURCE CODE:

import cv2

import numpy as np

import pandas as pd

import argparse

#Creating argument parser to take image path from command line

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True, help="Image Path")

args = vars(ap.parse_args())

img_path = args['image']

#Reading the image with opencv

img = cv2.imread(img_path)

#declaring global variables (are used later on)

clicked = False

r = g = b = xpos = ypos = 0

#Reading csv file with pandas and giving names to each column

index=["color","color_name","hex","R","G","B"]

csv = pd.read_csv('colors.csv', names=index, header=None)

#function to calculate minimum distance from all colors and get the most matching color

def getColorName(R,G,B):

minimum = 10000

for i in range(len(csv)):

d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))

if(d<=minimum):

minimum = d

cname = csv.loc[i,"color_name"]

return cname

#function to get x,y coordinates of mouse double click

def draw_function(event, x,y,flags,param):

if event == cv2.EVENT_LBUTTONDBLCLK:

global b,g,r,xpos,ypos, clicked

clicked = True

xpos = x

ypos = y

b,g,r = img[y,x]

b = int(b)

g = int(g)

r = int(r)

cv2.namedWindow('image')

cv2.setMouseCallback('image',draw_function)

while(1):

cv2.imshow("image",img)

if (clicked):

#cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle

cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

#Creating text string to display( Color name and RGB values )

text = getColorName(r,g,b) + ' R='+ str(r) + ' G='+ str(g) + ' B='+ str(b)

#cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )

cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

#For very light colours we will display text in black colour

if(r+g+b>=600):

cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)

clicked=False

#Break the loop when user hits 'esc' key

if cv2.waitKey(20) & 0xFF ==27:

break

cv2.destroyAllWindows()
