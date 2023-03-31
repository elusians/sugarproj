import matplotlib
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
plt.style.use('ggplot')
plt.title("Sheep Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")
 
image = mpimg.imread("sheep.jpg")
plt.imshow(image)
plt.show()
