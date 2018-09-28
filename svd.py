from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# 1. Read in image
filename = "./hendrix_final.png"
img = mpimg.imread(filename)

plt.imshow(img)
plt.title("Original Image")
plt.show()

# 2. Extract RGB color channels from the image tensor and convert to double precision
red_img = img[0:len(img), 0:len(img), 0]
green_img = img[0:len(img), 0:len(img), 1]
blue_img = img[0:len(img), 0:len(img), 2]

# 3. Convert each channel image to double precision
red_img = red_img.astype(np.float64)
green_img = green_img.astype(np.float64)
blue_img = blue_img.astype(np.float64)

# 4. Perform SVD on RGB channels of the image
red_u, red_s, red_vh = np.linalg.svd(red_img)
red_sigma = np.diag(red_s)
red_recon_img = np.matmul(np.matmul(red_u, red_sigma), red_vh)

green_u, green_s, green_vh = np.linalg.svd(green_img)
green_sigma = np.diag(green_s)
green_recon_img = np.matmul(np.matmul(green_u, green_sigma), green_vh)

blue_u, blue_s, blue_vh = np.linalg.svd(blue_img)
blue_sigma = np.diag(blue_s)
blue_recon_img = np.matmul(np.matmul(blue_u, blue_sigma), blue_vh)

# 5. Plot the singular values of R with a log-log plot
plt.loglog(red_s)
plt.xlabel('i')
plt.ylabel(r'$σ_i \/Value$')
plt.title('R Channel Singular Values')
plt.show()

# 6. Take Frobenius norm of the reconstruction error matrix subtracted from the  R, G, B channel images
#  for increasing dimensions (1,...,k=2000). Sample the Frobenius norm at every 100 dimensions such that there are 
#  20 samples to plot for each R, G, B image

red_norm_list = []
green_norm_list = []
blue_norm_list = []

for i in range(100, 2100, 100):
    r_u = red_u[:, 0:i]
    r_s = red_sigma[0:i, 0:i]
    r_vh = red_vh[0:i, :]    

    g_u = green_u[:, 0:i]
    g_s = green_sigma[0:i, 0:i]
    g_vh = green_vh[0:i, :]    

    b_u = blue_u[:, 0:i]
    b_s = blue_sigma[0:i, 0:i]
    b_vh = blue_vh[0:i, :]    

    red_norm = np.linalg.norm(red_img - np.matmul(np.matmul(r_u, r_s), r_vh))    
    red_norm_list.append(red_norm)

    blue_norm = np.linalg.norm(blue_img - np.matmul(np.matmul(b_u, b_s), b_vh))    
    blue_norm_list.append(blue_norm)

    green_norm = np.linalg.norm(green_img - np.matmul(np.matmul(g_u, g_s), g_vh))    
    green_norm_list.append(green_norm)
    
plt.plot([100 * x for x in range(1,21)], red_norm_list, 'r')
plt.plot([100 * x for x in range(1,21)], green_norm_list, 'g')
plt.plot([100 * x for x in range(1,21)], blue_norm_list, 'b')
plt.title('Frobenius Norm Values of Reconstruction Error Matrix vs. Dimension')
plt.xlabel('Dimension')
plt.ylabel('Frobenius Norm Value')
plt.legend(('R','G','B'))
plt.show()

# 7. Show the reconstructed image with chosen dimension k = 150!

k = 150

r_u = red_u[:, 0:k]
r_s = red_sigma[0:k, 0:k]
r_vh = red_vh[0:k, :]
reconstructed_r = np.matmul(np.matmul(r_u, r_s), r_vh)

g_u = green_u[:, 0:k]
g_s = green_sigma[0:k, 0:k]
g_vh = green_vh[0:k, :]
reconstructed_g = np.matmul(np.matmul(g_u, g_s), g_vh)

b_u = blue_u[:, 0:k]
b_s = blue_sigma[0:k, 0:k]
b_vh = blue_vh[0:k, :]
reconstructed_b = np.matmul(np.matmul(b_u, b_s), b_vh)

reconstructed_matrix = (np.dstack((reconstructed_r,reconstructed_g,reconstructed_b)) * 255.999) .astype(np.uint8) 
plt.imshow(reconstructed_matrix)
plt.title("Reconstructed Image (Dimension = 150)")
plt.show()
