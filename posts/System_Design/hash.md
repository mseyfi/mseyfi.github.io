pHash, ahash, and dhash are all types of **perceptual image hashing** algorithms. Unlike cryptographic hashes that change drastically with a tiny input change, these hashes are designed to be similar if the images look similar to the human eye, making them excellent for finding duplicate or near-duplicate images.

Here’s a breakdown of each:

***

### Average Hash (ahash)

The **average hash** is the simplest of the three. It creates a "fingerprint" of an image based on its average pixel value. It's fast but less accurate than the others.

**How it works:**
1.  **Resize the image:** The image is first scaled down to a small, fixed size, typically 8x8 pixels, to ignore details and focus on the overall structure.
2.  **Convert to grayscale:** The color information is removed to simplify the data.
3.  **Calculate the average pixel value:** The algorithm computes the average value of all 64 pixels.
4.  **Create the hash:** Each pixel is compared to the average. If the pixel is brighter than or equal to the average, it's assigned a `1`. If it's darker, it's assigned a `0`. This creates a 64-bit binary string, which is the ahash.



***

### Difference Hash (dhash)

The **difference hash** is generally more robust than ahash and still very fast. Instead of comparing pixels to a single average, it compares adjacent pixels to identify gradients or changes in brightness.

**How it works:**
1.  **Resize and grayscale:** Like ahash, the image is first resized (e.g., to 9x8 pixels) and converted to grayscale.
2.  **Compare adjacent pixels:** The algorithm iterates through the pixels, comparing each pixel to the one immediately to its right.
3.  **Create the hash:** If the pixel on the left is brighter than the one on the right, it's assigned a `1`; otherwise, it's a `0`. This comparison across each row generates a 64-bit hash. A vertical version is also possible, comparing pixels in adjacent rows.

This method is better at resisting minor brightness and contrast adjustments.



***

### Perceptual Hash (phash)

The **perceptual hash** is the most sophisticated and accurate of the three. It uses the **Discrete Cosine Transform (DCT)**, an algorithm also used in JPEG compression, to identify the image's most important low-frequency structures.

**How it works:**
1.  **Resize and grayscale:** The image is resized to a larger size than in ahash or dhash (e.g., 32x32) and converted to grayscale.
2.  **Apply DCT:** The DCT is performed on the 32x32 grayscale image. This transforms the image from a spatial domain to a frequency domain, concentrating the most significant visual information into the top-left corner of the DCT matrix.
3.  **Reduce the DCT:** The algorithm keeps only the top-left 8x8 portion of the DCT matrix, which represents the lowest frequencies of the image.
4.  **Calculate the median:** Instead of the mean (like ahash), the median value of these 64 DCT coefficients is calculated.
5.  **Create the hash:** A `1` is assigned if a DCT coefficient is greater than or equal to the median, and a `0` if it's less. This produces the final 64-bit phash.

pHash is significantly more resilient to changes in gamma, color adjustments, and minor distortions.



### Key Differences Summarized

| Feature | **Average Hash (ahash)** | **Difference Hash (dhash)** | **Perceptual Hash (phash)** |
| :--- | :--- | :--- | :--- |
| **Speed** | Fastest | Very Fast | Slowest |
| **Accuracy** | Good | Better | Best |
| **Method** | Compares pixels to the average value. | Compares adjacent pixels for gradients. | Analyzes image frequencies using DCT. |
| **Robustness** | Vulnerable to brightness/contrast changes. | More robust than ahash. | Very robust against various modifications. |

To compare two images, you calculate their respective hashes and then compute the **Hamming distance** between the two binary strings—that is, the number of bit positions that are different. A smaller distance means the images are more similar.
