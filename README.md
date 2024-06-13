# Supervised Deep Learning for Content-Aware Image Retargeting with Fourier Convolutions

Accepted in Multimedia Tools and Applications, Springer Journal
[[Springer Link](https://link.springer.com/article/10.1007/s11042-024-18876-8)]

<details>
    <summary>Abstract (click to view)</summary>
Image retargeting aims to alter the size of the image with attention to the contents. One of the main obstacles to training deep learning models for image retargeting is the need for a vast labeled dataset. Labeled datasets are unavailable for training deep learning models in the image retargeting tasks. As a result, we present a new supervised approach for training deep learning models. We use the original images as ground truth and create inputs for the model by resizing and cropping the original images. A second challenge is generating different image sizes in inference time. However, normal convolutional neural networks cannot generate images of different sizes than the input image. To address this issue, we introduced a new method for supervised learning. In our approach, a mask is generated to show the desired size and location of the object. Then the mask and the input image are fed to the network. Comparing image retargeting methods and our proposed method demonstrates the model’s ability to produce high-quality retargeted images. Afterward, we compute the image quality assessment score for each output image based on different techniques and illustrate the effectiveness of our approach.
</details>

<div>
    <img src="./Block diagram.png" width="90%">
</div>
(*Block diagram of the proposed method)

---

* [Dataset](https://iutbox.iut.ac.ir/index.php/s/6bLTj25fTWdCSy4)



## Citation
If you found this article helpful, please consider citing: 
```
Givkashi, M.H. et al.
Supervised deep learning for content-aware image retargeting with Fourier Convolutions.
Multimed Tools Appl (2024).
https://doi.org/10.1007/s11042-024-18876-8
```
