About
=====

This project recognize single & multiple object using linemode shape template. 

You can get the object location by recognizing given object template. 

I employ the multi-scale approach to accelerate the main computationally intensive part 

of the match procedure. To be precise, i reduce the amount of computation of response

map, which can reduce the running time to 1/3 of origial. In fact, the running time is 

~4ms with 300*200 matching image and 100*90 template image on ubuntu1604 

Core i7-8570H 2.2Ghz*12 8G. 

You can recognize multiple object if you give multiple object template. 

So you use it as one OCR method with some special tricks, which does not include 

this open source project(maybe also open source in future). Actually, 

i have tested the linemode ocr method on my own personal datasets. 

The test result show that the linemode ocr method can achieves 99.9% accuracy

for less complex typographic fonts of english charcter and number. 

Of cause, template match method cannot distinguish similar objects very well. 


Environments
=============

Ubuntu1604  OpenCV3.4.x 

sudo apt install libeigen3-dev


Build & Usage
==============

cd linemode_template_match

mkdir build

cmake ..

make -j8

./linemode_match_test path_to/datas/row_text_originals0 path_to/datas/ ocr path_to/datas/edges


recognize & get the location of chars of image

![image](https://github.com/NanKeRen2020/linemode_template_match/blob/main/datas/to_match0.png)

recognize & get the location of chars of image

![image](https://github.com/NanKeRen2020/linemode_template_match/blob/main/datas/to_match1.png)

match the location of the template on image

![image](https://github.com/NanKeRen2020/linemode_template_match/blob/main/datas/locate_text_area.png)




References
==========

[1] https://github.com/meiqua/shape_based_matching.

[2] S. Hinterstoisser et al. Gradient Response Maps for Real-Time Detection of Textureless Objects. 

IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 5, pp. 876-888, May 2012.


