# Vision-Project
Convert lecture video to lecture pdf



## level1

**fixed camera, fullscreen board, explicit board transition**



mainly videos from Rao IIT Academy : 

`https://www.youtube.com/channel/UCG0qvp06I-LULSJ7sTw8YoQ`



###human detection by semantic segmentation

https://gluon-cv.mxnet.io/build/examples_segmentation/demo_fcn.html



deeplab_resnet152_voc : highest segmentation score in pascal voc

- https://gluon-cv.mxnet.io/model_zoo/segmentation.html

- class list : http://host.robots.ox.ac.uk:8080/anonymous/XZEXL2.html
  - 0 : background
  - 15 : person





gpu error:

- https://discuss.mxnet.io/t/python-stopped-working-error-on-gpu/1737
- http://forthenextstep.tistory.com/29
- https://stankirdey.com/2017/03/09/installing-mxnet-deep-learning-framework-on-windows-10/
- could not resolve, just used linux





we ignore the human part when making pdf.

### board transition detection

in this series of lecture, a transition is explicitly white-out. we detect those white-out scenes.



### board overlay

between those transitions, we overlay the board to get a full lecture board image.

if change occurred in board, we choose the latter one.

if significant change occurs, maybe we should detect them as a transition.