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



### board transition detection

in this series of lecture, a transition is explicitly white-out. we detect those white-out scenes.

blur detection : variance of laplacian : 동영상에 blur가 많으므로 버림

white-like metric detection : could misdetect human behavior (사람이 영상에서 차지하는 면적이 줄거나 그럴때)

- 상대적으로 dist/pre_dist 를 해봤지만 잘 안 먹힘



board change detection:

- board 색깔을 알아낸 뒤에 그 색깔로 얼만큼 pixel이 덮어씌워졌는지 조사
- 또는 그냥 board끼리의 euclidian difference 측정 : 일정 threshold

### board overlay

between those transitions, we overlay the board to get a full lecture board image.

if change occurred in board, we choose the latter one.

if significant change occurs, maybe we should detect them as a transition.

IoU 감지나 multi scale patch aggregation (lec10) 참고