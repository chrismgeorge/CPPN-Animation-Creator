This code is refactored from:
`http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/`
`https://github.com/hardmaru/cppn-tensorflow`

The only major difference at this point is I removed the IPython dependency, and am now in a position where I feel comfortable editing the network, and adding new things onto it myself.

## Dependencies
I am currently using:
##### numpy '1.14.5'
##### tensorflow '1.11.0'
##### cv2 '3.4.1'
##### PIL '5.2.0'

## Run
`run.py` will run with default values for everything and output a video 720p video to `videos/please_name_me.mp4`.

## Arguments
`python3.6 run.py --x_dim=1080 --y_dim=720 --z_dim=12 --scale=8 --neurons_per_layer=12 --number_of_layers=7 --color_channels=3 --number_of_stills=4 --interpolations_per_image=12 --file_name=./videos/please_name_me.mp4`

#### `x_dim` : the width of the image/video
#### `y_dim` : the height of the image/video
#### `z_dim` : the dimension of the latent vector (larger means more complex images)
#### `scale` : how zoomed in on the coordiantes you are (larger means more zoomed in image)
#### `neurons_per_layer` : how many neurons on in each hidden layer of the neural network (larger means more complex images)
#### `number_of_layers` : how many hidden layers in the neural network (larger means more complex images)
#### `color_channels` : 3 for RGB images/video, 1 for black/white
#### `number_of_stills` : how many latent vectors will be made to create images for the interpolations
#### `interpolations_per_image` : how many images are between each "still"
#### `file_name` : the output file that the video will be saved to

## Todos
1) I'd like to add the ability to change the structure of the network even more, so allowing you to input a string like, 'tanh-softplus-tanh' or something, and then the network generate from said string.

***

2) Refactor and add in the ability to use a database to train said network, i.e. more refactoring of http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/
