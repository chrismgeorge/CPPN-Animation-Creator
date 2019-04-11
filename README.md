This code is refactored from:
`http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/`
`https://github.com/hardmaru/cppn-tensorflow`

The only major difference at this point is I removed the IPython dependency, and am now in a position where I feel comfortable editing the network, and adding new things onto it myself.

# Dependencies
I am currently using:
* numpy '1.14.5'
* tensorflow '1.11.0'
* cv2 '3.4.1'
* PIL '5.2.0'

# Run
`python3.6 run.py` will run with default values for everything and output a video 720p video to `videos/please_name_me.mp4`.

# Arguments
`python3.6 run.py --x_dim=1080 --y_dim=720 --z_dim=12 --scale=8 --neurons_per_layer=12 --number_of_layers=7 --color_channels=3 --number_of_stills=4 --interpolations_per_image=12 --file_name=./videos/please_name_me.mp4`

* `x_dim` : the width of the image/video
* `y_dim` : the height of the image/video
* `z_dim` : the dimension of the latent vector (larger means more complex images)
* `scale` : how zoomed in on the coordiantes you are (larger means more zoomed in image)
* `neurons_per_layer` : how many neurons on in each hidden layer of the neural network (larger means more complex images)
* `number_of_layers` : how many hidden layers in the neural network (larger means more complex images)
* `color_channels` : 3 for RGB images/video, 1 for black/white
* `number_of_stills` : how many latent vectors will be made to create images for the interpolations
* `interpolations_per_image` : how many images are between each "still"
* `file_name` : the output file that the video will be saved to

# run_edit
Inside of run_edit.py you can modify it to do what you want, on top of changing arguements. Here are some things you can do.
* Save a specific model
```
cppn.save_model('new_model')
```
* Load a specific model
```
cppn.load_model('new_model')
```
* Save single random image
```
# With a specific model loaded
z = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
cppn.save_png(z, file_name)

# Can additionally save latent vector with
with open('outfile', 'wb') as f: # 'outfile' can be renamed
    pickle.dump([z.tolist()], f)
```
* Make 100 random images and save their latent vectors
```
# With a specific model loaded
z_vectors = []
for i in range(100):
    file_name = './photos/test%d.png' % i
    z_vectors = np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32)
    cppn.save_png(z_vectors, file_name)
    zs.append(z.tolist())
    print('Done! The image is at %s' % file_name)

with open('outfile', 'wb') as f: # 'outfile' can be renamed
    pickle.dump(zs, f)
```
You can view the images in `photos/` and you will see next how to retrieve the corresponding latent vectors.
* Re-display a specific image from a saved model.
```
# With a specific model loaded
with open ('outfile', 'rb') as fp: # 'outfile' can be renamed
    reloaded_vectors = pickle.load(fp)
z = np.array(reloaded_vectors[10]) # would be named test10.png
cppn.save_png(z, file_name) # make sure you name this something you'll remember
```
* Wibble around a specific image and make a video from it.
```
# With a specific model loaded
with open ('outfile', 'rb') as fp: # 'outfile' can be renamed
    reloaded_vectors = pickle.load(fp)
z_start = np.array(reloaded_vectors[10]) # would be named test10.png
zs = []
for i in range(10): # how many 'key frames' you want
    z = np.random.uniform(-.1, .1, size=(z_dim)).astype(np.float32)
    z = np.add(z_start, z)
    zs.append(z)
    print('Done! The image is at %s' % file_name)
cppn.save_mp4(zs, file_name) # make sure this is named correctly
```
* Make a random video
```
zs = [] # list of latent vectors
for i in range(number_of_stills):
    zs.append(np.random.uniform(-1.0, 1.0, size=(z_dim)).astype(np.float32))
cppn.save_mp4(zs, file_name)
```

* If you change any arguements, like color channel, or network size, then you're going to have to create a new model, save it, and reload it.

# Todos
* I'd like to add the ability to change the structure of the network even more, so allowing you to input a string like, 'tanh-softplus-tanh' or something, and then the network generate from said string.
* Refactor and add in the ability to use a database to train said network, i.e. more refactoring of http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/
