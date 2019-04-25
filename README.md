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

# Workflow

* Use `make_many_models` to make a ton of different models and see an example test image in `./many_models/model_images`.
* Look through `./many_models/model_images` and find an image that you like.
* The image name (up to the .png) is the same as the model name. So if an image is named, `5-13-3-5_.png` then the corresponding model name is `5-13-3-5_`.
* From here you can now either make a random video in `run_edit.py` or move onto the gui.
* If you use the gui, you can make a specific custom animation, and then call `run_edit.py` with your specific `outfiles`.
* Please refer to the corresponding files below to see specifically how to call and use them.

## make_many_models
This is called from the command line with `python3.6 make_many_models.py`. When you want to modify what models are created simply edit these variables within the code itself.
* If you call the same code again, any models that have the same name will be overwritten :(.
```
# Variables for different images
x_dim = 1440
y_dim = 1080

color_channels = 1

# Edit these to make specific types of models
zs = list(range(9, 10))
neurons = list(range(15, 20))
layers = list(range(3, 5))

tests = [0, 1, 2, 3, 4, 5] # the different types of networks
scale = 24
```

## run_edit
* `run_edit.py` should be called after creating a model from `make_many_models.py`. Call this function and pass in the model name and the function you want to use (either `random_video` or `make_gui_video`).
* You need a file name for the video that will be created.
* If you are using `make_gui_video` then you need to pass in the outfiles comma seperated that were made from using the gui.
```
python3.6 run_edit.py --ml=model_name --fc=function_name --ofs=outfile_names,comma,seperated --fn=out_video_file_name.mp4
```

# gui
* After making a model with `make_many_models.py` you can open the gui by specifying the model name from the image you like, and an outfile, which will store the data.
* When using the gui, if you click `save data` then those parameters will be saved. When you press `pickle data` all of the data you have saved up to that point will be saved to the outfile name you specified. That name can then be used in `run_edit.py`.
* Pressing `pickle data` multiple times will override the previous save.
* If you make multiple outfiles with the same model (opening and closing the window with a new outfile name) then you can string all of those together into a single animations by adding them comma seperated when you call run_edit.py
```
python3.6 gui.py --ml=7-8-7-3_ --of=outfile
```

# masking_videos
This file will take two (black and white) videos and mask the first video with the second video. It needs the model name from one of the videos for getting the dimensions, and it will output the video into the `./videos/` directory.
```
python3.6 masking_videos.py --mn=main_video.mp4 --mk=masking_video.mp4 --ml=model_name --fn=output_video_name.mp4
```

