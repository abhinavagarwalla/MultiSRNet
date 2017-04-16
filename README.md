# MultiSRNet

For Testing the model on your images, just use `upscale_image.py`.

Specify the input image path using `--image_path` argument, or a directory containing images using `--image-dir`.

Use `--scale` to specify upscaling factor (choose from 2,3,4)

Specify the directory to save upscaled images to using `--save_path`.

Example: `python upscale_image.py --image_dir='X2/' --scale=2 --save_path=outputs/`
