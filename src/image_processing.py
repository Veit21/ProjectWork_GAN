import cv2 as cv
import h5py
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import v2


class ImageProcessor:
    """
    A utility class for image processing tasks.

    This class provides static methods to perform various image processing operations,
    such as converting an RGB image to grayscale and converting a grayscale image to
    binary using thresholding.
    """

    def __init__(self):
        """
        Initialize the ImageProcessor class.
        """

    @staticmethod
    def grayscale_to_binary(an_image):
        """
        Convert a grayscale image to a binary image using a fixed threshold.

        This function takes a grayscale image as input and applies a fixed thresholding
        technique to convert it into a binary image. Pixels with intensity values greater
        than or equal to the threshold value will be set to the maximum intensity (white),
        while pixels with intensity values below the threshold will be set to the minimum
        intensity (black).

        :param an_image: A grayscale image (single-channel) to be converted to binary.
        :type an_image: numpy.ndarray

        :return: A binary image where pixels are either black or white based on the threshold.
        :rtype: numpy.ndarray

        :raises AssertionError: If the input image is None.
        """

        assert an_image is not None, "file could not be read."
        blurred_img = cv.GaussianBlur(an_image, (5, 5), 0)
        ret, image_black_white = cv.threshold(blurred_img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)    # thresh_binary_inv important for correct assignment of black/white
        image_black_white[image_black_white < 255] = 0  # per array masking make sure everything is 0 || 255
        return image_black_white

    @staticmethod
    def rgb_to_grayscale(an_rgb_image):
        """
        Convert an RGB image to grayscale using OpenCV's cvtColor function.

        This static method takes an RGB image as input and converts it to grayscale using
        OpenCV's cvtColor function with the COLOR_RGB2GRAY flag. The resulting grayscale
        image will have a single channel, representing intensity values.

        :param an_rgb_image: An RGB image to be converted to grayscale.
        :type an_rgb_image: numpy.ndarray

        :return: A grayscale image.
        :rtype: numpy.ndarray
        """

        # get dimensions of the image
        x_dim, y_dim, z_dim = an_rgb_image.shape

        # assert 3 color channels in order to correctly convert rgb --> grayscale
        if z_dim != 3:
            expanded_rgb_image = np.zeros((x_dim, y_dim, 3), dtype=an_rgb_image.dtype)
            expanded_rgb_image[..., :2] = an_rgb_image
        else:
            expanded_rgb_image = an_rgb_image

        # convert RGB to grayscale
        assert expanded_rgb_image.shape[2] == 3, "Number of channels of the input image does not match 3."
        grayscale_image = cv.cvtColor(expanded_rgb_image, cv.COLOR_RGB2GRAY)
        return grayscale_image

    @staticmethod
    def normalize(an_image):
        """
        Normalizes an image from [0, 255] to [-1, 1].

        :param an_image: Image to be normalized.
        :type an_image: numpy.ndarray

        :return: Normalized image.
        :rtype: numpy.ndarray
        """

        # make sure input image is a numpy array
        an_image_np = np.asarray(an_image)

        # normalize to [-1, 1]
        normalized_image = (an_image_np / 127.5) - 1
        return normalized_image

    @staticmethod
    def resize_image(an_image, scale_factor=1.25):
        """
        Resized an image by scale_factor

        :param an_image: Image to resize (PIL IMage).
        :param scale_factor: Factor by which to rescale the image. The higher the value, the more pixels have to be interpolated!
            Default is 1.25, a 25% upscale.
        :return: Resized image (NumPy array).
        """

        # get current and new size of image
        img_shape = np.asarray(an_image.size)
        img_shape_new = (img_shape * scale_factor).astype(np.uint16)

        # resize
        img_resized = np.asarray(an_image.resize(img_shape_new))
        return img_resized

    @staticmethod
    def crop(image_1, image_2, a_width, a_height, a_number_crops, a_minimum_separation, max_crops=False, single_rescale=False):
        """
        Crop two input images into multiple regions of specified width and height. The resulting images may overlap.
        To assure that both the raw and the corresponding binary images are cropped identically, the input awaits both
        the raw and the corresponding binary image.

        :param image_1: First image to crop (PIL Image).
        :param image_2: Second image to crop identically to image_1 (PIL Image).
        :param a_width: The width of the cropped regions.
        :param a_height: The height of the cropped regions.
        :param a_number_crops: The desired number of cropped regions. Only necessary if max_crops=False.
        :param a_minimum_separation: The minimum separation (in pixels) between the centers of adjacent cropped regions.
        :param max_crops: (Optional) If True, generate as many crops as possible while maintaining
                          the minimum separation. Overrides 'a_number_crops' if set to True.
        :param single_rescale: (Optional) If True, rescales the image a single time by 25% to try the cropping another time.
                                If the image is still too small, continues the routine w/o changes. Default is False.
        :return: A list of cropped regions (NumPy arrays). If no valid crops can be generated, returns (0, 0).
        """

        # Convert PIL Image to NumPy array
        image_1_arr = np.asarray(image_1)
        image_2_arr = np.asarray(image_2)

        # List with all cropped images
        cropped_images_1 = []
        cropped_images_2 = []

        # Get dimensions of the image
        assert image_1_arr.shape == image_2_arr.shape, "Input images don't have the same dimensions!"
        image_height, image_width = image_1_arr.shape

        # Calculate the maximum starting positions to ensure the selected region fits within the image
        max_x = image_width - a_width
        max_y = image_height - a_height

        # Single resize
        if (max_x or max_y) < 0 and single_rescale:
            print("Rescaling...")
            image_1_arr = ImageProcessor.resize_image(image_1)
            image_2_arr = ImageProcessor.resize_image(image_2)

            # Calculate the maximum starting positions again
            image_height, image_width = image_1_arr.shape
            max_x = image_width - a_width
            max_y = image_height - a_height

        # Make sure the picture is large enough to be cropped to the desired size
        if (max_y and max_x) > 0:

            # Check if original image should be cropped as often as possible
            if max_crops:

                # Calculate maximal number of crops depending on the smallest dimension of the image and the set minimum separation of the pixels
                if image_width < image_height:
                    a_number_crops = (image_width - a_width) // a_minimum_separation + 1
                else:
                    a_number_crops = (image_height - a_height) // a_minimum_separation + 1

            # Generate random starting positions with a minimum separation
            start_x = np.random.choice(np.arange(0, max_x, a_minimum_separation), a_number_crops, replace=False)
            start_y = np.random.choice(np.arange(0, max_y, a_minimum_separation), a_number_crops, replace=False)

            # Crop the randomly selected region and append to list
            for y, x in zip(start_y, start_x):
                selected_region_1 = image_1_arr[y:y + a_height, x:x + a_width]
                selected_region_2 = image_2_arr[y:y + a_height, x:x + a_width]
                cropped_images_1.append(selected_region_1)
                cropped_images_2.append(selected_region_2)
        else:

            # If original picture is too small for desired size
            return 0, 0

        if len(cropped_images_1) > 0 and len(cropped_images_2) > 0:
            return cropped_images_1, cropped_images_2
        else:

            # If original picture is too small for desired size
            return 0, 0

    @staticmethod
    def execute_cropping(an_import_path_raw, an_import_path_bin, an_output_path_raw, an_output_path_bin):
        """
        Executes the cropping to images in the raw and binary folders. Names are hardcoded, so only for the one local
        use this method is suitable.

        :param an_import_path_raw: Path to the raw images.
        :param an_import_path_bin: Path to the binary images.
        :param an_output_path_raw: Path where the cropped raw images shall be saved.
        :param an_output_path_bin: Path where the cropped binary images shall be saved.

        :raises: ValueError: If corresponding input images are not the same dimensions.
        """

        # import raw and bw images for cropping
        training_im_raw = [f for f in os.listdir(an_import_path_raw) if os.path.isfile(os.path.join(an_import_path_raw, f)) if
                           f.split(".")[1] == "tif" or f.split(".")[1] == "jpg"]

        # sort to assert correct order
        training_im_raw.sort()

        for i, img in enumerate(training_im_raw):
            print("\nImage {}/{}".format(i + 1, len(training_im_raw)))

            # load the image and get the name w/o file extension
            image = Image.open(os.path.join(an_import_path_raw, img)).convert('L')
            image_name = img.split(".")[0]
            image_name_bw = image_name + "_Probabilities_binary"    # be careful, name is hardcoded. Here suits its purpose.
            image_file_bw = image_name_bw + ".tif"
            image_bw = Image.open(os.path.join(an_import_path_bin, image_file_bw)).convert('L')

            try:

                # crop the image n times
                image_patches_raw, image_patches_bin = ImageProcessor.crop(image, image_bw, 2048, 1024,
                                                                           4, 750,
                                                                           max_crops=True, single_rescale=True)

                if image_patches_raw != 0 and image_patches_bin != 0:
                    print("For raw image, {} crops found!".format(len(image_patches_raw)))
                    print("For bin image, {} crops found!".format(len(image_patches_bin)))

                    counter = 1
                    for cropped_raw, cropped_bin in zip(image_patches_raw, image_patches_bin):

                        # Save the cropped images
                        cv.imwrite(os.path.join(an_output_path_raw, image_name + "_crop{}.tif".format(counter)),
                                   cropped_raw)
                        cv.imwrite(os.path.join(an_output_path_bin, image_name + "_crop{}.tif".format(counter)),
                                   cropped_bin)
                        counter += 1

            except ValueError:
                print("Not enough distinct integers can be generated in function 'ImageProcessor.crop()'."
                      "\nTry different resolution, less crops or smaller pixel separation. Otherwise, input"
                      "images don't have the same dimensions.")

    @staticmethod
    def execute_binary_conversion(an_import_path, an_output_path):
        """
        Executes the conversion of special h5 files, previously generated from original images using ilastik, to black &
        white binary images. This method is not very flexible and should only suit the purpose to start the routine locally.

        :param an_import_path: Path to the images to import.
        :param an_output_path: Path to where the binary images shall be saved.
        """

        # get the h5 files from path
        h5_files = [f for f in os.listdir(an_import_path) if
                    os.path.isfile(os.path.join(an_import_path, f)) and f.split(".")[1] == "h5"]
        num_files = len(h5_files)
        print(h5_files)
        assert num_files > 0, "List is empty."

        # loop over all probability files to generate binaries
        for i, h5_file in enumerate(h5_files):
            print("Converting image {}/{}".format(i + 1, num_files))

            # get only the name of the file w/o extension
            file_name = h5_file.split(".")[0]

            # import .h5 file
            f = h5py.File(os.path.join(an_import_path, h5_file), "r")
            img = f['exported_data']
            img_array = np.array(img)

            # convert original image to grayscale
            image_gray = ImageProcessor.rgb_to_grayscale(img_array)  # create image with dtype float32
            image_gray *= 255  # converts image from (0.0, 1.0) to (0, 255), important for correct assignment of pixels
            image_gray = image_gray.astype(np.uint8)  # change of dtype to uint8 necessary because grayscale to binary doesn't work with floats apparently?

            # generate binary (black white image)
            image_bw = ImageProcessor.grayscale_to_binary(image_gray)

            # safe images
            cv.imwrite(os.path.join(an_output_path, file_name + ".tif"), image_bw)

        print("Images converted to binary and saved.")

    @staticmethod
    def get_range(a_threshold, sigma=0.33):
        """
        Calculate the upper and lower boundary values for a given threshold value
        to be used in edge detection.

        :param a_threshold: The threshold value for edge detection.
        :param sigma: A scaling factor to determine the range.
                It is used to calculate the upper and lower boundaries around the threshold.
                A smaller `sigma` tightens the range, while a larger one widens it.
                Defaults to 0.33.
        :return: A tuple containing the lower and upper boundary values for the threshold range.
        """
        return (1 - sigma) * a_threshold, (1 + sigma) * a_threshold

    @staticmethod
    def image_to_edges(an_image, a_threshold, use_automatic_thresholding=False):
        """
        Convert an input image to edges using the Canny edge detection algorithm.

        :param an_image: The input image to be processed.
        :param a_threshold: The threshold value for edge detection. If
            `use_automatic_thresholding` is set to True, this threshold is
            overridden by the automatically calculated threshold using Otsu's method.
        :param use_automatic_thresholding: Whether to use automatic thresholding
            (Otsu's method) to determine the threshold. Defaults to False.
        :return: A NumPy array representing the edges detected in the image.
        """
        thresh = a_threshold

        # img from PIL Image to numpy array
        img_array = np.asarray(an_image)
        img_array = img_array.astype(np.uint8)

        # initial gaussian blur image
        img_blur = cv.GaussianBlur(img_array, (3, 3), 0)

        # get thresholds
        if use_automatic_thresholding:
            thresh, _ = cv.threshold(img_blur, 0, 255, cv.THRESH_OTSU)

        # get upper and lower boundary
        boundaries = ImageProcessor.get_range(thresh)

        # get edges
        img_edges = cv.Canny(img_blur, *boundaries)

        return img_edges

    @staticmethod
    def execute_image_to_edge_conversion(an_input_path, an_output_path):
        """
        Executes the conversion of images to edge maps.

        :param an_input_path: Path to the real images.
        :param an_output_path: Path where the edge images should be saved.
        """
        images = os.listdir(an_input_path)
        num_files = len(images)
        assert num_files > 0, "List of images is empty."

        # iterate over all images
        for i in range(num_files):
            image = images[i]
            if ".tif" in image or ".png" in image or ".jpg" in image:

                # get file name w/o extension
                file_name = image.split(".")[0]

                # read image and convert
                real_img = Image.open(os.path.join(an_input_path, images[i])).convert('L')
                img_edges = ImageProcessor.image_to_edges(real_img, 140, True)

                # safe images
                cv.imwrite(os.path.join(an_output_path, file_name + "_edges.tif"), img_edges)

        print("Images converted!")

    @staticmethod
    def distort_elastic_cv2(image1, image2=None, image3=None, alpha=80, sigma=20, random_state=None, grid_scale=1):
        """
        Apply elastic deformation to input images as described in [Simard2003].

        This function distorts the input image(s) using an elastic deformation technique. It creates random displacement fields
        and applies them to the input image(s) to introduce elastic-like distortions.

        Source:
        https://github.com/rwightman/tensorflow-litterbox/blob/ddeeb3a6c7de64e5391050ffbb5948feca65ad3c/litterbox/fabric/image_processing_common.py#L220

        :param image1: The first input image to be distorted.
        :param image2: The second input image to be distorted. Should have the same shape as `image1`.
            If provided, two distorted images will be returned as a tuple.
        :param image3: The third input image to be distorted. Should have the same shape as `image1` and `image2`.
            If provided, three distorted images will be returned as a tuple.
        :param alpha: Controls the intensity of distortion. Higher values result in more distortion.
        :param sigma: Controls the spatial smoothness of the distortion. Higher values result in smoother distortions.
        :param random_state: A random number generator to ensure reproducibility. If not
            provided, a new random state will be created.
        :param grid_scale: Controls the scale of the grid used for deformation. Smaller values result in finer
            distortions.
        :return: If `image2` is not provided, the distorted image as a numpy array. If `image2` is provided,
        a tuple containing two distorted images as numpy arrays.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape_size1 = image1.shape[:2]
        if image2 is not None:
            assert image1.shape == image2.shape, "Arguments image1 and image2 don't have the same shape!"
        if image3 is not None:
            assert image1.shape == image2.shape == image3.shape, "Arguments image1, image2 and image3 don't have the same shape!"

        # Downscaling the random grid and then upsizing post filter
        # improves performance. Approx 3x for scale of 4, diminishing returns after.
        alpha //= grid_scale  # Does scaling these make sense? seems to provide
        sigma //= grid_scale  # more similar end result when scaling grid used.
        grid_shape = (shape_size1[0] // grid_scale, shape_size1[1] // grid_scale)

        blur_size = int(4 * sigma) | 1
        rand_x = cv.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        rand_y = cv.GaussianBlur(
            (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
            ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        if grid_scale > 1:
            rand_x = cv.resize(rand_x, shape_size1[::-1])
            rand_y = cv.resize(rand_y, shape_size1[::-1])

        grid_x, grid_y = np.meshgrid(np.arange(shape_size1[1]), np.arange(shape_size1[0]))
        grid_x = (grid_x + rand_x).astype(np.float32)
        grid_y = (grid_y + rand_y).astype(np.float32)

        distorted_img1 = cv.remap(image1, grid_x, grid_y, borderMode=cv.BORDER_REFLECT_101, interpolation=cv.INTER_LINEAR)

        if image2 is None and image3 is None:
            return distorted_img1
        elif image3 is None and image2 is not None:
            distorted_img2 = cv.remap(image2, grid_x, grid_y, borderMode=cv.BORDER_REFLECT_101,
                                       interpolation=cv.INTER_LINEAR)
            return distorted_img1, distorted_img2
        elif image2 is None and image3 is not None:
            distorted_img3 = cv.remap(image3, grid_x, grid_y, borderMode=cv.BORDER_REFLECT_101, interpolation=cv.INTER_LINEAR)
            return distorted_img1, distorted_img3
        else:
            distorted_img2 = cv.remap(image2, grid_x, grid_y, borderMode=cv.BORDER_REFLECT_101, interpolation=cv.INTER_LINEAR)
            distorted_img3 = cv.remap(image3, grid_x, grid_y, borderMode=cv.BORDER_REFLECT_101, interpolation=cv.INTER_LINEAR)
            return distorted_img1, distorted_img2, distorted_img3

    @staticmethod
    def execute_elastic_distortion(path_imgs1, path_imgs2, path_imgs3, exp_path_imgs1, exp_path_imgs2, exp_path_imgs3):
        """
        Executes the elastic deformation for images in paths path_imgs1, path_imgs2, path_imgs3.

        :param path_imgs1: Path to images1.
        :param path_imgs2: Path to images2.
        :param path_imgs3: Path to images3.
        :param exp_path_imgs1: Path to where images1 should be saved.
        :param exp_path_imgs2: Path to where images2 should be saved.
        :param exp_path_imgs3: Path to where images3 should be saved.
        """

        # Read images
        imgs1 = sorted(os.listdir(path_imgs1))
        imgs2 = sorted(os.listdir(path_imgs2))
        imgs3 = sorted(os.listdir(path_imgs3))
        assert len(imgs1) == len(imgs2) == len(imgs3), "Number of images in the folders are not the same!"

        for i in range(len(imgs1)):
            if (".tif" or ".jpg" or ".png") in imgs1[i] and (".tif" or ".jpg" or ".png") in imgs2[i] and (".tif" or ".jpg" or ".png") in imgs3[i]:
                img1 = cv.imread(os.path.join(path_imgs1, imgs1[i]))
                img2 = cv.imread(os.path.join(path_imgs2, imgs2[i]))
                img3 = cv.imread(os.path.join(path_imgs3, imgs3[i]))

                # Get file names
                img1_name = imgs1[i].split(".")[0]
                img2_name = imgs2[i].split(".")[0]
                img3_name = imgs3[i].split(".")[0]

                # Deform
                deformed_image, deformed_image2, deformed_image3 = ImageProcessor.distort_elastic_cv2(img1, img2, img3,
                                                                                                      alpha=350,
                                                                                                      sigma=10,
                                                                                                      grid_scale=1)

                # Safe images
                cv.imwrite(os.path.join(exp_path_imgs1, img1_name + "_deformed.tif"), deformed_image)
                cv.imwrite(os.path.join(exp_path_imgs2, img2_name + "_deformed.tif"), deformed_image2)
                cv.imwrite(os.path.join(exp_path_imgs3, img3_name + "_deformed.tif"), deformed_image3)

        print("Images deformed!")

    @staticmethod
    def execute_rgb_to_grayscale(an_import_path, an_output_path):
        """
        Executes the conversion of rgb images to grayscale images on an entire folder of images.

        :param an_import_path: Path to the folder which contains the images to be converted.
        :param an_output_path: Path to the folder where the converted images are to be saved.
        """

        # List all files
        all_rgb = os.listdir(an_import_path)
        images_rgb = [x for x in all_rgb if "tif" in x and "synthesized" in x]
        N_images = len(images_rgb)

        # Loop over all files
        for i in range(N_images):

            # Read file and convert to grayscale
            img_rgb = cv.imread(os.path.join(an_import_path, images_rgb[i]))
            img_gs = ImageProcessor.rgb_to_grayscale(img_rgb)

            # Save grayscale image
            cv.imwrite(os.path.join(an_output_path, images_rgb[i]), img_gs)

    @staticmethod
    def execute_affine_warp(an_import_path, an_output_path):
        """
        Executes an affine warp on a set of images in a folder defined by an_import_path.

        :param an_import_path: Path to the folder which includes the images to be transformed.
        :param an_output_path: Path to the folder where the transformed images are to be saved.
        """

        # List all files
        all_images = os.listdir(an_import_path)
        images = [x for x in all_images if "tif" in x and "synthesized" in x]
        N_images = len(images)

        # Loop over all files
        for i in range(N_images):

            # Import image
            img = cv.imread(import_path)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # Get weight for x and y movement
            weight_xy = 0
            while weight_xy == 0:
                weight_xy = np.random.uniform(-0.125, 0.125)

            # Define matrix
            MAT = np.float32([
                [1, 0, weight_xy * w],
                [0, 1, weight_xy * h]
            ])

            # Shift image
            img_shifted = cv.warpAffine(img_rgb, MAT, (w, h), borderMode=cv.BORDER_REFLECT_101)

            # Save grayscale image
            cv.imwrite(os.path.join(an_output_path, images[i]), img_shifted)

    @staticmethod
    def execute_torch_transforms(an_import_path_img, an_import_path_label, an_import_path_edges, an_export_path_img, an_export_path_label, an_export_path_edges, transform="gaussian_blur"):
        """
        Executes the transforms given by torch functions. Gaussian blur, resize and crop and flipping are possible.

        :param an_import_path_img: Path to the image folder.
        :param an_import_path_label: Path to the label folder.
        :param an_import_path_edges: Path to the edge folder.
        :param an_export_path_img: Path to the folder where the transformed images are to be saved.
        :param an_export_path_label: Path to the folder where the transformed labels are to be saved.
        :param an_export_path_edges: Path to the folder where the transformed edges are to be saved.
        :param transform: Can take the arguments ["gaussian_blur", "random_resize_crop", "flip"], otherwise the function will break.
        """

        # Read images
        images = sorted(os.listdir(an_import_path_img))
        labels = sorted(os.listdir(an_import_path_label))
        edges = sorted(os.listdir(an_import_path_edges))

        # Loop over all images and do transform
        for img, label, edge in zip(images, labels, edges):
            if ".tif" in img and ".tif" in label and ".tif" in edge:

                # Get the names w/o file endings
                img_name = img.split(".")[0]
                label_name = label.split(".")[0]
                edge_name = edge.split(".")[0]

                # Read images
                img_PIL = Image.open(os.path.join(an_import_path_img, img))
                label_PIL = Image.open(os.path.join(an_import_path_label, label))
                edge_PIL = Image.open((os.path.join(an_import_path_edges, edge)))

                # Transform Image to Tensor
                img_Tensor = T.ToTensor()(img_PIL)
                label_Tensor = T.ToTensor()(label_PIL)
                edge_Tensor = T.ToTensor()(edge_PIL)

                # Concat images to apply the same transformation to all
                images_concat = torch.cat(
                    (img_Tensor.unsqueeze(0), label_Tensor.unsqueeze(0), edge_Tensor.unsqueeze(0)), 0)

                # Transform images and change format back to PIL
                if transform == "gaussian_blur":
                    transform_blur = v2.GaussianBlur(kernel_size=9, sigma=7)
                    transformed_imgs = transform_blur(images_concat)
                    file_ending = "_blurred.tif"

                if transform == "random_resize_crop":
                    transform_resize_crop = v2.RandomResizedCrop(size=(1024, 2048),
                                                                 scale=(0.35, 0.6),
                                                                 ratio=(2.0, 2.0))
                    transformed_imgs = transform_resize_crop(images_concat)
                    file_ending = "_cropped.tif"

                if transform == "flip":
                    transform_flipping = v2.Compose([
                        v2.RandomHorizontalFlip(p=1),
                        v2.RandomVerticalFlip(p=0.7),
                    ])
                    transformed_imgs = transform_flipping(images_concat)
                    file_ending = "_flipped.tif"

                else:
                    print("The given transformation is not known. Break.")
                    break

                img_PIL_transformed = T.ToPILImage()(transformed_imgs[0])
                label_PIL_transformed = T.ToPILImage()(transformed_imgs[1])
                edge_PIL_transformed = T.ToPILImage()(transformed_imgs[2])

                # save transformed images
                img_PIL_transformed.save(os.path.join(an_export_path_img, img_name + file_ending))
                label_PIL_transformed.save(os.path.join(an_export_path_label, label_name + file_ending))
                edge_PIL_transformed.save(os.path.join(an_export_path_edges, edge_name + file_ending))

if __name__ == "__main__":

    # Define I/O options
    import_path = ""    # insert path to the images to be edited
    export_path = ""    # insert path to where the edited images should be saved