
import os
from typing import Tuple

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import warnings
from scipy.special import comb
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from utils import str2bool

def generate_Image_from_ndarray(image):
    try: # 3 channel
        im = Image.fromarray((image).astype(np.uint8), mode='RGB')
    except: # 1 channel
        im = Image.fromarray((image.squeeze()).astype(np.uint8), mode='L')
    return im 

class Element:
    def __init__(self, size:Tuple[int,...]=None, color:Tuple[int,...]=None, position:Tuple[int,...]=None):
        self.size = size if size is not None else None
        self.row1, self.row2 = None, None
        self.col1, self.col2 = None, None 
        self.color = None

        if color is not None:
            self.assign_color(color)

        if position is not None and size is not None:
            self.assign_position(position)

    def assign_color(self, color:Tuple[int,...]):
        assert isinstance(color,tuple), "The color must be a tuple. If there is only one channel, please pass in a 1-tuple."
        assert all([item%1==0 for item in color]), "The color for each channel should be an integer in value!" 
        if color is not None: self.color = color if len(color)>1  else int(color[0]) 

    def assign_position(self, position) -> None:
        assert self.size is not None, "Please assign size to an element first before assigning its position!"
        self.row1, self.col1 = position
        self.row2 = self.row1+self.size[0]-1  
        self.col2 = self.col1+self.size[1]-1

    def draw(self, *args, **kwargs):
        raise NotImplementedError

class GroupOfElements():
    """The objects generated using this class will be one row of elements with the same size and color."""
    def __init__(self, elements: Tuple[Element,...], element_size: Tuple[int,...], color: Tuple[int,...], spacing: int, position: Tuple[int,...]=None): 
        self.elements = elements # Note: elements might not have position info in themselves, the positions can be assigned later
        self.element_size = element_size# currently, only support all the elements to have the same size
        self.spacing = spacing # The space between the adjacent two elements in this group
        self.row1, self.row2 = None, None
        self.col1, self.col2 = None, None 
        self.assign_size() # update the size field of GroupOfElements and all the elements within
        self.assign_color(color) # update the color field of GroupOfElements and all the elements within
        if position is not None: self.assign_position(position) # update the position-related fields of GroupOfElements and all the elements within
    
    def assign_color(self, color: Tuple[int,...]):
        """update the color fields of GroupOfElements and all the elements within"""
        assert isinstance(color,tuple), "The color must be a tuple. If there is only one channel, please pass in a 1-tuple."
        assert all([item%1==0 for item in color]), "The color for each channel should be an integer in value!" # data type might not be correct, but must be an integer in value!
        self.color = color if len(color)>1  else int(color[0]) # the color field might not be useful if the element is part of an GroupOfElements
        for element in self.elements:
            element.assign_color(color)

    def assign_size(self, ):
        """update the size field of GroupOfElements and all the elements within"""
        for element in self.elements: element.size = self.element_size
        element_num = len(self.elements)
        row_num = self.element_size[0]
        col_num = element_num*self.element_size[1] + (element_num-1)*self.spacing
        self.size = (row_num, col_num)

    def assign_position(self, position: Tuple[int,...]):
        """ calculate own position and update the position of all elements. Note, this function requires that the size of the elements are determined already."""
        self.row1, self.col1 = position
        self.row2 = self.row1+self.size[0]-1  
        self.col2 = self.col1+self.size[1]-1

        for i, element in enumerate(self.elements):
            element_position = (position[0], position[1]+i*(element.size[1]+self.spacing ))
            element.assign_position(element_position)

    def draw(self, image,) -> ndarray:
        for element in self.elements:
            image = element.draw(image)
        return image

class Rectangle(Element):
    def __init__(self, size: Tuple[int,...]=None, color: Tuple[int,...]=None, position:Tuple[int,...] = None, ):
        super(Rectangle,self).__init__( size, color, position=position) #type:ignore

    def __repr__(self) -> str:
        return 'Rectangle'

    def draw(self, image, ) -> ndarray:
        im = generate_Image_from_ndarray(image)
        image_draw = ImageDraw.Draw(im)
        image_draw.rectangle(xy=((self.col1, self.row1), (self.col2, self.row2)), fill=self.color, ) #type:ignore
        image = np.asarray(im).reshape(image.shape[0],image.shape[1],-1) # for 1-channel dataset, we need to add 1 dim to the end
        return image

class VerticalGradientRectangle(Element):
    """This element is useful in the dataset GradOcclusion."""
    def __init__(self, size: Tuple[int,...]=None, color: Tuple[int, ...]=None, position:Tuple[int,...] = None, end_color_ratio = 0.5 ):
        super(VerticalGradientRectangle,self).__init__( size, color, position=position) #type:ignore
        self.end_color_ratio = end_color_ratio

    def __repr__(self) -> str:
        return 'Vertical_Gradient_Rectangle'

    def draw(self, image,) -> ndarray:
        assert image.ndim == 3 ,"Image should have 3 dimensions: (size, size, channel)"
        assert self.size[0] > 1, "If we desire to have vertical gradient, then the vertical height should be larger than 1."
        # first generate a patch of coeffients (fade_mask) which will later be multiplied with the color
        # to determined the color finally painted in the shape. fade_mask has a vertical gradient.
        fade_mask_row_ls = []
        for j in range(self.size[0]):
            coefficient = self.end_color_ratio + j/(self.size[0]-1) * (1-self.end_color_ratio)
            fade_mask_row = np.ones((1,self.size[1]))*coefficient
            fade_mask_row_ls.append(fade_mask_row)
        fade_mask= np.expand_dims(np.concatenate(fade_mask_row_ls, axis = 0),axis=2)
        # multiply the fade_mask with the color.
        colored_shape_patch = fade_mask * np.array(self.color).reshape(1,1,image.shape[-1]) # broadcasting
        image[self.row1:self.row2+1, self.col1:self.col2+1] = colored_shape_patch
        return image

class HorizontalGradientRectangle(VerticalGradientRectangle):
    def __init__(self, size: Tuple[int,...]=None, color: Tuple[int, ...]=None, position:Tuple[int,...] = None, end_color_ratio = 0.5 ):
        super(HorizontalGradientRectangle,self).__init__( size, color, position=position, 
                                                         end_color_ratio = end_color_ratio ) #type:ignore

    def __repr__(self) -> str:
        return 'Horizontal_Gradient_Rectangle'

    def draw(self, image,) -> ndarray:
        """quite similar to the same function in VerticalGradientRectangle"""
        assert image.ndim == 3 ,"image should have 3 dimensions: (size, size, channel)"
        assert self.size[1] > 1, "If we desire to have horizontal gradient, then the horizontal width should be larger than 1."
        # The drawing process is quite similar to that in VerticalGradientRectangle().
        fade_mask_col_ls = []
        for j in range(self.size[1]):
            coefficient = self.end_color_ratio + j/(self.size[1]-1) * (1-self.end_color_ratio)
            fade_mask_col= np.ones((self.size[0],1))*coefficient
            fade_mask_col_ls.append(fade_mask_col)
        fade_mask= np.expand_dims(np.concatenate(fade_mask_col_ls, axis = 1),axis=2)
        colored_shape_patch = fade_mask * np.array(self.color).reshape(1,1,image.shape[-1]) # broadcasting
        image[self.row1:self.row2+1, self.col1:self.col2+1] = colored_shape_patch
        return image

class Line():
    """Simply adapting the Image.Draw.Line object for convenience."""
    def __init__(self, xy, color:Tuple[int,...], width ):
        self.xy = xy
        assert isinstance(color,tuple),"The color must be a tuple. If there is only one channel, please pass in a 1-tuple."
        self.color = color if len(color)>1 else int(color[0])
        self.width = width

    def __repr__(self) -> str:
        return 'Line'

    def draw(self, image) -> ndarray:
        im = generate_Image_from_ndarray(image)
        image_draw = ImageDraw.Draw(im)
        image_draw.line(xy=self.xy, fill=self.color, width = self.width)
        image = np.asarray(im).reshape(image.shape[0],image.shape[1],-1) # for 1-channel dataset, we need to add 1 dim to the end
        return image

class Ellipse(Element):
    """Simply adapting the Image.Draw.ellipse object for convenience."""
    def __init__(self, size: Tuple[int,...]=None, color: Tuple[int, ...]=None, position:Tuple[int,...] = None, ):
        super(Ellipse,self).__init__( size, color, position=position) #type:ignore

    def __repr__(self) -> str:
        return 'Circle'

    def draw(self, image) -> ndarray:
        im = generate_Image_from_ndarray(image)
        image_draw = ImageDraw.Draw(im)
        image_draw.ellipse(xy=((self.col1, self.row1), (self.col2, self.row2)), fill=self.color, ) #type:ignore
        image = np.asarray(im).reshape(image.shape[0],image.shape[1],-1) # for 1-channel dataset, we need to add 1 dim to the end
        return image

class Arc(Element):
    """Simply adapting the Image.Draw.arc object for convenience."""
    def __init__(self, size: Tuple=None, color: Tuple[int, ...]=None, start = 0, end = 360, position = None, width = 5):
        super(Arc,self).__init__( size, color, position=position) #type:ignore
        self.start = start
        self.end = end
        self.width = width

    def __repr__(self) -> str:
        return 'Arc'
    
    def draw(self, image) -> ndarray:
        im = generate_Image_from_ndarray(image)
        image_draw = ImageDraw.Draw(im)
        image_draw.arc(xy=((self.col1, self.row1), (self.col2, self.row2)),
                           start = self.start, end = self.end, fill=self.color, width=self.width) #type:ignore
        image = np.asarray(im).reshape(image.shape[0],image.shape[1],-1) # for 1-channel dataset, we need to add 1 dim to the end
        return image

class PieSlice(Arc):
    """Simply adapting the Image.Draw.pieslice object for convenience."""
    def __init__(self, size: Tuple[int,...]=None, color: Tuple[int, ...]=None, start = 0, end = 360, position = None, width = 5):
        super(PieSlice,self).__init__( size, color, start = start, end = end, position=position, width = width) #type:ignore
    
    def __repr__(self) -> str:
        return 'Arc'

    def draw(self,image) -> ndarray:
        im = generate_Image_from_ndarray(image)
        image_draw = ImageDraw.Draw(im)
        image_draw.pieslice(xy=((self.col1, self.row1), (self.col2, self.row2)),
                           start = self.start, end = self.end, fill=self.color, width=self.width) #type:ignore
        image = np.asarray(im).reshape(image.shape[0],image.shape[1],-1) # for 1-channel dataset, we need to add 1 dim to the end
        return image

class BaseDataset:
    """ A base class indicative of the general structure of the actual dataset classes.
    """
    def __init__(self,
                 image_size: int,
                 channel_num: int,
                 savedir_name: str,
                 visualize: bool = True):
        """
        """
        self.image_size = image_size
        self.channel_num = channel_num
        self.savedir_name= savedir_name
        self.visualize = visualize
        self.train_val_dataset_generated = False


    def generate_train_val_dataset(self):
        """Generate training and validation dataset"""
        raise NotImplementedError
    
    def generate_test_dataset(self):
        """Generate test dataset (usually just special cases) including the corresponding masks together with other
        special information for future use"""
        raise NotImplementedError

    def generate_image(self):
        """Generate one image given enough information to determine the shapes, colors, and postitions
        to describe one image."""
        raise NotImplementedError

    def generate_mask(self):
        """Generate one mask given the positions of the shapes in the image."""
        raise NotImplementedError

    def generate_detailed_mask(self):
        """Sometimes, certain datasets have detailed masks. In detailed masks, different parts
        of shapes or backgrounds are marked differently."""
        raise NotImplementedError

class ClosureDataset(BaseDataset):
    def __init__(self, channel_num=3, image_size = 64,
                  dataset_name = 'dClosure', suffix = "",visualize = True):
        super(ClosureDataset, self).__init__(image_size=image_size,channel_num=channel_num,#type:ignore
                                                   savedir_name=dataset_name+suffix,
                                                   visualize=visualize,)

    def generate_detailed_mask(self, square_size, square_position, square_margin_width = 2, image_size=64, ):
        mask = np.zeros((image_size, image_size))
        # draw a first square, whose rim serves as an outline
        outline_square = Rectangle((square_size,square_size),(1,),position=square_position,)
        mask = outline_square.draw(mask)

        # draw a second square inside the first one, whose color is the color of the square
        inner_square_position = [p+square_margin_width for p in square_position]
        inner_square = Rectangle((square_size-2*square_margin_width, square_size-2*square_margin_width),(2,), position=inner_square_position,)
        mask = inner_square.draw(mask)

        return mask.squeeze()

    def generate_mask(self, square_size, square_position, image_size = 64, ):
        mask = np.zeros((image_size, image_size))
        square = Rectangle((square_size,square_size),(1,),position=square_position,)
        mask=square.draw(mask).squeeze()
        return mask

    def generate_train_val_dataset(self, train_image_num= 30000, valid_image_num = 300, test_image_num = 300,
                        bg_color_min=0, bg_color_max = 4/7, square_color_min = 3/7, square_color_max = 1,
                        outline_color_ratio=(0.4,0.5,0.7),
                        min_square_size = 40, max_square_size = 40,
                        square_margin_width = 2, downsampling = False, downsampling_factor = 4,
                        seed = 42):

        # The following assertion makes sure that the color ranges of squares and backgrounds overlap.
        assert bg_color_min <= square_color_min <= bg_color_max <= square_color_max,"We require: bg_color_min <= square_color_min <= bg_color_max <= square_color_max"
        # register some fields for test dataset generation
        self.test_color_min, self.test_color_max = square_color_min, bg_color_max
        self.outline_color_ratio = outline_color_ratio
        self.min_square_size, self.max_square_size = min_square_size, max_square_size
        self.square_margin_width = square_margin_width
        self.train_val_seed = seed
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor

        rng = np.random.default_rng(seed = seed)

        print(f"======generating {self.savedir_name} training & validation dataset======")
        image_ls = []
        test_mask_ls = []
        test_detailed_mask_ls = []
        for i in tqdm(range(train_image_num+valid_image_num+test_image_num)):
            square_size = rng.integers(self.min_square_size, self.max_square_size+1)
            coordinate_min = 0
            coordinate_max = self.image_size-square_size
            square_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]
            
            # [Generate image] This dataset does not have a generate_image function like the others.
            # The images are generated by filling in colors into the detailed masks. This is to make 
            # sure that the color ratio between relevant parts of the images are strictly enforced.
            # If we draw the images as in other datasets using generate_image function, then the colors
            # will need to be rounded up to cater to the requirement of the pillow package, in which 
            # RGB mode only takes in 8-bit integers. In that case, the color ratio will not be exactly as stipulated, 
            # which will compromise the models' performance to combine covarying parts of the images together.
            detailed_mask = self.generate_detailed_mask(square_size = square_size, square_position=square_position,
                                                        square_margin_width=square_margin_width, image_size = self.image_size)
            bg_color = rng.integers(np.ceil(bg_color_min*255), np.floor(bg_color_max*255)+1, self.channel_num)/255
            square_color = rng.integers(np.ceil(square_color_min*255), np.floor(square_color_max*255)+1, self.channel_num)/255
            # The above colors are chosen to be n/255, which works together with [Color trick] (see below) to make sure that
            # the training/validation/testing datasets do not have the same images.
            outline_color = square_color*np.asarray(outline_color_ratio)
            image = np.zeros((self.image_size, self.image_size, self.channel_num))
            image[detailed_mask == 0] = bg_color
            image[detailed_mask == 1] = outline_color
            image[detailed_mask == 2] = square_color            
            
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if not self.downsampling: image = image.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            image_ls.append(image)
            if i >= train_image_num + valid_image_num:
                detailed_mask = np.expand_dims(detailed_mask, axis = 0)
                if self.downsampling: detailed_mask = detailed_mask[:,::self.downsampling_factor,::self.downsampling_factor]
                test_detailed_mask_ls.append(detailed_mask)

                mask = self.generate_mask(square_size = square_size, square_position=square_position, 
                                          image_size=self.image_size)
                mask = np.expand_dims(mask, axis = 0)
                if not self.downsampling: mask = mask.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
                test_mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        test_masks = np.concatenate(test_mask_ls, axis = 0)
        test_detailed_masks = np.concatenate(test_detailed_mask_ls, axis = 0)
        # [Color trick] In the following, train_images and valid_images are multiplied with 999/1000 and 
        # 998/1000, such that the training set, validation set, and testing set do not overlap.
        # This is achieved by guaranteeing that the colors showing up in the three sets do not overlap.
        # E.g. the possible colors in the test set are 0,1/255,2/255,...,1. The colors in the 
        # training set are these numbers times 999/1000. 
        # Note, if train/valid/test set admits samples with all pixels being zero, then the three
        # sets can still overlap. But we avoided this in the design of this dataset.
        train_images = images[:train_image_num]*999/1000  
        valid_images = images[train_image_num:train_image_num+valid_image_num]*998/1000
        test_images = images[train_image_num+valid_image_num:]
        # [Testing samples] Note, even though this function is for generating training and validation
        # sets, we still generate test_images alongside. These testing samples will be saved in 
        # test_1.npz. These samples are the ones sampled from the same distribution in which 
        # the training and validation sets are sampled. These test samples are not used in the end.
        # But we generate them for future convenience. Testing samples generated in the 
        # generate_test_dataset() function are the specially-designed ones where special 
        # visual processing effects show up. 

        print("saving ...")
        save_train_val_dataset(savedir_name=self.savedir_name, train_images=train_images, 
                               valid_images=valid_images, test_images=test_images, test_masks = test_masks,
                               test_detailed_masks=test_detailed_masks, visualize=self.visualize)

        self.train_val_dataset_generated = True

    def generate_test_dataset(self, test_image_num = 500, closure_rate = 0.5, 
                              random_square_size_min = 5, random_square_size_max = 10, random_square_num = 20,
                              seed = 41):
        assert self.train_val_dataset_generated, "Need to generate training and validation dataset first!"
        if seed == self.train_val_seed: warnings.warn("It is highly recommended that you use different seeds to generate the training/validation dataset and the testing dataset.")

        rng = np.random.default_rng(seed = seed)

        print(f"======generating {self.savedir_name} testing dataset======")
        image_ls = []
        mask_ls = []
        detailed_mask_ls = []
        for i in tqdm(range(test_image_num)):
            # in this case, square and background has the same color
            square_size = rng.integers(self.min_square_size, self.max_square_size+1)
            coordinate_min = 0
            coordinate_max = self.image_size-square_size
            square_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]
            detailed_mask = self.generate_detailed_mask(square_size = square_size, square_position=square_position,
                                                        square_margin_width=self.square_margin_width, image_size = self.image_size)
            square_bg_color = rng.integers(np.ceil(self.test_color_min*255), np.floor(self.test_color_max*255)+1, self.channel_num)/255
            outline_color = square_bg_color*np.asarray(self.outline_color_ratio)
            image = np.zeros((self.image_size, self.image_size, self.channel_num))
            image[detailed_mask == 0] = square_bg_color 
            image[detailed_mask == 1] = outline_color
            image[detailed_mask == 2] = square_bg_color
            
            if (rng.uniform()>=closure_rate):
                # image = self.random_set_to_background(image, outline_erasure_rate, square_bg_color, rng)
                image= self.random_add_bg_colored_square(image,
                                                          square_size_min = random_square_size_min,
                                                          square_size_max = random_square_size_max,
                                                          square_num = random_square_num, 
                                                          square_bg_color = tuple(square_bg_color),
                                                          rng = rng)


            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if not self.downsampling: image = image.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            image_ls.append(image)
            mask = self.generate_mask(square_size = square_size, square_position=square_position, 
                                        image_size=self.image_size)
            mask = np.expand_dims(mask, axis = 0)
            if not self.downsampling: mask = mask.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            mask_ls.append(mask)

            # save the detailed masks
            detailed_mask = np.expand_dims(detailed_mask, axis = 0)
            if not self.downsampling: mask = mask.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            detailed_mask_ls.append(detailed_mask)


        images = np.concatenate(image_ls, axis = 0)
        masks = np.concatenate(mask_ls, axis = 0)
        detailed_masks = np.concatenate(detailed_mask_ls, axis = 0)

        print("saving ...")
        save_test_dataset(savedir_name = self.savedir_name, test_image_num=test_image_num,
                          images = images, masks = masks, detailed_masks = detailed_masks, 
                          visualize = self.visualize)
    
    def random_add_bg_colored_square(self, image, square_size_min, square_size_max, square_num, 
                                     square_bg_color, rng ):
        """randomly cover the testing image with squares that has the same color as the background"""
        mask = np.zeros((self.image_size,self.image_size))

        for i in range(square_num):
            square_size = rng.integers(square_size_min, square_size_max+1)
            position = tuple(rng.integers(0, self.image_size-square_size, size = 2))
            square = Rectangle((square_size, square_size), (1,), position)
            mask = square.draw(mask)
        mask = mask.squeeze()

        image[mask == 1] = square_bg_color        

        # The following is to make sure that there does not exist part of the 
        # square sides erased partially in width.
        assert self.image_size % self.square_margin_width == 0, "Only support image_size to be multiples of square_margin_width"
        image = image[::self.square_margin_width, ::self.square_margin_width,:]
        image = image.repeat(self.square_margin_width,axis=0).repeat(self.square_margin_width,axis=1)

        return image 

class ContinuityDataset(BaseDataset):
    def __init__(self, channel_num=3, image_size = 256,
                  dataset_name= 'dContinuity', suffix = "", visualize = True):
        super(ContinuityDataset, self).__init__(image_size=image_size,channel_num=channel_num,#type:ignore
                                                   savedir_name=dataset_name+suffix,
                                                   visualize=visualize,)

    def generate_image(self, circle_color, circle_position, bg_color = np.array((255,255,255)), circle_size = 100, width = 10,
                        image_size=64, channel_num = 3):
        image = np.ones((image_size,image_size,channel_num))*bg_color.reshape(1,1,self.channel_num)
        # draw the circle
        circle = Arc((circle_size, circle_size), tuple(circle_color), 0 , 360, circle_position, width = width)
        image = circle.draw(image)

        image = image/255

        return image

    def generate_mask(self, circle_position, circle_size = 25, width = 2, image_size=64, ):
        mask = np.zeros((image_size,image_size,))
        # draw the circle
        circle = Arc((circle_size, circle_size), (1,), 0 , 360, circle_position, width = width)
        mask = circle.draw(mask).squeeze()
        return mask 

    def generate_detailed_mask(self, circle_position, circle_size = 25, width = 2, image_size=64, arc_angle = 30):
        """In this function, a more detailed mask is generated. The masks differentiate between the 
        gaps in the circle and the actually painted part of the circle."""
        detailed_mask = np.zeros((image_size,image_size,))
        # draw the whole circle
        circle = Arc((circle_size, circle_size), (1,), 0 , 360, circle_position, width = width)
        detailed_mask = circle.draw(detailed_mask).squeeze()

        # draw the actually painted circle
        assert arc_angle <= 60
        # draw the fragmented circle
        covered_angle = 0
        while covered_angle < 360:
            # 6 arcs in total, every arc occupies arc_angle degrees
            circle_arc = Arc((circle_size, circle_size), (2,), covered_angle , 
                             covered_angle+arc_angle, circle_position, width = width)
            detailed_mask = circle_arc.draw(detailed_mask)
            covered_angle += 60

        return detailed_mask 

    def generate_test_image(self, circle_color, random_arc_color_1, random_arc_color_2, circle_position, bg_color = np.array((255,255,255)), circle_size = 80, width = 6,
                        random_frag_num_min = 5, random_frag_num_max = 20,  arc_angle = 30,
                        image_size=64, channel_num = 3, rng=None):
        # Draw six arcs which compose one circle, then draw two sets of randomly-positioned
        # arcs. Random arcs in one set share the same color.
        assert arc_angle <= 60
        image = np.ones((image_size,image_size,channel_num))*bg_color.reshape(1,1,channel_num)
        # draw the fragmented circle
        covered_angle = 0
        while covered_angle < 360:
            # 6 arcs in total, every arc occupies arc_angle degrees
            circle_arc = Arc((circle_size, circle_size), tuple(circle_color), covered_angle , 
                             covered_angle+arc_angle, circle_position, width = width)
            image = circle_arc.draw(image)
            covered_angle += 60
        
        # draw some random fragments with different colors
        random_frag_num = rng.integers(random_frag_num_min, random_frag_num_max)
        for i in range(random_frag_num):
            color_choices = [random_arc_color_1, random_arc_color_2]
            arc_color = color_choices[rng.choice(len(color_choices), p = [0.5,0.5])]
            arc_start_angle = rng.integers(0,360+1)
            # Note: the arc_circle_position allows negative number at each coordinate. 
            # Because even if the upper left corner of one circle is located outside the image we are drawing,
            # the arc of this circle can still show up in the image, although not guaranteed.
            arc_circle_position = [rng.integers(-(circle_size), self.image_size+1) for k in range(2)] 
            circle_arc = Arc((circle_size, circle_size), tuple(arc_color), arc_start_angle , arc_start_angle+arc_angle, 
                             arc_circle_position, width = width)
            image = circle_arc.draw(image)

        image = image/255

        return image
    
    def generate_train_val_dataset(self, train_image_num = 30000, valid_image_num = 300, test_image_num = 300,
                        circle_color_min = 0, circle_color_max = 5/7, bg_color_min = 6/7, bg_color_max = 1,
                        min_circle_size = 160, max_circle_size = 160,width = 10,
                        downsampling = True, downsampling_factor = 4, seed = 42):
        assert circle_color_max < bg_color_min, "cirle_color_max should be smaller than bg_color, otherwise there could be a chance that the fore/back-ground have the same color."
        # register some fields for test dataset generation
        self.circle_color_min, self.circle_color_max = circle_color_min, circle_color_max
        self.bg_color_min, self.bg_color_max = bg_color_min, bg_color_max
        self.width = width
        self.min_circle_size, self.max_circle_size = min_circle_size, max_circle_size
        self.train_val_seed = seed
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor

        rng = np.random.default_rng(seed = seed)
        
        print(f"======generating {self.savedir_name} training & validation dataset======")
        image_ls = []
        test_mask_ls = []

        for i in tqdm(range(train_image_num+valid_image_num+test_image_num)):
            circle_color = rng.integers(np.ceil(circle_color_min*255), np.floor(circle_color_max*255)+1, self.channel_num)
            bg_color = rng.integers(np.ceil(bg_color_min*255), np.floor(bg_color_max*255)+1, self.channel_num)
            circle_size = rng.integers(self.min_circle_size, self.max_circle_size+1)
            coordinate_min = 0
            coordinate_max = self.image_size-circle_size
            circle_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]
            image = self.generate_image(circle_color, circle_position, bg_color, circle_size, width, 
                                        image_size = self.image_size, channel_num = self.channel_num)
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::self.downsampling_factor,::self.downsampling_factor]
            image_ls.append(image)
            if i >= train_image_num + valid_image_num:
                mask = self.generate_mask(circle_position = circle_position, circle_size = circle_size,
                                          width = width, image_size = self.image_size)
                mask = np.expand_dims(mask, axis = 0)
                if self.downsampling: mask = mask[:,::self.downsampling_factor,::self.downsampling_factor]
                test_mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        test_masks = np.concatenate(test_mask_ls, axis = 0)
        # Please refer to [Color trick] and [Testing samples] in ClosureDataset
        # to understand the following
        train_images = images[:train_image_num]*999/1000 
        valid_images = images[train_image_num:train_image_num+valid_image_num]*998/1000
        test_images = images[train_image_num+valid_image_num:]

        print("saving ...")
        save_train_val_dataset(savedir_name=self.savedir_name, train_images=train_images, 
                               valid_images=valid_images, test_images=test_images, test_masks = test_masks,
                               visualize=self.visualize)

        self.train_val_dataset_generated = True
    
    def generate_test_dataset(self, test_image_num = 500, random_frag_num_min = 5, random_frag_num_max = 20, arc_angle = 30,
                              seed = 41):
        assert self.train_val_dataset_generated, "Need to generate training and validation dataset first!"
        if seed == self.train_val_seed: warnings.warn("It is highly recommended that you use different seeds to generate the training/validation dataset and the testing dataset.")

        rng = np.random.default_rng(seed = seed)

        print(f"======generating {self.savedir_name} testing dataset======")
        image_ls = []
        mask_ls = []
        detailed_mask_ls = []
        
        for i in tqdm(range(test_image_num)):
            circle_color = rng.integers(np.ceil(self.circle_color_min*255), np.floor(self.circle_color_max*255)+1, self.channel_num)
            bg_color = rng.integers(np.ceil(self.bg_color_min*255), np.floor(self.bg_color_max*255)+1, self.channel_num)
            random_arc_color_1 = rng.integers(0,256, self.channel_num) # Note the choice of the random arc colors can exceed the range of circle colors, which might be risky
            random_arc_color_2 = rng.integers(0,256, self.channel_num) # Note the choice of the random arc colors can exceed the range of circle colors, which might be risky

            circle_size = rng.integers(self.min_circle_size, self.max_circle_size+1)
            coordinate_min = 0
            coordinate_max = self.image_size-circle_size
            circle_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]

            image = self.generate_test_image(circle_color, random_arc_color_1, random_arc_color_2,
                                             circle_position, bg_color, circle_size, self.width, 
                                             random_frag_num_min, random_frag_num_max, arc_angle = arc_angle, 
                                            image_size = self.image_size, channel_num=self.channel_num, rng=rng )
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::self.downsampling_factor,::self.downsampling_factor]
            image_ls.append(image)

            mask = self.generate_mask(circle_position, circle_size, width=self.width, image_size = self.image_size)
            mask = np.expand_dims(mask, axis = 0)
            if self.downsampling: mask = mask[:,::self.downsampling_factor,::self.downsampling_factor]
            mask_ls.append(mask)

            detailed_mask = self.generate_detailed_mask(circle_position, circle_size, width = self.width, image_size = self.image_size, arc_angle = arc_angle )
            detailed_mask = np.expand_dims(detailed_mask, axis = 0)
            if self.downsampling: detailed_mask = detailed_mask[:,::self.downsampling_factor,::self.downsampling_factor]
            detailed_mask_ls.append(detailed_mask)

        images = np.concatenate(image_ls, axis = 0)
        masks = np.concatenate(mask_ls, axis = 0)
        detailed_masks = np.concatenate(detailed_mask_ls, axis = 0)

        print("saving ...")
        save_test_dataset(savedir_name = self.savedir_name, test_image_num=test_image_num,
                          images = images, masks = masks, detailed_masks=detailed_masks, visualize = self.visualize)

class IllusoryOcclusionDataset(BaseDataset):
    def __init__(self, channel_num=3, image_size = 256,
                  dataset_name= 'dIlluOcclusion',suffix = "", visualize = True):
        super(IllusoryOcclusionDataset, self).__init__(image_size=image_size,channel_num=channel_num,#type:ignore
                                                   savedir_name=dataset_name+suffix,
                                                   visualize=visualize,)

    def generate_mask(self, square_size, square_position, image_size = 64, ):
        mask = np.zeros((image_size, image_size))
        square = Rectangle((square_size,square_size),(1,),position=square_position,)
        mask=square.draw(mask).squeeze()
        return mask

    def generate_train_val_dataset(self, train_image_num= 30000, valid_image_num = 300, test_image_num = 300,
                        bg_color_min=0, bg_color_max = 5/7, square_color_min = 2/7, square_color_max = 1,
                        bg_stripe_color_ratio = [0.2,0.6,0.3], square_stripe_color_ratio = [0.8,0.7,0.3],
                        bg_stripe_width = 10, bg_stripe_interval = 40,
                        min_square_size = 80, max_square_size = 80,
                        bg_stripe_offset = 10,
                        downsampling = True, downsampling_factor = 4,
                        seed = 42):

        # register some fields for test dataset generation
        assert(square_color_min<bg_color_max), "square_color_min should be smaller than bg_color_max"
        assert(bg_color_min==0), "bg_color_min need to be 0, otherwise unreasonable images might be generated for testing."
        # The above assertion is just for technical simplicity. To understand them, please refer to [bg_color_min] in generate_test_dataset().
        self.square_color_min, self.square_color_max, self.bg_color_max= square_color_min, square_color_max, bg_color_max
        self.bg_stripe_color_ratio, self.square_stripe_color_ratio = bg_stripe_color_ratio, square_stripe_color_ratio
        self.bg_stripe_width, self.bg_stripe_interval = bg_stripe_width, bg_stripe_interval
        self.min_square_size, self.max_square_size = min_square_size, max_square_size
        self.bg_stripe_offset = bg_stripe_offset
        self.train_val_seed = seed
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor

        rng = np.random.default_rng(seed = seed)
        
        print(f"======generating {self.savedir_name} training & validation dataset======")
        image_ls = []
        test_mask_ls = []
        test_detailed_mask_ls = []
        for i in tqdm(range(train_image_num+valid_image_num+test_image_num)):
            
            bg_color = rng.integers(np.ceil(bg_color_min*255), np.floor(bg_color_max*255)+1, self.channel_num)/255
            bg_stripe_color = bg_color*np.asarray(bg_stripe_color_ratio)
            square_color = rng.integers(np.ceil(square_color_min*255), np.floor(square_color_max*255)+1, self.channel_num)/255
            square_stripe_color = square_color*np.asarray(square_stripe_color_ratio)

            square_size = rng.integers(min_square_size, max_square_size+1)
            coordinate_min = 0
            coordinate_max = self.image_size-square_size
            square_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]

            # Please refer to the [Generate image] part in ClosureDataset to see how images are generated
            detailed_mask = self.generate_detailed_mask(square_size=square_size,
                                                        square_position = square_position,
                                                        bg_stripe_width=bg_stripe_width,
                                                        bg_stripe_interval = bg_stripe_interval,
                                                        bg_stripe_offset=self.bg_stripe_offset,
                                                        image_size = self.image_size)
            image = np.zeros((self.image_size, self.image_size, self.channel_num))
            image[detailed_mask == 0] = bg_color
            image[detailed_mask == 1] = bg_stripe_color
            image[detailed_mask == 2] = square_color
            image[detailed_mask == 3] = square_stripe_color
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::downsampling_factor,::downsampling_factor]
            image_ls.append(image)
            if i >= train_image_num + valid_image_num:
                detailed_mask = np.expand_dims(detailed_mask, axis = 0)
                if self.downsampling: detailed_mask = detailed_mask[:,::self.downsampling_factor,::self.downsampling_factor]
                test_detailed_mask_ls.append(detailed_mask)

                mask = self.generate_mask(square_size = square_size, 
                                          square_position=square_position,
                                          image_size = self.image_size)
                mask = np.expand_dims(mask, axis = 0)
                if self.downsampling: mask = mask[:,::downsampling_factor,::downsampling_factor]
                test_mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        test_masks = np.concatenate(test_mask_ls, axis = 0)
        test_detailed_masks = np.concatenate(test_detailed_mask_ls, axis = 0)
        # Please refer to [Color trick] and [Testing samples] in ClosureDataset
        # to understand the following
        train_images = images[:train_image_num]*999/1000  
        valid_images = images[train_image_num:train_image_num+valid_image_num]*998/1000
        test_images = images[train_image_num+valid_image_num:]

        print("saving ...")
        save_train_val_dataset(savedir_name=self.savedir_name, train_images=train_images, 
                               valid_images=valid_images, test_images=test_images, test_masks = test_masks,
                               test_detailed_masks=test_detailed_masks, visualize=self.visualize)

        self.train_val_dataset_generated = True

    def generate_test_dataset(self, test_image_num = 500, seed = 41):

        assert self.train_val_dataset_generated, "Need to generate training and validation dataset first!"
        if seed == self.train_val_seed: warnings.warn("It is highly recommended that you use different seeds to generate the training/validation dataset and the testing dataset.")

        rng = np.random.default_rng(seed = seed)
        print(f"======generating {self.savedir_name} testing dataset======")
        image_ls = []
        mask_ls = []
        detailed_mask_ls = [] # because both foreground swiss flag and the background have two parts, so we can have detailed masks
        for i in tqdm(range(test_image_num)):
            # generate the position
            square_size = rng.integers(self.min_square_size, self.max_square_size+1)
            coordinate_min = 0
            coordinate_max = self.image_size-square_size
            square_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]
            # generate the color
            if rng.uniform() < 0.5:
                # generate an image in which the square has the same color as the background
                square_bg_color = rng.integers(np.ceil(self.square_color_min*255), np.floor(self.bg_color_max*255)+1, self.channel_num)/255
                # generate the image
                detailed_mask = self.generate_detailed_mask(square_size=square_size,
                                                            square_position = square_position,
                                                            bg_stripe_width=self.bg_stripe_width,
                                                            bg_stripe_interval = self.bg_stripe_interval,
                                                            bg_stripe_offset=self.bg_stripe_offset,
                                                            image_size = self.image_size)
                image = np.zeros((self.image_size, self.image_size, self.channel_num))
                image[detailed_mask == 0] = square_bg_color # bg
                image[detailed_mask == 1] = square_bg_color * np.asarray(self.bg_stripe_color_ratio) # bg_stripe
                image[detailed_mask == 2] = square_bg_color # square
                image[detailed_mask == 3] = square_bg_color * np.asarray(self.square_stripe_color_ratio) # square_stripe
            else:
                # generate an image in which the square stripe has the same color as the background
                square_stripe_color_min = np.asarray(self.square_color_min) * np.array(self.square_stripe_color_ratio)
                square_stripe_color_max = self.square_color_max * np.array(self.square_stripe_color_ratio)
                # calculate which color (probably multi-channeled) we should assign to square_stripe and bg
                square_stripe_bg_color_ls = []
                for c in range(self.channel_num):
                    color_lower_bound = np.ceil(square_stripe_color_min[c]*255)
                    color_upper_bound = np.floor(np.min((self.bg_color_max, square_stripe_color_max[c]))*255)+1
                    square_stripe_bg_color_c = rng.integers(color_lower_bound, color_upper_bound ,1).reshape(-1)/255
                    square_stripe_bg_color_ls.append(square_stripe_bg_color_c)
                square_stripe_bg_color = np.concatenate(square_stripe_bg_color_ls).reshape(-1) # notice the bg_color is still guaranteed to be n/255, thus avoiding taking the same values as in training and validation dataset.
                 # [bg_color_min]: Note, in the above, we would require bg_color_min to be 0 to make sure that square_stripe_color_min and bg_color_max overlap.
                # calculate the square_color, (just for convenience.)
                square_color = (square_stripe_bg_color/(np.asarray(self.square_stripe_color_ratio).reshape(-1)))
                bg_stripe_color = square_stripe_bg_color * np.asarray(self.bg_stripe_color_ratio)
                # generate image
                detailed_mask = self.generate_detailed_mask(square_size=square_size,
                                                            square_position = square_position,
                                                            bg_stripe_width=self.bg_stripe_width,
                                                            bg_stripe_interval = self.bg_stripe_interval,
                                                            bg_stripe_offset=self.bg_stripe_offset,
                                                            image_size = self.image_size)
                image = np.zeros((self.image_size, self.image_size, self.channel_num))
                image[detailed_mask == 0] = square_stripe_bg_color # bg
                image[detailed_mask == 1] = bg_stripe_color # bg_stripe
                image[detailed_mask == 2] = square_color # square
                image[detailed_mask == 3] = square_stripe_bg_color # square_stripe

            # generate the image based on the position and the color
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::self.downsampling_factor,::self.downsampling_factor]
            image_ls.append(image)
            # generate the masks
            mask = self.generate_mask(square_size,square_position, self.image_size)
            mask = np.expand_dims(mask, axis = 0)
            if self.downsampling: mask = mask[:,::self.downsampling_factor,::self.downsampling_factor]
            mask_ls.append(mask)
            # save the detailed masks
            detailed_mask = np.expand_dims(detailed_mask, axis = 0)
            if self.downsampling: detailed_mask = detailed_mask[:,::self.downsampling_factor,::self.downsampling_factor]
            detailed_mask_ls.append(detailed_mask)
        
        images = np.concatenate(image_ls, axis = 0)
        masks = np.concatenate(mask_ls, axis = 0)
        detailed_masks = np.concatenate(detailed_mask_ls, axis = 0)

        print("saving ...")
        save_test_dataset(savedir_name = self.savedir_name, test_image_num=test_image_num,
                          images = images, masks = masks, detailed_masks = detailed_masks, 
                          visualize = self.visualize)

    def generate_detailed_mask(self, square_size, square_position, bg_stripe_width, 
                               bg_stripe_interval, bg_stripe_offset = 10, 
                               image_size = 64, ):
        """Generate detailed masks in which a background and its stripes, a square and its crossings, 
        are marked differently. 0 = bg, 1 = bg_stripe, square = 2, square_stripe = 3"""
        mask = np.zeros((image_size, image_size))
        l = 2
        while l < image_size*2:
            xy = [(-bg_stripe_offset,l),(l,-bg_stripe_offset)] # -10 for aesthetic concern
            bg_stripe = Line(xy,(1,),bg_stripe_width)
            mask= bg_stripe.draw(mask)
            l+=bg_stripe_interval
        # draw square
        square = Rectangle(size=(square_size,square_size), color=(2,),position=square_position)
        mask= square.draw(mask)
        # draw the crossing on the square
        ## first calculate the position
        offset = np.floor(square_size/10*3)
        stripe_width = square_size - offset*2
        stripe1_size, stripe2_size = (stripe_width,square_size), (square_size, stripe_width)
        stripe1_pos, stripe2_pos = (square_position[0]+offset, square_position[1]), (square_position[0], square_position[1]+offset)
        ## then draw the stripes
        stripe1 = Rectangle(size=stripe1_size,color=(3,),position=stripe1_pos)
        stripe2 = Rectangle(size=stripe2_size,color=(3,),position=stripe2_pos)
        mask= stripe1.draw(mask)
        mask= stripe2.draw(mask)
        return mask.squeeze()        

class KanizsaDataset(BaseDataset):
    def __init__(self, channel_num=3, image_size = 256, 
                  dataset_name = 'dKanizsa', suffix = "", visualize = True):
        super(KanizsaDataset, self).__init__(image_size=image_size,channel_num=channel_num,#type:ignore
                                                   savedir_name=dataset_name + suffix,
                                                   visualize=visualize,)
    
    def generate_image(self, bg_color, square_color:Tuple[int,...], circle_color:Tuple[int,...],
                        square_size, square_position,
                        circle_pos_random_range, 
                        train_val_test1 = False, test = False, 
                        generate_test1_mask = False,
                        image_size=64, channel_num = 3, rng=None):
        
        assert (train_val_test1 and not test) or (not train_val_test1 and test)
        image = np.ones((image_size,image_size,channel_num))*bg_color.reshape(1,1,self.channel_num)

        # If generating training/validation/test1 pictures, then:
        ##  generate the four circles with randomized positions (still constrained to a certain scope, though), then generate a square
        # If generating testing pictures, then:
        ##  generate four circles with no randomness in their positions, then generate a square

        # For technical convenience, we also draw the mask for the test_1.npz dataset in this generate_image function.

        # draw the circles
        circle_pos_ls = [
                        (square_position[0] - self.circle_size/2, square_position[1] - self.circle_size/2),
                        (square_position[0] - self.circle_size/2 + square_size, square_position[1] - self.circle_size/2),  
                        (square_position[0] - self.circle_size/2, square_position[1] - self.circle_size/2 + square_size),  
                        (square_position[0] - self.circle_size/2 + square_size, square_position[1] - self.circle_size/2 + square_size),  
                        ]

        test1_mask = None
        if train_val_test1:
            # draw four circles
            if generate_test1_mask: test1_mask = np.zeros((image_size, image_size))
            for circle_pos in (circle_pos_ls):
                circle_pos_ls_ = list(circle_pos)
                circle_pos_ls_[0] += rng.integers(-circle_pos_random_range, circle_pos_random_range+1)
                circle_pos_ls_[1] += rng.integers(-circle_pos_random_range, circle_pos_random_range+1)
                circle_pos = tuple(circle_pos_ls_)
                circle = Ellipse((self.circle_size,self.circle_size,),circle_color,circle_pos)
                image = circle.draw(image)
                if generate_test1_mask:
                    circle_mask = Ellipse((self.circle_size,self.circle_size,),(1,),circle_pos)
                    test1_mask= circle_mask.draw(test1_mask)

            # draw the square
            square = Rectangle((square_size, square_size,), square_color, square_position)
            image = square.draw(image)
            if generate_test1_mask:
                square_mask = Rectangle((square_size,square_size),(2,),position=square_position,)
                test1_mask=square_mask.draw(test1_mask)

        
        if test:
            for circle_pos in (circle_pos_ls):
                circle = Ellipse((self.circle_size,self.circle_size,),circle_color,circle_pos)
                image = circle.draw(image)
            square = Rectangle((square_size, square_size,), square_color, square_position)
            # normally, here, the square has the same color as the background
            image = square.draw(image)
            
        image = image/255

        if generate_test1_mask:
            return image, test1_mask
        else:
            return image

    def generate_mask(self, square_size, square_position, image_size,):
        """In this function, the masks for the testing samples in test.npz are generated, which are the 
        special cases we created for testing."""
        mask = np.zeros((image_size, image_size))

        circle_pos_ls = [
                        (square_position[0] - self.circle_size/2, square_position[1] - self.circle_size/2),
                        (square_position[0] - self.circle_size/2 + square_size, square_position[1] - self.circle_size/2),  
                        (square_position[0] - self.circle_size/2, square_position[1] - self.circle_size/2 + square_size),  
                        (square_position[0] - self.circle_size/2 + square_size, square_position[1] - self.circle_size/2 + square_size),  
                        ]

        for circle_pos in circle_pos_ls:
            circle = Ellipse((self.circle_size,self.circle_size,),(1,),circle_pos)
            mask = circle.draw(mask)
        square = Rectangle((square_size, square_size,), (2,), square_position)
        # normally, here, the square has the same color as the background
        mask= square.draw(mask)

        return mask.squeeze()
    
    def generate_train_val_dataset(self, train_image_num = 30000, valid_image_num = 300, test_image_num = 300,
                    bg_color_min= 5/7, bg_color_max = 6/7, square_color_min = 5/7, square_color_max = 1,
                    circle_color_min = 2/7, circle_color_max = 3/7,
                    min_square_size = 100, max_square_size = 100, circle_size = 48, 
                    circle_pos_randomness = True, circle_pos_random_range = 20,
                    downsampling = True, downsampling_factor = 4,
                    seed = 42
                    ):

        self.min_square_size, self.max_square_size = min_square_size, max_square_size
        self.circle_size = circle_size
        self.test_color_min, self.test_color_max = square_color_min, bg_color_max
        self.circle_color_min, self.circle_color_max = circle_color_min, circle_color_max
        self.train_val_seed = seed
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor

        rng = np.random.default_rng(seed = seed)

        assert bg_color_min <= square_color_min <= bg_color_max <= square_color_max,"We require: bg_color_min <= square_color_min <= bg_color_max <= square_color_max"

        print(f"======generating {self.savedir_name} training & validation dataset======")
        image_ls = []
        test_mask_ls = []
        for i in tqdm(range(train_image_num+valid_image_num+test_image_num)):
            bg_color = rng.integers(np.ceil(bg_color_min*255), np.floor(bg_color_max*255)+1, self.channel_num)
            square_color = rng.integers(np.ceil(square_color_min*255), np.floor(square_color_max*255)+1, self.channel_num)
            circle_color = rng.integers(np.ceil(circle_color_min*255), np.floor(circle_color_max*255)+1, self.channel_num)
            square_size = rng.integers(self.min_square_size, self.max_square_size+1)
            coordinate_min = self.circle_size/2 + circle_pos_random_range if circle_pos_randomness else self.circle_size/2
            coordinate_max = self.image_size - square_size - self.circle_size/2- circle_pos_random_range if circle_pos_random_range else self.image_size - square_size - self.circle_size/2
            square_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]
            if i < train_image_num+valid_image_num:
                image = self.generate_image(bg_color=bg_color, square_color = tuple(square_color), 
                                            circle_color = tuple(circle_color), square_size = square_size,
                                            square_position=square_position,
                                            circle_pos_random_range=circle_pos_random_range, train_val_test1 = True, 
                                            test = False, 
                                            generate_test1_mask=False, image_size = self.image_size, 
                                            channel_num = self.channel_num, rng = rng)
            else:
                image,mask = self.generate_image(bg_color=bg_color, square_color = tuple(square_color), 
                                            circle_color = tuple(circle_color), square_size = square_size,
                                            square_position=square_position,
                                            circle_pos_random_range=circle_pos_random_range, train_val_test1 = True, 
                                            test = False, 
                                            generate_test1_mask=True, image_size = self.image_size, 
                                            channel_num = self.channel_num, rng = rng)
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::downsampling_factor,::downsampling_factor]
            image_ls.append(image)
            if i >= train_image_num + valid_image_num: # only then the mask will be generated
                mask = np.expand_dims(mask, axis = 0)
                if self.downsampling: mask = mask[:,::downsampling_factor,::downsampling_factor]
                test_mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        test_masks = np.concatenate(test_mask_ls, axis = 0)
        # Please refer to [Color trick] and [Testing samples] in ClosureDataset
        # to understand the following
        train_images = images[:train_image_num]*999/1000                                                         
        valid_images = images[train_image_num:train_image_num+valid_image_num]*998/1000
        test_images = images[train_image_num+valid_image_num:]

        print("saving ...")
        save_train_val_dataset(savedir_name=self.savedir_name, train_images=train_images, 
                               valid_images=valid_images, test_images=test_images, test_masks = test_masks,
                               visualize=self.visualize)

        self.train_val_dataset_generated = True
    
    def generate_test_dataset(self, test_image_num = 500, seed =41, ):
        assert self.train_val_dataset_generated, "Need to generate training and validation dataset first!"
        if seed == self.train_val_seed: warnings.warn("It is highly recommended that you use different seeds to generate the training/validation dataset and the testing dataset.")

        rng = np.random.default_rng(seed = seed)
        
        print(f"======generating {self.savedir_name} testing dataset======")
        image_ls = []
        mask_ls = []

        for i in tqdm(range(test_image_num)):
            # in this case, square and background has the same color
            square_bg_color = rng.integers(np.ceil(self.test_color_min*255), np.floor(self.test_color_max*255)+1, self.channel_num)
            circle_color = rng.integers(np.ceil(self.circle_color_min*255), np.floor(self.circle_color_max*255)+1, self.channel_num)

            square_size = rng.integers(self.min_square_size, self.max_square_size+1)
            coordinate_min = self.circle_size/2  
            coordinate_max = self.image_size - square_size - self.circle_size/2
            square_position = [rng.integers(coordinate_min, coordinate_max+1) for k in range(2)]
            
            image = self.generate_image(bg_color = square_bg_color, square_color = tuple(square_bg_color), 
                                        circle_color = tuple(circle_color), square_size=square_size,
                                        square_position = square_position, 
                                        circle_pos_random_range=None, train_val_test1 = False, 
                                        test = True, 
                                        generate_test1_mask=False, image_size = self.image_size, 
                                        channel_num = self.channel_num, rng = rng,)
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::self.downsampling_factor,::self.downsampling_factor]
            image_ls.append(image)

            mask =  self.generate_mask(square_size = square_size, square_position = square_position, 
                                       image_size = self.image_size, )
            mask = np.expand_dims(mask, axis = 0)
            if self.downsampling: mask = mask[:,::self.downsampling_factor,::self.downsampling_factor]
            mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        masks = np.concatenate(mask_ls, axis = 0)

        print("saving ...")
        save_test_dataset(savedir_name = self.savedir_name, test_image_num=test_image_num,
                          images = images, masks = masks,  
                          visualize = self.visualize)

class GradientOcclusionDataset(BaseDataset):
    def __init__(self, channel_num=3, image_size = 256,
                  dataset_name= 'dGradOcclusion', suffix = "", visualize = True):
        super(GradientOcclusionDataset, self).__init__(image_size=image_size,channel_num=channel_num,#type:ignore
                                                   savedir_name=dataset_name + suffix,
                                                   visualize=visualize,)
        
    def generate_image(self, bg_color: Tuple[int,...], shape_color_1: Tuple[int,...], shape_color_2: Tuple[int,...],  
                       shape_size_1:Tuple[int,...], shape_size_2:Tuple[int,...], shape_position_1:Tuple[int,...], shape_position_2:Tuple[int,...],
                       end_color_ratio_1: float = 0.4, end_color_ratio_2: float = 0.3,
                       order = 0, image_size=64, channel_num = 3):
        """end_color_ratio_1 and end_color_ratio_2 determine how strong the gradients are in the two shapes. The 
        color gradient effect is implemented by linearly interpolating shape_color_1 and 
        shape_color_1*end_color_ratio_1."""        
        image = np.ones((image_size,image_size,channel_num))*np.array(bg_color).reshape(1,1,self.channel_num)
        if order == 0:# This branch is not used in the end, you may ignore it.
            # vertical gradient rectangle first
            shape_1 = VerticalGradientRectangle(shape_size_1, shape_color_1, shape_position_1, end_color_ratio=end_color_ratio_1)
            image = shape_1.draw(image)
            shape_2 = HorizontalGradientRectangle(shape_size_2, shape_color_2, shape_position_2, end_color_ratio=end_color_ratio_2)
            image = shape_2.draw(image)
        elif order == 1:
            shape_2 = HorizontalGradientRectangle(shape_size_2, shape_color_2, shape_position_2, end_color_ratio=end_color_ratio_2)
            image = shape_2.draw(image)
            shape_1 = VerticalGradientRectangle(shape_size_1, shape_color_1, shape_position_1, end_color_ratio=end_color_ratio_1)
            image = shape_1.draw(image)
        else:
            raise Exception("Order can only be 0 or 1.")

        image = image / 255
        
        return image
    
    def generate_mask(self,shape_size_1:Tuple[int,...], shape_size_2:Tuple[int,...], shape_position_1:Tuple[int,...], shape_position_2:Tuple[int,...], order = 0,
                      image_size = 64, ): 
        mask = np.zeros((image_size, image_size))
        if order == 0:
            # vertical gradient rectangle first
            shape_1 = Rectangle(shape_size_1, (1,), shape_position_1, )
            mask = shape_1.draw(mask)
            shape_2 = Rectangle(shape_size_2, (2,), shape_position_2, )
            mask = shape_2.draw(mask)
        elif order == 1:
            shape_2 = Rectangle(shape_size_2, (2,), shape_position_2, )
            mask = shape_2.draw(mask)
            shape_1 = Rectangle(shape_size_1, (1,), shape_position_1, )
            mask = shape_1.draw(mask)
        else:
            raise Exception("only take order to be 0 or 1.")
        
        return mask.squeeze()

    def generate_detailed_mask(self,shape_size_1:Tuple[int,...], shape_size_2:Tuple[int,...], shape_position_1:Tuple[int,...], shape_position_2:Tuple[int,...], order = 0,
                      image_size = 64, ): 
        """The overlapping area is marked with a different label number"""
        mask = np.zeros((image_size, image_size))
        aux_mask = np.zeros((image_size, image_size)) # an auxiliary mask to simplify implementation
        if order == 0:
            # vertical gradient rectangle first
            shape_1 = Rectangle(shape_size_1, (1,), shape_position_1, )
            mask = shape_1.draw(mask)
            shape_2 = Rectangle(shape_size_2, (2,), shape_position_2, )
            aux_mask = shape_2.draw(aux_mask)
            mask = aux_mask + mask# by adding the labels of the two masks together, we can get a different number at the overlapping area
        elif order == 1:
            shape_2 = Rectangle(shape_size_2, (2,), shape_position_2, )
            mask = shape_2.draw(mask)
            shape_1 = Rectangle(shape_size_1, (1,), shape_position_1, )
            aux_mask = shape_1.draw(aux_mask)
            mask = aux_mask + mask
        else:
            raise Exception("only take order to be 0 or 1.")
        
        return mask.squeeze()

    def generate_train_val_dataset(self, train_image_num = 30000, valid_image_num = 300, test_image_num = 300,
                        shape1_color_min = 3/7, shape1_color_max = 5/7, shape2_color_min = 3/7, shape2_color_max = 5/7,
                        bg_color_min = 6/7, bg_color_max = 1,
                        shape1_size = (120,60), shape2_size = (60,120), end_color_ratio_1 = 1, end_color_ratio_2 = 0.4,
                        downsampling = True, downsampling_factor = 4,
                        seed = 42):
        # register some fields for test dataset generation
        self.shape1_color_min, self.shape1_color_max = shape1_color_min, shape1_color_max
        self.shape2_color_min, self.shape2_color_max = shape2_color_min, shape2_color_max
        self.bg_color_min, self.bg_color_max = bg_color_min, bg_color_max
        self.shape1_size, self.shape2_size = shape1_size, shape2_size
        self.end_color_ratio_1, self.end_color_ratio_2 = end_color_ratio_1, end_color_ratio_2
        self.train_val_seed = seed 
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor
        
        rng = np.random.default_rng(seed = seed)

        print(f"======generating {self.savedir_name} training & validation dataset======")
        image_ls = []
        test_mask_ls = []
        test_detailed_mask_ls = []
        for i in tqdm(range(train_image_num+valid_image_num+test_image_num)):

            shape1_color = rng.integers(np.ceil(shape1_color_min*255), np.floor(shape1_color_max*255)+1, self.channel_num)
            shape2_color = rng.integers(np.ceil(shape2_color_min*255), np.floor(shape2_color_max*255)+1, self.channel_num)
            bg_color = rng.integers(np.ceil(bg_color_min*255), np.floor(bg_color_max*255)+1, self.channel_num)

            shape1_coordinate_min = [0,0]
            shape1_coordinate_max = [self.image_size-l for l in shape1_size]
            shape1_position = [rng.integers(shape1_coordinate_min[k], shape1_coordinate_max[k]+1) for k in range(2)]
            shape2_coordinate_min = [0,0]
            shape2_coordinate_max = [self.image_size-l for l in shape2_size]
            shape2_position = [rng.integers(shape2_coordinate_min[k], shape2_coordinate_max[k]+1) for k in range(2)]

            # order = rng.integers(0,2) 
            order = 1 # only generate "vertical-over-horizontal" pattern
            
            image = self.generate_image(bg_color=bg_color,shape_color_1 = tuple(shape1_color), shape_color_2 = tuple(shape2_color),
                                        shape_size_1=shape1_size, shape_size_2=shape2_size, 
                                        shape_position_1=shape1_position, shape_position_2=shape2_position,
                                        end_color_ratio_1 = end_color_ratio_1, end_color_ratio_2=end_color_ratio_2, 
                                        order = order, image_size = self.image_size, channel_num =self.channel_num )
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::downsampling_factor,::downsampling_factor]
            image_ls.append(image)
            if i >= train_image_num + valid_image_num:

                mask = self.generate_mask(shape_size_1=shape1_size, shape_size_2= shape2_size, shape_position_1 = shape1_position,
                                          shape_position_2=shape2_position, order = order, image_size = self.image_size)
                mask = np.expand_dims(mask, axis = 0)
                if self.downsampling: mask = mask[:,::downsampling_factor,::downsampling_factor]
                test_mask_ls.append(mask)

                detailed_mask = self.generate_detailed_mask(shape_size_1=shape1_size, shape_size_2= shape2_size, shape_position_1 = shape1_position,
                                          shape_position_2=shape2_position, order = order, image_size = self.image_size)
                detailed_mask = np.expand_dims(detailed_mask, axis = 0)
                if self.downsampling: detailed_mask = detailed_mask[:,::self.downsampling_factor,::self.downsampling_factor]
                test_detailed_mask_ls.append(detailed_mask)

        images = np.concatenate(image_ls, axis = 0)
        test_masks = np.concatenate(test_mask_ls, axis = 0)
        test_detailed_masks = np.concatenate(test_detailed_mask_ls, axis = 0)
        # Please refer to [Color trick] and [Testing samples] in ClosureDataset
        # to understand the following
        train_images = images[:train_image_num]*999/1000                                                         # dataset overlaps with the validation and testing datasets
        valid_images = images[train_image_num:train_image_num+valid_image_num]*998/1000
        test_images = images[train_image_num+valid_image_num:]

        print("saving ...")
        save_train_val_dataset(savedir_name=self.savedir_name, train_images=train_images, 
                               valid_images=valid_images, test_images=test_images, test_masks = test_masks,
                               test_detailed_masks=test_detailed_masks, visualize=self.visualize)

        self.train_val_dataset_generated = True
        
    def generate_test_dataset(self, test_image_num = 500, seed = 41):
        
        # The test dataset generating process is the same as the training dataset, except that we impose the
        # two shapes to overlap.
        assert self.train_val_dataset_generated, "Need to generate training and validation dataset first!"
        if seed == self.train_val_seed: warnings.warn("It is highly recommended that you use different seeds to generate the training/validation dataset and the testing dataset.")

        rng = np.random.default_rng(seed = seed)

        print(f"======generating {self.savedir_name} testing dataset======")
        image_ls = []
        mask_ls = []
        detailed_mask_ls = []

        for i in tqdm(range(test_image_num)):
            # choose color
            shape1_color = rng.integers(np.ceil(self.shape1_color_min*255), np.floor(self.shape1_color_max*255)+1, self.channel_num)
            shape2_color = rng.integers(np.ceil(self.shape2_color_min*255), np.floor(self.shape2_color_max*255)+1, self.channel_num)
            bg_color = rng.integers(np.ceil(self.bg_color_min*255), np.floor(self.bg_color_max*255)+1, self.channel_num)
            # choose position for shape1
            shape1_coordinate_min = [0,0]
            shape1_coordinate_max = [self.image_size-l for l in self.shape1_size]
            shape1_position = [rng.integers(shape1_coordinate_min[k], shape1_coordinate_max[k]+1) for k in range(2)]
            # choose position for shape2 and make sure it overlaps with shape1
            shape2_coordinate_min = [np.max((0,shape1_position[m]-self.shape2_size[m]+1)) for m in range(2)]
            shape2_coordinate_max_orig = [self.image_size-l for l in self.shape2_size]
            shape2_coordinate_max = [np.min((shape2_coordinate_max_orig[m],shape1_position[m]+self.shape1_size[m]-1)) for m in range(2)]
            shape2_position = [rng.integers(shape2_coordinate_min[k], shape2_coordinate_max[k]+1) for k in range(2)]

            # order = rng.integers(0,2)
            order = 1
 
            image = self.generate_image(bg_color=tuple(bg_color), shape_color_1 = tuple(shape1_color), shape_color_2 = tuple(shape2_color),
                                        shape_size_1=self.shape1_size, shape_size_2=self.shape2_size, 
                                        shape_position_1=shape1_position, shape_position_2=shape2_position,
                                        end_color_ratio_1 = self.end_color_ratio_1, end_color_ratio_2=self.end_color_ratio_2, 
                                        order = order, image_size = self.image_size, channel_num =self.channel_num )
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if self.downsampling: image = image[:,:,::self.downsampling_factor,::self.downsampling_factor]
            image_ls.append(image)

            mask = self.generate_mask(shape_size_1 = self.shape1_size, shape_size_2=self.shape2_size,
                                      shape_position_1=shape1_position, shape_position_2=shape2_position,
                                      order = order, image_size = self.image_size)
            mask = np.expand_dims(mask, axis = 0)
            if self.downsampling: mask = mask[:,::self.downsampling_factor,::self.downsampling_factor]
            mask_ls.append(mask)

            detailed_mask = self.generate_detailed_mask(shape_size_1 = self.shape1_size, shape_size_2=self.shape2_size,
                                      shape_position_1=shape1_position, shape_position_2=shape2_position,
                                      order = order, image_size = self.image_size)
            detailed_mask = np.expand_dims(detailed_mask, axis = 0)
            if self.downsampling: detailed_mask = detailed_mask[:,::self.downsampling_factor,::self.downsampling_factor]
            detailed_mask_ls.append(detailed_mask)

        images = np.concatenate(image_ls, axis = 0)
        masks = np.concatenate(mask_ls, axis = 0)
        detailed_masks = np.concatenate(detailed_mask_ls, axis = 0)

        print("saving ...")
        save_test_dataset(savedir_name = self.savedir_name, test_image_num=test_image_num,
                          images = images, masks = masks, detailed_masks=detailed_masks, visualize = self.visualize)

class ProximityGroupingDataset(BaseDataset):
    def __init__(self, channel_num=3, image_size = 64,
                  dataset_name = 'dProximity', suffix = "", visualize = True):
        # image_size is designed to be 64, since the visual scene only contains squares, the resolution 
        # does not matter at all. Setting image_size to 64. 
        super(ProximityGroupingDataset, self).__init__(image_size=image_size,channel_num=channel_num,#type:ignore
                                                   savedir_name=dataset_name + suffix,
                                                   visualize=visualize,)

    def generate_image(self, bg_color:Tuple[int,...],
                       group1_color:Tuple[int,...], group1_position:Tuple[int,...], 
                       group_element_size:Tuple[int,...], # all groups have the same element size
                       group2_color:Tuple[int,...]= None,group2_position:Tuple[int,...]= None,  
                       group3_color:Tuple[int,...]= None, group3_position:Tuple[int,...]= None,
                       mode = "separate",
                       image_size = 64, channel_num=3): 
        image = np.ones((image_size,image_size,channel_num))*np.array(bg_color).reshape(1,1,self.channel_num)
        
        if mode == "separate":
            group1 = GroupOfElements((Rectangle(), Rectangle()), group_element_size, group1_color, 1, group1_position)
            group2 = GroupOfElements((Rectangle(), Rectangle()), group_element_size, group2_color, 1, group2_position)
            group3 = GroupOfElements((Rectangle(), Rectangle()), group_element_size, group3_color, 1, group3_position)
            # it will be specified outside this function to ensure the spacing between different groups is larger than 1
            for group in [group1, group2, group3]: image = group.draw(image)
        elif mode == "same":
            elements = (Rectangle(), Rectangle(), Rectangle(), Rectangle(), Rectangle(), Rectangle())
            group = GroupOfElements(elements, group_element_size, group1_color, 1, group1_position)
            image = group.draw(image)
        else: raise Exception("mode must be 'separate' or 'same'.")
        return image/255
        
    def generate_mask(self, group1_position:Tuple[int,...], group_element_size:Tuple[int,...], # all groups have the same element size
                       group2_position:Tuple[int,...]= None,  
                       group3_position:Tuple[int,...]= None,
                       mode = "separate",
                       image_size = 64, ):
        mask = np.zeros((image_size, image_size))
        
        if mode == "separate":
            group1 = GroupOfElements((Rectangle(), Rectangle()), group_element_size, (1,), 1, group1_position)
            group2 = GroupOfElements((Rectangle(), Rectangle()), group_element_size, (2,), 1, group2_position)
            group3 = GroupOfElements((Rectangle(), Rectangle()), group_element_size, (3,), 1, group3_position)
            # it will be specified outside this function to ensure the spacing between different groups is larger than 1
            for group in [group1, group2, group3]: mask= group.draw(mask)
        elif mode == "same":
            elements = (Rectangle(), Rectangle(), Rectangle(), Rectangle(), Rectangle(), Rectangle())
            group = GroupOfElements(elements, group_element_size, (1,), 1, group1_position)
            mask = group.draw(mask)
        else: raise Exception("mode must be 'separate' or 'same'.")
        return mask.squeeze()

    def generate_train_val_dataset(self, train_image_num=30000, valid_image_num=300, test_image_num = 300,
                                   bg_color_min = 6/7, bg_color_max = 1, shape_color_min = 3/7, shape_color_max = 5/7,
                                   square_size = 7, bounding_box_size = (8, 7*6+2*3+3*1+5), min_intergroup_gap = 3, same_ratio = None,
                                   downsampling = False, downsampling_factor = 4,
                                   seed = 42):
        # [Bounding box]: It is desirable that the samples generated by the "separate" mode sufficiently
        # support the proximity of samples generated by the "same" mode. Here, under the
        # "separate" mode, we only generate samples with three groups of shapes that are roughly 
        # on the same row and do not overlap columnwise. To achieve this, we specify a bounding box
        # first, and then draw the three groups of shapes within it. The three groups of shapes
        # can occupy arbitrary rows in the bounding box, but can only occupy different columns, and are
        # subject to the requirements of minimal spacings between the shapes and groups.
        assert bounding_box_size[0] >= square_size, "The bounding box need to be higher than the square!"
        assert bounding_box_size[1] >= 6*square_size+2*min_intergroup_gap + 3, "The bounding box need to be wider than 3 groups of 2-square with a spacing larger than 2 between groups and exactly one within groups!"
        self.bg_color_min, self.bg_color_max = bg_color_min, bg_color_max
        self.shape_color_min, self.shape_color_max = shape_color_min, shape_color_max
        self.square_size = square_size
        self.min_intergroup_gap=min_intergroup_gap
        self.train_val_seed = seed
        self.downsampling = downsampling
        self.downsampling_factor = downsampling_factor

        rng = np.random.default_rng(seed = seed)

        # [Column position]: In the ProximityGroupingDataset and the GroupingDataset, we need to determine the columns occupied 
        # by shapes. To achive this, we first calculate the number of free pixels. Free pixels are,
        # more accurately, free "columns" that are not taken by the groups of shapes and 
        # the necessary spacings between these shapes. Then, arranging the column position of 
        # the shapes can be understood as inserting the groups of shapes (and minimal spacings
        # between these shapes) in between these free pixels. To make sure that all possible ways 
        # of column position arrangements have the same probability (which is a reasonable assumption
        # for the setup of the dataset), we use generate_possible_free_pixel_offset() to generate all possible ways of occupying the 
        # column positions, and then select from them uniformly.
        free_pixel_num = bounding_box_size[1] - 6*square_size -3 - 2*min_intergroup_gap # 3 is yielded by: the minimal spacing (1) between shapes within each group times the group number (3).
        possible_free_pixel_offset = generate_possible_free_pixel_offset(free_pixel_num = free_pixel_num)
        possible_free_pixel_offset_num = len(possible_free_pixel_offset)
        group_size = 2*square_size + 1 # columns of groups occupied by one group.

        print(f"======generating {self.savedir_name} training & validation dataset======")
        print(f"same_ratio:{same_ratio}")
        image_ls = []
        test_mask_ls = []
        
        for i in tqdm(range(train_image_num+valid_image_num+test_image_num)):
            bg_color = rng.integers(np.ceil(bg_color_min*255), np.floor(bg_color_max*255)+1, self.channel_num)
            # generate a bounding box
            bounding_box_position = [rng.integers(0,self.image_size-bounding_box_size[o]+1) for o in range(2)]
            if rng.uniform()<same_ratio:
                # generate an image with 6 shapes being in a one group
                group_color = rng.integers(np.ceil(shape_color_min*255), np.floor(shape_color_max*255)+1, self.channel_num)
                # calculate the offset
                big_group_size = [square_size, square_size*6+5]
                group_position_offset = [rng.integers(0,bounding_box_size[o]-big_group_size[o]+1) for o in range(2)]
                group_position = tuple(np.array(group_position_offset) + np.array(bounding_box_position))
                image = self.generate_image(bg_color=bg_color, group1_color = tuple(group_color), group1_position=group_position,
                                            group_element_size = (square_size, square_size), mode = "same",
                                            image_size=self.image_size, channel_num=self.channel_num)
                if i >= train_image_num + valid_image_num:
                    mask = self.generate_mask(group1_position=group_position, group_element_size=(square_size, square_size),
                                            mode = "same", image_size = self.image_size )
            else:
                group1_color, group2_color, group3_color = \
                [rng.integers(np.ceil(shape_color_min*255), np.floor(shape_color_max*255)+1, self.channel_num) for g in range(3)]
                # calculate the position of the grouped squares
                groups_row_offset_ls = [rng.integers(0,bounding_box_size[0]-square_size+1) for o in range(3)]
                # below is a combinatorial problem...
                free_pixel_offset = possible_free_pixel_offset[rng.choice(possible_free_pixel_offset_num)]
                # [Column offset]: The below list contains all the column offset of the three groups.
                # To calculate the offset from free_pixel_offset,
                # we need to take into account the column number taken up by squares and intervals, by 
                # using the min_intergroup_gap, we enforce that the spacings between different groups
                # cannot be smaller than this number.
                groups_col_offset_ls = \
                    [free_pixel_offset[0], free_pixel_offset[1] + group_size+min_intergroup_gap, free_pixel_offset[2] + 2*group_size + 2*min_intergroup_gap]
                group1_position, group2_position, group3_position = \
                [(bounding_box_position[0]+groups_row_offset_ls[p], bounding_box_position[1]+groups_col_offset_ls[p]) for p in range(3)]
                image = self.generate_image(bg_color=bg_color, group_element_size=(square_size, square_size),
                                            group1_color=tuple(group1_color), group1_position=group1_position,
                                            group2_color = tuple(group2_color), group2_position = group2_position, 
                                            group3_color = tuple(group3_color), group3_position = group3_position,
                                            mode="separate", image_size = self.image_size, channel_num = self.channel_num)
                if i >= train_image_num + valid_image_num:
                    mask = self.generate_mask(group1_position=group1_position, group_element_size=(square_size, square_size),
                                            group2_position=group2_position, group3_position=group3_position, mode = "separate", image_size = self.image_size )

            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if not self.downsampling: image = image.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            image_ls.append(image)
            if i >= train_image_num + valid_image_num:
                mask = np.expand_dims(mask, axis = 0)
                if not self.downsampling: mask = mask.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
                test_mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        test_masks = np.concatenate(test_mask_ls, axis = 0)
        train_images = images[:train_image_num]*999/1000 
        valid_images = images[train_image_num:train_image_num+valid_image_num]*998/1000
        test_images = images[train_image_num+valid_image_num:]

        print("saving ...")
        save_train_val_dataset(savedir_name=self.savedir_name, train_images=train_images, 
                               valid_images=valid_images, test_images=test_images, test_masks = test_masks,
                               visualize=self.visualize)

        self.train_val_dataset_generated = True

    def generate_test_dataset(self, test_image_num = 500, seed = 41):
        assert self.train_val_dataset_generated, "Need to generate training and validation dataset first!"
        if seed == self.train_val_seed: warnings.warn("It is highly recommended that you use different seeds to generate the training/validation dataset and the testing dataset.")

        rng = np.random.default_rng(seed = seed)
        print(f"======generating {self.savedir_name} testing dataset======")
        image_ls = []
        mask_ls = []
        group_size = 2*self.square_size + 1

        for i in tqdm(range(test_image_num)):
            bg_color = rng.integers(np.ceil(self.bg_color_min*255), np.floor(self.bg_color_max*255)+1, self.channel_num)
            group_color = rng.integers(np.ceil(self.shape_color_min*255), np.floor(self.shape_color_max*255)+1, self.channel_num)
            if rng.uniform() < 0.5: # generate a test image with one big group of 6 squares
                # calculate the offset
                big_group_size = (self.square_size, self.square_size*6+5)
                group_position = tuple([rng.integers(0,self.image_size-big_group_size[m]+1) for m in range(2)])
                image = self.generate_image(bg_color=bg_color, group1_color = tuple(group_color), group1_position=group_position,
                                            group_element_size = (self.square_size, self.square_size), mode = "same",
                                            image_size=self.image_size, channel_num=self.channel_num)
                mask = self.generate_mask(group1_position = group_position, group_element_size = (self.square_size,self.square_size),
                                          mode = "same")
            else: # generate an image with 3 groups of squares, separated by an interval of min_intergroup_gap, with the same color
                big_group_size = (self.square_size, self.square_size*6+3+2*self.min_intergroup_gap)
                group1_position = tuple([rng.integers(0,self.image_size-big_group_size[m]+1) for m in range(2)])
                group2_position = tuple(np.array(group1_position)+np.array([0,group_size+self.min_intergroup_gap]))
                group3_position = tuple(np.array(group2_position)+np.array([0,group_size+self.min_intergroup_gap]))
                image = self.generate_image(bg_color=bg_color, group_element_size=(self.square_size, self.square_size),
                                            group1_color=tuple(group_color), group1_position=group1_position,
                                            group2_color = tuple(group_color), group2_position = group2_position, 
                                            group3_color = tuple(group_color), group3_position = group3_position,
                                            mode="separate", image_size = self.image_size, channel_num = self.channel_num)
                mask = self.generate_mask(group1_position=group1_position, group2_position = group2_position, group3_position = group3_position,
                                          group_element_size = (self.square_size,self.square_size), mode = "separate" )
                                
            image = np.einsum('ijk->kij',image)
            image = np.expand_dims(image,axis = 0)
            if not self.downsampling: image = image.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            image_ls.append(image)

            mask = np.expand_dims(mask, axis = 0)
            if not self.downsampling: mask = mask.repeat(self.downsampling_factor,-1).repeat(self.downsampling_factor,-2)
            mask_ls.append(mask)

        images = np.concatenate(image_ls, axis = 0)
        masks = np.concatenate(mask_ls, axis = 0)

        print("saving ...")
        save_test_dataset(savedir_name = self.savedir_name, test_image_num=test_image_num,
                          images = images, masks = masks, visualize = self.visualize)


def generate_celebA_dataset(savedir = "CelebAMask-HQ", suffix = "", total_image_num = 15*2000, image_size = 64, 
                            valid_image_num = 300, test_image_num = 300, visualize = True):
    ### adapted from https://github.com/zllrunning/face-parsing.PyTorch/blob/master/face_dataset.py
    face_data = os.path.join(savedir,'CelebA-HQ-img')
    face_sep_mask =os.path.join(savedir,'CelebAMask-HQ-mask-anno')
    counter = 0
    total = 0
    assert 512%image_size == 0
    mask_sample_interval = int(512/image_size)
    image_sample_interval = int(1024/image_size)

    mask_ls = []
    image_ls = []

    pbar = tqdm(total=total_image_num)
    for i in range(15):

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in range(i*2000, (i+1)*2000):
            if j >= total_image_num: break
            pbar.update(1)
            mask = np.zeros((512, 512))
            down_sampled_mask = np.zeros((image_size, image_size))

            # get the down-sampled masks
            if j < test_image_num: # only register test images' masks
                for l, att in enumerate(atts, 1):
                    total += 1
                    file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                    path = os.path.join(face_sep_mask, str(i), file_name)
                    if os.path.exists(path):
                        counter += 1
                        sep_mask = np.array(Image.open(path).convert('P'))
                        # print(np.unique(sep_mask))

                        mask[sep_mask == 225] = l # l is the number of the face
                
                # down sample the mask according to image_size
                down_sampled_mask = mask[::mask_sample_interval,::mask_sample_interval]
                mask_ls.append(np.expand_dims(down_sampled_mask, axis = 0))

            # get the down-sampled images
            file_name = ''.join([str(j), '.jpg'])
            path = os.path.join(face_data,  file_name)
            image = np.array(Image.open(path).convert('RGB'))
            down_sampled_image = image[::image_sample_interval,::image_sample_interval]
            image_ls.append(np.expand_dims(down_sampled_image, axis = 0))

    detailed_masks = np.concatenate(mask_ls, axis = 0)
    images = np.concatenate(image_ls, axis = 0)
    images = np.einsum('lijc->lcij',images)/255 # 0~255 -> 0~1

    test_images = images[:test_image_num]
    test_detailed_masks = detailed_masks
    valid_images = images[test_image_num:valid_image_num+test_image_num]
    train_images = images[valid_image_num+test_image_num:]

    save_train_val_dataset('dCelebA'+suffix, train_images=train_images, valid_images=valid_images,test_images=None,
                        test_detailed_masks=None, visualize=visualize)
    # We do not generate test1.npz for this dCelebA dataset.
    save_test_dataset(savedir_name = 'dCelebA'+suffix, test_image_num=test_image_num,
                          images = test_images, masks = test_detailed_masks, detailed_masks = test_detailed_masks, visualize = visualize)

def show_images_grid(imgs_, num_images=25, vmin = None, vmax = None, random_select = False):
    """This function comes from dsprites_reloading_example.ipynb, which is the tutorial provided by 
    dsprites' creators. This function shows a number of images in a grid. 

    Args:
        imgs_ (_type_): _description_
        num_images (int, optional): _description_. Defaults to 25.
    """
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    if random_select:
        img_num = imgs_.shape[0]
        selected_id = np.random.choice(img_num, num_images, replace = False)
    else:
        selected_id = np.linspace(0,num_images-1, num_images, dtype=int)
    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[selected_id[ax_i]].squeeze(), cmap='Greys_r', vmin = vmin, vmax = vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

def generate_possible_free_pixel_offset(free_pixel_num,):
    """generate all possible combination of intervals among all the free pixels that might be occupied by three groups of shapes"""
    legit_pos_num = free_pixel_num+1 # given n free pixels, there are n+1 intervals
    possible_pos_comb = [[i,j,k] for i in range(legit_pos_num) for j in range(legit_pos_num) for k in range(legit_pos_num) if i<=j<=k]
    assert len(possible_pos_comb) == comb(free_pixel_num +3,3) # sanity check 
    return possible_pos_comb

def save_train_val_dataset(savedir_name, train_images, valid_images, test_images, test_masks = None, visualize = True,
                           test_detailed_masks = None ):
    os.makedirs(savedir_name,exist_ok=True)
    train_savepath = os.path.join(savedir_name,"train.npz")
    np.savez_compressed(train_savepath, images = train_images)
    valid_savepath = os.path.join(savedir_name,"valid.npz")
    np.savez_compressed(valid_savepath, images = valid_images)
    test_savepath = os.path.join(savedir_name,"test_1.npz")
    np.savez_compressed(test_savepath, images = test_images, masks = test_masks, detailed_masks = test_detailed_masks )

    # visualize
    if visualize:
        show_images_grid(np.einsum('lkij->lijk',train_images))
        plt.savefig(os.path.join(savedir_name,"training_samples.png"), dpi = 300)
        plt.close()

def save_test_dataset(savedir_name, test_image_num, images, masks = None, detailed_masks = None,
                      visualize = True):
    os.makedirs(savedir_name,exist_ok=True)
    train_savepath = os.path.join(savedir_name,"test.npz")
    np.savez_compressed(train_savepath, images = images, masks = masks, detailed_masks = detailed_masks)

    if visualize:
        show_images_grid(np.einsum('lkij->lijk',images),num_images = np.min((test_image_num, 25)), random_select=False)
        plt.savefig(os.path.join(savedir_name,"testing_samples.png"), dpi = 300)
        plt.close()
        if masks is not None:
            show_images_grid(masks,num_images = np.min((test_image_num, 25)), random_select=False)
            plt.savefig(os.path.join(savedir_name,"testing_mask_samples.png"), dpi = 300)
            plt.close()
        if detailed_masks is not None:
            show_images_grid(detailed_masks,num_images = np.min((test_image_num, 25)), random_select=False) # the /3 is jsut an artifact. leave it there!
            plt.savefig(os.path.join(savedir_name,"testing_detailed_mask_samples.png"), dpi=300)
            plt.close()


def main():
    
    parser = parse_arguments()
    args, unparsed = parser.parse_known_args()

    high_res = args.high_res
    include_celeba = args.include_celeba

    train_image_num_k = 30
    train_image_num = train_image_num_k*1000
    valid_image_num = 300
    test_image_num= 100

    if not high_res: 
        downsampling = True
        suffix = "" # For default datasets, no suffix

        closure_dataset = ClosureDataset(suffix=suffix)
        closure_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num, test_image_num=test_image_num, downsampling=downsampling)
        closure_dataset.generate_test_dataset(test_image_num=test_image_num)

        illusory_ocllusion_dataset = IllusoryOcclusionDataset(suffix=suffix)
        illusory_ocllusion_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num, test_image_num=test_image_num,downsampling=downsampling)
        illusory_ocllusion_dataset.generate_test_dataset(test_image_num=test_image_num)

        kanizsa_dataset = KanizsaDataset(suffix=suffix)
        kanizsa_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, downsampling=downsampling)
        kanizsa_dataset.generate_test_dataset(test_image_num)

        contiuity_dataset = ContinuityDataset(suffix=suffix)
        contiuity_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, downsampling=downsampling)
        contiuity_dataset.generate_test_dataset(test_image_num)

        gradient_occlusion_dataset = GradientOcclusionDataset(suffix = suffix)
        gradient_occlusion_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, downsampling=downsampling)
        gradient_occlusion_dataset.generate_test_dataset(test_image_num)

        proximity_grouping_dataset = ProximityGroupingDataset(suffix=suffix)
        proximity_grouping_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, same_ratio=0.30, downsampling=downsampling)
        proximity_grouping_dataset.generate_test_dataset(test_image_num)

        ## The following line generates the dCelebA dataset.
        if include_celeba: 
            generate_celebA_dataset(suffix = suffix, valid_image_num=valid_image_num, test_image_num = test_image_num)

    ## The following codes, not used in this project, are to provide the dataset with a higher resolution (256 × 256)
    else:
        downsampling = False
        suffix = f"_256" # indicating that the image size is 256 × 256

        closure_dataset = ClosureDataset(suffix=suffix)
        closure_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num, test_image_num=test_image_num, downsampling=downsampling)
        closure_dataset.generate_test_dataset(test_image_num=test_image_num)

        illusory_ocllusion_dataset = IllusoryOcclusionDataset(suffix=suffix)
        illusory_ocllusion_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num, test_image_num=test_image_num,downsampling=downsampling)
        illusory_ocllusion_dataset.generate_test_dataset(test_image_num=test_image_num)

        kanizsa_dataset = KanizsaDataset(suffix=suffix)
        kanizsa_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, downsampling=downsampling)
        kanizsa_dataset.generate_test_dataset(test_image_num)

        contiuity_dataset = ContinuityDataset(suffix=suffix)
        contiuity_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, downsampling=downsampling)
        contiuity_dataset.generate_test_dataset(test_image_num)

        gradient_occlusion_dataset = GradientOcclusionDataset(suffix = suffix)
        gradient_occlusion_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, downsampling=downsampling)
        gradient_occlusion_dataset.generate_test_dataset(test_image_num)

        proximity_grouping_dataset = ProximityGroupingDataset(suffix=suffix)
        proximity_grouping_dataset.generate_train_val_dataset(train_image_num=train_image_num, valid_image_num=valid_image_num,test_image_num=test_image_num, same_ratio=0.30, downsampling=downsampling)
        proximity_grouping_dataset.generate_test_dataset(test_image_num)

        ## The following line generates the dCelebA dataset.
        if include_celeba: 
            generate_celebA_dataset(suffix = suffix, valid_image_num=valid_image_num, test_image_num = test_image_num, image_size = 256)


def parse_arguments():
    parser = argparse.ArgumentParser(description='dataset creation parser')

    parser.add_argument('--high_res', default=False, type=str2bool, help='If True, then generate 256*256-sized image. Otherwise, generate 64*64-sized image.')
    parser.add_argument('--include_celeba', default=False, type=str2bool, help='If True, then generate the CelebA dataset, which takes more computing resource.')

    return parser

if __name__ == '__main__':
    main()



