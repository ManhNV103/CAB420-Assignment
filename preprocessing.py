import os
import cv2
import random
import math
from xml.etree import ElementTree # xml parsing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class BloodCellDataset:

    def __init__(self, data_dir=None):

        if data_dir:
            self.data_base_dir = data_dir
        else:
            self.data_base_dir = os.path.join("BCCD_Dataset", "BCCD")

        self.image_dir = os.path.join(self.data_base_dir, "JPEGImages")
        self.annotation_dir = os.path.join(self.data_base_dir, "Annotations")

        if not os.path.exists(self.image_dir) or \
           not os.path.exists(self.annotation_dir):

            error_text = "%s was passed as data_dir which does not point " + \
                         "the dataset base directory. The parameter " + \
                         "data_dir must point to the directory where " + \
                         "'JPEGImages' and 'Annotations' are located."
            error_text = error_text % self.data_base_dir

            raise ValueError(error_text)

    def get_classes(self, annotations):
        """
        given an iterable of DataFrames representing image annotations,
        this function counts the number of classes as the number of
        unique annotation["class"] values

        @param data: Iterable of annotation

        @return:
            nuber of distinct classes
        """
        classes = set()
        for annotation in annotations:
            for class_name in annotation["class"]:
                classes.add(class_name)

        self.classes = classes
        return list(classes)

    def parse_xml_annotation(self, annotation_file):
        '''
        Parses an annotation file associated with a training image
        that is given in xml format to extract the locations and
        class of bounding boxes.

        @param annotation_file: path to the xml file

        @returns:
            a dataframe with a row for each bounding box in the image
        '''
        tree = ElementTree.parse(annotation_file)
        root = tree.getroot()

        objects = []

        for obj in root.findall(".//object"):
            obj_dict = dict()

            # object class
            obj_dict["class"] = obj.find(".//name").text

            # bounding box
            obj_dict["x_min"] = int(obj.find(".//xmin").text)
            obj_dict["x_max"] = int(obj.find(".//xmax").text)
            obj_dict["y_min"] = int(obj.find(".//ymin").text)
            obj_dict["y_max"] = int(obj.find(".//ymax").text)

            objects.append(obj_dict)

        objects_df = pd.DataFrame(objects)

        return objects_df

    def load_all(self,
                 image_dir=None,
                 annotation_dir=None,
                 annotation_ending=".xml"):
        '''
        Loads all images from image dir and for each image attempts to loads an 
        annotation with the same base name and the ending 'annotation_ending' from
        the annotation_dir.

        @param image_dir: relative or absolute path to image directory
        @param annotation_dir: relative or absolute path to annotation directory. 
            Annotations are assumed to be in xml format.
        @param annotation_ending: file ending that all annotation xml files are 
            expected to have.

        @return: 
            (List of images, list of annotations) where each image is a cv2 
            representation of the image and annotation is a pandas table 
            with bounding box positions and class annotations.
        '''
        image_dir = image_dir or self.image_dir
        annotation_dir = annotation_dir or self.annotation_dir

        images = []
        annotations = []

        for image_name in os.listdir(image_dir):
            try:
                base_name, _ = image_name.split(".")
            except ValueError as e:
                print(e)
                print("[!] Image %s has invalid name/ending and will be skipped" % image_name)
                continue

            # construct the path of the annotation file based on the image name
            annotation_name = base_name + annotation_ending
            annotation_filepath = os.path.join(annotation_dir, annotation_name)

            # construct the full path of the image file
            image_filepath = os.path.join(image_dir, image_name)

            # load image and annotation
            annotation = self.parse_xml_annotation(annotation_filepath)
            image = cv2.imread(image_filepath)

            images.append(image)
            annotations.append(annotation)

        self.n_images = len(images)

        return images, annotations

    def plot_images(self, images, n=10, randomize=True, show=False):
        '''
        Displays a subset of the given list of images.

        @param data: iterable of cv2 images
        @param n: number of images ot be displayed
        @param randomize: if True, images displayed will be randomly
            drawn.
        '''

        if not show:
            plt.ioff()

        fig_cols = 5
        fig_rows = max(1, n // fig_cols)

        fig = plt.figure(figsize=(3*fig_cols,3*fig_rows))

        if randomize:
            images = random.sample(images, n)

        for i, img in enumerate(images, start=1):
            if i > n:
                break

            ax = fig.add_subplot(fig_rows, fig_cols, i)
            ax.imshow(img)
            ax.axis("off")

        fig.tight_layout()

        if show:
            plt.show()
        else:
            plt.close()
            return fig

    def apply_bounding_boxes(self, images, annotations, inplace=False):
        '''
        Given a list of images and annotations, returns a copy of the images
        with the annotated bounding boxes drawn.

        @param images: list of cv2 images
        @param annotation: list of DataFrames representing image annotations

        @return:
            list of cv2 images
        '''
        out = []

        classes = self.get_classes(annotations)
        n_classes = len(classes)
        # color palette, such that every class is represented
        # by a unique easily distinuishable color
        palette = [(int(r*255), int(g*255), int(b*255))
                    for (r,g,b) in sns.color_palette(None, n_classes)]
        colors = {cls: col for cls, col in zip(classes, palette)}

        line_thickness = 2

        for or_img, annotation in zip(images, annotations):

            # modify the images that were passed
            if inplace:
                image = or_img
            else:
                image = or_img.copy()

            for i, bnd_box in annotation.iterrows():
                color = colors[bnd_box["class"]]
                p1 = (int(bnd_box["x_min"]), int(bnd_box["y_min"]))
                p2 = (int(bnd_box["x_max"]), int(bnd_box["y_max"]))
#                print("p1: %s, p2: %s" %(p1,p2))
                cv2.rectangle(image, p1, p2, color, line_thickness)
                out.append(image)

        return out

    def apply_downscale(self, images, annotations, factor=0.5):
        '''
        Scale all images and their annotated bounding boxes by a scaling
        factor. It is asserted, that the scaling fator be lower than one

        @param images: list of cv2 images
        @param annotations: list of pandas DataFrames representing annotations

        @return:
            list of resized images, list of adjusted annotations
        '''

        assert factor < 1, "Scaling-factor must be < 1 for downsampling"

        res_images = list()
        res_annotations = list()

        try:
            old_shape = images[0].shape
            new_shape = (math.floor(old_shape[1] * factor),
                         math.floor(old_shape[0] * factor))
        except IndexError:
            return None, None

        for img in images:
            downscaled = cv2.resize(img, new_shape)

            res_images.append(downscaled)

        for ann in annotations:
            res_ann = dict()

            res_ann["class"] = ann["class"].values
            res_ann["x_min"] = np.floor(factor * ann["x_min"].values)
            res_ann["x_max"] = np.floor(factor * ann["x_max"].values)
            res_ann["y_min"] = np.floor(factor * ann["y_min"].values)
            res_ann["y_max"] = np.floor(factor * ann["y_max"].values)

            res_ann_df = pd.DataFrame(res_ann)
            res_annotations.append(res_ann_df)

        return res_images, res_annotations
