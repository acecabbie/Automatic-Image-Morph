import numpy as np

import config
import warper
from warper import Face, Warper


class ParappaTheFaceWrappa:
    def __init__(self, parameters):
        """
        Initializes the face morphing setup with given parameters.

        Parameters should be a dictionary like:
        {
            'face1': <filepath to face1>,
            'face2': <filepath to face2>,
            'align': True or False,
            'resizeToFit': True or False,
            'dlib': True or False,              # Use Dlib detector; if False, use custom CNN
            'addBackgroundPoints': True or False
        }
        """
        self.parameters = parameters
        self.path1 = parameters['face1']
        self.path2 = parameters['face2']
        self.selected_indices=[]
        self.selected_indices = config.selected_indices.copy()

        self.background = parameters.get('addBackgroundPoints', False)
        self.shrinkValue = parameters.get('shrinkValue', 256)
        self.backAlign=False
        self.model_points = config.model_points


        self.original_image = None
        if(self.background and self.parameters.get("align",False)):
            self.backAlign=True

    def setup_faces(self):
        """
        Loads the faces, applies optional resizing and alignment, and initializes the Warper object.
        """
        # Choose keypoint detector
        redoFlag=False
        cnn_kp_detector = warper.CNNDetector()

        if(self.backAlign):
            self.background=False
        if self.parameters.get('dlib', False):
            face1 = Face(self.path1, None, None, self.background)
            face2 = Face(self.path2, None, None, self.background)
        else:

            face1 = Face(self.path1, None, None, self.background, detector=cnn_kp_detector)
            face2 = Face(self.path2, None, None, self.background, detector=cnn_kp_detector)

        # Resize face1 to match face2, or vice versa

        if(face1.imc1.shape!=face2.imc1.shape):
            redoFlag=True
            resize1, resize2 = face1.resize_to_fit(face2)
            face1.imc1=resize1
            face2.imc1=resize2

            #face1 = Face(self.path1, resize1, None, self.background,detector=cnn_kp_detector)
            #face2 = Face(self.path2, resize2, None, self.background,detector=cnn_kp_detector)

        else:
            print("images are the same size, skipping resizing")

        h1, w1 = face1.imc1.shape[:2]
        h2, w2 = face2.imc1.shape[:2]
        # Check if both are square
        if h1 == w1 and h2 == w2 and h1<=self.shrinkValue and h2<=self.shrinkValue:
            print("images are already square")
        else:
            redoFlag = True
            face1.imc1, face2.imc1 = Warper.resizeBoth(face1.imc1, face2.imc1)

        if(redoFlag):
            face1 = Face(self.path1, face1.imc1, None, self.background, detector=cnn_kp_detector)
            face2 = Face(self.path2, face2.imc1, None, self.background, detector=cnn_kp_detector)
            # print("image shap: ",face1.imc1.shape,face2.imc1.shape)


        if self.parameters.get("align", False):
            if(self.backAlign):
                self.background=True
            face1.center_face(self.background)
            face2.center_face(self.background)

        if self.parameters.get("alignCropOutBlack",False):
            bbox1 = self.warp.get_crop_box_from_keypoints(self.face1.points, 0, image_shape=self.face1.imc1.shape)
            bbox2 = self.warp.get_crop_box_from_keypoints(self.face2.points, 0, image_shape=self.face2.imc1.shape)
            combined_bbox = self.warp.union_boxes(bbox1, bbox2)

            cropped_im1, cropped_pts1 = self.warp.crop_image_and_keypoints(self.face1.imc1, self.face1.points,
                                                                           combined_bbox)
            cropped_im2, cropped_pts2 = self.warp.crop_image_and_keypoints(self.face2.imc1, self.face2.points,
                                                                           combined_bbox)
            face1.imc1=cropped_im1
            face2.imc1=cropped_im2
            face1.draw_reset_points(cropped_pts1)
            face2.draw_reset_points(cropped_pts2)

        self.original_image = face1.imc1
        self.face1 = face1
        self.face2 = face2
        self.warp = Warper(face1, face2)
        # Assume: im1, im2 are aligned images; pts1, pts2 are keypoints




    def morph(self, morphf, b=0, e=1, timelapse=10, custom_t=None, plot=False):
        """
        Performs the morphing or interpolation, depending on morphf flag.

        morphf = 1     → Standard morphing
        morphf = 422   → SLERP-based pose interpolation using model points

        :param b: Start interpolation value (default=0)
        :param e: End interpolation value (default=1)
        :param timelapse: Number of frames between b and e
        :param custom_t: If set, use this specific t for a single frame
        :return: List of morphed images
        """
        t_list = np.linspace(b, e, timelapse)
        imagelist = []

        if morphf == 1:
            if timelapse is not None:
                imagelist = self.warp.timeLapse(t_list, 0, plot=plot)
            else:
                imagelist.append(self.warp.morph(custom_t))
        elif morphf == 422:
            # Interpolation via 3D head rotation (SLERP)
            imagelist = self.warp.timeLapse(t_list, 422, self.model_points, plot=plot)
        elif morphf==4:
            imagelist = self.warp.timeLapse(t_list,4,self.model_points, plot=plot)

        return imagelist

    def typical_morph(self, b=0, e=1, timelapse=10, custom_t=None):
        """
        Wrapper around morph() for typical morphing use-case.
        """
        return self.morph(1, b, e, timelapse, custom_t)


    def typical_interpolate_morph(self, timelapse=10, before_and_after=-1, plot=False):
        """
        First interpolates head pose between face1 and face2 using SLERP + projection,
        then morphs face1 → interpolated → face2 in sequence.

        :param b: start value for typical morph
        :param e: end value for typical morph
        :param timelapse: frames during interpolation phase
        :param before_and_after: frames before and after interpolation
        :return: final morph sequence as list of images
        """
        # Save originals
        if(before_and_after<0):
            before_and_after=round(0.6*timelapse)
        face2_original = self.face2.imc1.copy()
        face1_original = self.face1.imc1.copy()

        # Highlight selected points
        self.face1.draw_reset_points(self.face1.points[self.selected_indices], False)
        self.face2.draw_reset_points(self.face2.points[self.selected_indices], False)

        self.warp.reset_mean()

        # Interpolation stage (head pose SLERP)
        if(not self.background):
            interpolated_images = self.morph(4, 0, 1, timelapse)
        else:
            interpolated_images = self.morph(422, 0, 1, timelapse)
        last_image = interpolated_images[-1]
        first_image = interpolated_images[0]
        original_face2d_points=self.face2.points
        original_face1_points=self.face1.points

        # Create new face objects from morphed intermediates
        if(not self.background):
            self.face1.imc1=last_image
            self.face1.points=self.warp.get_last(self.face1,self.face2,self.model_points,1,False)
            print("self points: ",self.face1.points.shape)
            self.face2 = Face('b', face2_original)

        else:
            self.face1.imc1 = last_image
            self.face1.points = self.warp.get_last(self.face1, self.face2, self.model_points, 1,True)
            print("self points: ", self.face1.points.shape)
            self.face2 = Face('b', face2_original)
            #self.face1 = Face('a', last_image)
            #self.face2 = Face('b', face2_original)
            #self.warp = Warper(self.face1, self.face2)

        # Morph from interpolated to face2
        interpolated_to_face2 = self.typical_morph(0.1, 1, before_and_after)
        if (not self.background):
            self.face2.imc1=first_image
            self.face2.points=original_face2d_points
            self.face1.points=original_face1_points
            print("self points: ", self.face2.points.shape, self.face1.points.shape)
            self.face2.points=self.warp.get_first(self.face1,self.face2,self.model_points)
            self.face1 = Face('a', face1_original)
            self.face1.points=original_face1_points
            print("self pointsEE: ", self.face2.points.shape, self.face1.points.shape)
            self.warp = Warper(self.face1, self.face2)

        else:
            self.face1 = Face('a', face1_original)
            self.face2 = Face('b', first_image)
            self.warp = Warper(self.face1, self.face2)
        face1_to_interpolated = self.typical_morph(0, 0.9, before_and_after)
        # Morph from face1 to interpolated
        #self.face1 = Face('a', face1_original)
        #self.face2 = Face('b', first_image)
        #self.warp = Warper(self.face1, self.face2)
        #face1_to_interpolated = self.typical_morph(0, 0.9, before_and_after)
        #face1_to_interpolated=[]

        # Final full morph sequence
        print("")
        return face1_to_interpolated + interpolated_images + interpolated_to_face2
    def convert_to_video(self,images,FPS,nameOfVideo, pad_beginning_frames: int=0, pad_end_frames: int=0):
        self.warp.convert_to_video(
            images,
            FPS,
            nameOfVideo,
            pad_beginning_frames=pad_beginning_frames,
            pad_end_frames=pad_end_frames,
        )
