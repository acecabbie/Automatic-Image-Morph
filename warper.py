from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from model.model import KeypointModel


class IFaceDetector(ABC):
    face_chip = None

    def __init__(self):
        pass

    @abstractmethod
    def get_landmarks(self, img, useFaceChip=False):
        raise NotImplementedError("Subclasses must implement get_landmarks()")
class CNNDetector(IFaceDetector):
    def __init__(self, model_path: Union[Path, str] = "best_model.keras"):
        super().__init__()
        self.model = KeypointModel(model_file='best_model.keras')

    def get_landmarks(self, img: np.ndarray, useFaceChip: bool = False):
        print("getting landmarks from CNN model")
        img_w, img_h, _ = img.shape
        img_imputed = self.model.image_imputer.forward(img)
        pred = self.model.predict(img_imputed, show=False).reshape(2, 68)
        pred[0] = pred[0] * img_w / self.model.keypoint_imputer.im_width
        pred[1] = pred[1] * img_h / self.model.keypoint_imputer.im_height
        return pred.T
class Warper:

    def __init__(self,face1,face2):
        self.face1 = face1
        self.face2 = face2
        self.mean_pts = 0.5 * face1.points + 0.5 * face2.points
        tri = Delaunay(self.mean_pts)
        self.triangle_indices = tri.simplices  # shape (n_triangles, 3)


        self.triangle_coords_1 = face1.points[self.triangle_indices]
        self.triangle_coords_2 = face1.points[self.triangle_indices]
        self.sample1=self.drawFace(self.face1)
        self.sample2 = self.drawFace(self.face2)
    def reset_mean(self):
        self.mean_pts = 0.5 * self.face1.points + 0.5 * self.face2.points
        tri = Delaunay(self.mean_pts)
        self.triangle_indices = tri.simplices  # shape (n_triangles, 3)

        # For each triangle, you can access the actual points:
        self.triangle_coords_1 = self.face1.points[self.triangle_indices]
        self.triangle_coords_2 = self.face1.points[self.triangle_indices]





    def estimate_pose(self, face,model3d_points):
        """Estimate rvec and tvec for a given face using 6 landmarks."""
        # Define 3D model points (fixed for all faces)

        # Select corresponding 2D points
        #indices = [30, 8, 36, 45, 48, 54]
        #indices = [39, 45, 8, 36, 42, 30, 66, 57, 62, 51, 14, 15, 1, 0]
        image_points = face.points

        model3d_points = model3d_points.astype(np.float32)
        image_points = image_points.astype(np.float32)
        h, w = face.imc1.shape[:2]
        focal_length = w
        center = (w // 2, h // 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model3d_points, image_points, camera_matrix, dist_coeffs
        )
        return rvec.flatten(), tvec.flatten()

    def timeLapse(self, tList, morphF, model_points=None, blend_colors=True,plot=False):
        """
        Generates a sequence of morphed images over time values in tList.

        Parameters:
        - tList: list of float values (0 to 1) representing morph progress
        - morphF: integer flag determining morphing method
        - model_points: 3D model points (optional, for pose morphing)
        - blend_colors: whether to blend colors between faces(standard morphing algorithm)

        Returns:
        - imageList: list of morphed image frames
        """
        imageList = []
        static_points_face_1 = self.face1.points.copy()
        static_points_face2 = self.face2.points.copy()
        h, w = self.face1.imc1.shape[:2]

        for t in tList:
            print("t: ", t)
            image = np.zeros((h, w, 3), dtype=np.uint8)

            if morphF == 4:
                # Morph with interpolated pose and blended warp
                image = self.morph_with_pose4(self.face1, self.face2, model_points, t, 0.5, 0.5, blend_colors)

            elif morphF == 422:
                # Morph with special pose method (version 422)
                image = self.morph_with_pose422(self.face1, self.face2, model_points, t,plot=plot)

                # Reset landmark positions and Delaunay triangulation
                self.face1.draw_reset_points(static_points_face_1)
                self.face2.draw_reset_points(static_points_face2)
                self.reset_mean()

            elif morphF == 0:
                image = self.morph(t)
            else:
                image = self.morph(t)

            imageList.append(image)

        return imageList

    

   


    def morph(self, t, blend_colors=True):
        """
        Basic image morphing using affine warping between two faces.

        Parameters:
        - t: interpolation parameter (0 to 1)
        - blend_colors: if True, blend pixel colors from both faces

        Returns:
        - morphed_img: the morphed image at time t
        """
        image = self.face1.imc1.copy()
        interp_pts = (1 - t) * self.face1.points + t * self.face2.points
        h, w = self.face1.imc1.shape[:2]
        morphed_img = np.zeros((h, w, 3), dtype=np.uint8)

        for tri in self.triangle_indices:
            tri1 = np.float32(self.face1.points[tri])
            tri2 = np.float32(self.face2.points[tri])
            tri_interp = np.float32(interp_pts[tri])

            # Affine transforms from each source triangle to the interpolated one
            M1 = cv2.getAffineTransform(tri1, tri_interp)
            M2 = cv2.getAffineTransform(tri2, tri_interp)
            M1_inv = cv2.invertAffineTransform(M1)
            M2_inv = cv2.invertAffineTransform(M2)

            for py in range(h):
                for px in range(w):
                    if cv2.pointPolygonTest(tri_interp, (px, py), False) >= 0:
                        # Back-map to source coordinates in both images
                        src1_x, src1_y = np.dot(M1_inv, [px, py, 1])
                        src2_x, src2_y = np.dot(M2_inv, [px, py, 1])

                        # Bilinear interpolation
                        color1 = self.bilinear_interpolate(self.face1.imc1, src1_x, src1_y)
                        color2 = self.bilinear_interpolate(self.face2.imc1, src2_x, src2_y)

                        if blend_colors:
                            blended = ((1 - t) * color1 + t * color2).astype(np.uint8)
                            morphed_img[py, px] = blended
                        else:
                            morphed_img[py, px] = color1

        return morphed_img

    def interpolate_pose(self, rvec1, tvec1, rvec2, tvec2, t):
        """
        Interpolates rotation and translation vectors using SLERP and linear blend.

        Parameters:
        - rvec1, rvec2: rotation vectors (Rodrigues format)
        - tvec1, tvec2: translation vectors
        - t: interpolation value (0 to 1)

        Returns:
        - rvec_interp: interpolated rotation vector
        - tvec_interp: interpolated translation vector
        """
        R1 = cv2.Rodrigues(rvec1)[0]
        R2 = cv2.Rodrigues(rvec2)[0]

        q1 = R.from_matrix(R1)
        q2 = R.from_matrix(R2)
        slerp = Slerp([0, 1], R.from_quat([q1.as_quat(), q2.as_quat()]))

        R_interp = slerp([t]).as_matrix()[0]
        rvec_interp = cv2.Rodrigues(R_interp)[0]
        tvec_interp = (1 - t) * tvec1 + t * tvec2

        return rvec_interp, tvec_interp

    def project_points(self, model_points, rvec, tvec, camera_matrix, dist_coeffs=np.zeros((4, 1))):
        """
        Projects 3D model points to 2D using the given pose and camera intrinsics.

        Parameters:
        - model_points: 3D facial keypoints (N×3)
        - rvec: rotation vector
        - tvec: translation vector
        - camera_matrix: 3×3 intrinsic camera matrix
        - dist_coeffs: distortion coefficients (default = no distortion)

        Returns:
        - 2D projected points (N×2)
        """
        proj_points, _ = cv2.projectPoints(model_points, rvec, tvec, camera_matrix, dist_coeffs)
        return proj_points.squeeze()

    def plot_face_projection(self,img, original_pts, projected_pts, title="Face Projection"):
        """
        Plots original 2D landmarks and PnP-projected 3D landmarks on top of the image.

        Parameters:
            img: the image (e.g., face1.imc1)
            original_pts: 2D keypoints used for PnP (shape Nx2)
            projected_pts: 3D model points projected using solvePnP (shape Nx2)
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.scatter(original_pts[:, 0], original_pts[:, 1], c='lime', label='Original 2D points', s=20)
        plt.scatter(projected_pts[:, 0], projected_pts[:, 1], c='red', label='PnP Projected Points', s=20, marker='x')
        plt.legend()
        plt.title(title)
        plt.axis('off')
        plt.show()


    def morph_with_pose4(self,face1, face2, model3d_points,  t,k=0.5,l=0.5,blend_colors=True):
        """
        Full morph: interpolate geometry, pose (rotation), and color.
        """


        h, w = face1.imc1.shape[:2]
        morphed_img = np.zeros((h, w, 3), dtype=np.uint8)
        # 1. Solve PnP for both faces
        model3d_pointsf = model3d_points.copy()
        model3d_pointsf[:, 1] *= -1  # Flip Y
        rvec1, tvec1 = self.solve_pnp(face1.points, model3d_pointsf, w, h)
        rvec2, tvec2 = self.solve_pnp(face2.points, model3d_pointsf, w, h)

        # 2. Project model points onto each face
        focal = w
        cam_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)


        interp_rvec,_=self.interpolate_pose(rvec1, tvec1, rvec2, tvec2, t)
        #interp_rvec=self.interpolate_pose_and_translation(interp_rvec, proj_pts1, proj_pts2, t)
        interp_tvec = (1 - t) * tvec1 + t * tvec2

        # 3. Project model points onto the morphed face using the interpolated pose
        projected_interp = self.project_points(model3d_pointsf, interp_rvec, interp_tvec, cam_matrix)
        # 4. Also interpolate shape (optional for identity morphing)
        interp_shape = (1 - t) * face1.points + t * face2.points

        # 5. Combine projected pose + shape morph (optional balance)
        interp_pts = k * projected_interp + l * interp_shape

       # print("rvec1:", rvec1.ravel())
       # print("rvec2:", rvec2.ravel())
       # print("Δrvec:", np.linalg.norm(rvec1 - rvec2))
        # 3. Interpolate 2D projections
        #interp_pts = (1 - t) * proj_pts1 + t * proj_pts2

        # 3. Warp triangles
        for tri in self.triangle_indices:
            tri1 = np.float32(face1.points[tri])
            tri2 = np.float32(face2.points[tri])
            tri_interp = np.float32(interp_pts[tri])

            # Get affine transforms
            M1 = cv2.getAffineTransform(tri1, tri_interp)
            M2 = cv2.getAffineTransform(tri2, tri_interp)
            M1_inv = cv2.invertAffineTransform(M1)
            M2_inv = cv2.invertAffineTransform(M2)

            for py in range(h):
                for px in range(w):
                    if cv2.pointPolygonTest(tri_interp, (px, py), False) >= 0:
                        src1_x, src1_y = np.dot(M1_inv, [px, py, 1])
                        src2_x, src2_y = np.dot(M2_inv, [px, py, 1])

                        color1 = self.bilinear_interpolate(face1.imc1,src1_x, src1_y)
                        color2 = self.bilinear_interpolate(face2.imc1,src2_x, src2_y)

                        if blend_colors:
                            blended = ((1 - t) * color1 + t * color2).astype(np.uint8)
                            morphed_img[py, px] = blended
                        else:
                            morphed_img[py, px] = color1

        return morphed_img

    def get_first(self, face1, face2, model3d_points, t=0, k=0.5, l=0.5, blend_colors=True):
        """

               Returns the status of the last image of face 2 at t=1, so the minimorph can be done to convert from 2d to 3d

        """
        

        h, w = face1.imc1.shape[:2]

        #  Solve PnP for both faces
        model3d_pointsf = model3d_points.copy()
        model3d_pointsf[:, 1] *= -1  # Flip Y
        rvec1, tvec1 = self.solve_pnp(face1.points, model3d_pointsf, w, h)
        rvec2, tvec2 = self.solve_pnp(face2.points, model3d_pointsf, w, h)


        # 2. Project model points onto each face
        focal = w
        cam_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        interp_rvec, _ = self.interpolate_pose(rvec1, tvec1, rvec2, tvec2, t)

        interp_tvec = tvec1

        projected_interp = self.project_points(model3d_pointsf, interp_rvec, interp_tvec, cam_matrix)
        interp_shape = face1.points

        # Combine projected pose + shape morph (optional balance)
        interp_pts = k * projected_interp + l * interp_shape



        return interp_pts
    def get_last(self,face1,face2,model3d_points,t,background,k=0.5,l=0.5,blend_colors=True):
        """
               Returns the status of the last image of face 2 at t=1, so the minimorph can be done to convert from 2d to 3d
               """


        h, w = face1.imc1.shape[:2]


        model3d_pointsf = model3d_points.copy()
        model3d_pointsf[:, 1] *= -1  # Flip Y
        rvec1, tvec1 = self.solve_pnp(face1.points, model3d_pointsf, w, h)
        rvec2, tvec2 = self.solve_pnp(face2.points, model3d_pointsf, w, h)

        t=1

        border_pts = np.array([
            [0, 0],  # top-left corner
            [w - 1, 0],  # top-right corner
            [w - 1, h - 1],  # bottom-right
            [0, h - 1],  # bottom-left
            [w // 2, 0],  # top edge midpoint
            [w - 1, h // 2],  # right edge midpoint
            [w // 2, h - 1],  # bottom edge midpoint
            [0, h // 2]  # left edge midpoint
        ])

        # 2. Project model points onto each face
        focal = w
        cam_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        interp_rvec, _ = self.interpolate_pose(rvec1, tvec1, rvec2, tvec2, t)

        interp_tvec =  tvec2

        # Project model points onto the morphed face using the interpolated pose
        projected_interp = self.project_points(model3d_pointsf, interp_rvec, interp_tvec, cam_matrix)

        if(background):
            projected_border = border_pts  # same for both faces
            projected_interp_ext = np.vstack([projected_interp, projected_border])
            points_2dmm = np.vstack([face1.points, border_pts])
            points_pro = np.vstack([face2.points, border_pts])
            face1.draw_reset_points(points_2dmm)
            face2.draw_reset_points(points_pro)
        else:
            projected_interp_ext=projected_interp



        self.reset_mean()
        interp_shape = face2.points

        # 5. Combine projected pose + shape morph (optional balance)
        interp_pts = k * projected_interp_ext + l * interp_shape





        return interp_pts
 
    def morph_with_pose422(self,face1, face2, model3d_points,  t,k=0.5,l=0.5,blend_colors=True, plot=False):
        """
        Full morph: interpolate geometry, pose (rotation), and color.
        """
        if plot:
            print("here is t: ", t,model3d_points.shape,face1.points.shape)

        h, w = face1.imc1.shape[:2]
        morphed_img = np.zeros((h, w, 3), dtype=np.uint8)
        # 1. Solve PnP for both faces
        model3d_pointsf = model3d_points.copy()
        model3d_pointsf[:, 1] *= -1  # Flip Y
        rvec1, tvec1 = self.solve_pnp(face1.points, model3d_pointsf, w, h)
        rvec2, tvec2 = self.solve_pnp(face2.points, model3d_pointsf, w, h)

        #  Project model points onto each face
        focal = w
        cam_matrix = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        border_pts = np.array([
            [0, 0],  # top-left corner
            [w - 1, 0],  # top-right corner
            [w - 1, h - 1],  # bottom-right
            [0, h - 1],  # bottom-left
            [w // 2, 0],  # top edge midpoint
            [w - 1, h // 2],  # right edge midpoint
            [w // 2, h - 1],  # bottom edge midpoint
            [0, h // 2]  # left edge midpoint
        ])


        interp_rvec,_=self.interpolate_pose(rvec1, tvec1, rvec2, tvec2, t)
        #interp_rvec=self.interpolate_pose_and_translation(interp_rvec, proj_pts1, proj_pts2, t)
        interp_tvec = (1 - t) * tvec1 + t * tvec2

        #  Project model points onto the morphed face using the interpolated pose
        projected_interp = self.project_points(model3d_pointsf, interp_rvec, interp_tvec, cam_matrix)

        #  Also interpolate shape (optional for identity morphing)
        projected_border = border_pts  # same for both faces
        projected_interp_ext = np.vstack([projected_interp, projected_border])

        if plot:
            self.plot_face_projection(face1.imc1, face1.points, projected_interp , title="PnP Projection for Face1")

        points_2dmm = np.vstack([face1.points, border_pts])
        points_pro = np.vstack([face2.points, border_pts])
        face1.draw_reset_points(points_2dmm)
        face2.draw_reset_points(points_pro)

        self.reset_mean()
        interp_shape = (1 - t) * face1.points + t * face2.points
        # 5. Combine projected pose + shape morph (optional balance)
        interp_pts = k * projected_interp_ext + l * interp_shape

        #print("rvec1:", rvec1.ravel())
        #print("rvec2:", rvec2.ravel())
        #print("Δrvec:", np.linalg.norm(rvec1 - rvec2))
        # 3. Interpolate 2D projections
        #interp_pts = (1 - t) * proj_pts1 + t * proj_pts2

        # 3. Warp triangles
        for tri in self.triangle_indices:
            tri1 = np.float32(face1.points[tri])
            tri2 = np.float32(face2.points[tri])
            tri_interp = np.float32(interp_pts[tri])

            # Get affine transforms
            M1 = cv2.getAffineTransform(tri1, tri_interp)
            M2 = cv2.getAffineTransform(tri2, tri_interp)
            M1_inv = cv2.invertAffineTransform(M1)
            M2_inv = cv2.invertAffineTransform(M2)

            for py in range(h):
                for px in range(w):
                    if cv2.pointPolygonTest(tri_interp, (px, py), False) >= 0:
                        src1_x, src1_y = np.dot(M1_inv, [px, py, 1])
                        src2_x, src2_y = np.dot(M2_inv, [px, py, 1])

                        color1 = self.bilinear_interpolate(face1.imc1,src1_x, src1_y)
                        color2 = self.bilinear_interpolate(face2.imc1,src2_x, src2_y)

                        if blend_colors:
                            blended = ((1 - t) * color1 + t * color2).astype(np.uint8)
                            morphed_img[py, px] = blended
                        else:
                            morphed_img[py, px] = color1

        return morphed_img








    def estimate_affine_3d_to_2d(self,X_3d, x_2d):

        N = X_3d.shape[0]
        A, b = [], []
        for i in range(N):
            X, Y, Z = X_3d[i]
            x, y = x_2d[i]
            A.append([X, Y, Z, 1, 0, 0, 0, 0])
            A.append([0, 0, 0, 0, X, Y, Z, 1])
            b.append(x)
            b.append(y)

        A = np.array(A)
        b = np.array(b)
        affine_params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return affine_params.reshape(2, 4)
    def solve_pnp(self,image_points, model3d_points, w, h):
        """ Helper function to estimate rvec, tvec from image and model points. """
        focal = w
        cam = np.array([
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist = np.zeros((4, 1))


        image_points = image_points
        model_points = model3d_points
        # print("image: ",image_points.shape,model3d_points.shape,flush=True)
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        model_points = np.asarray(model3d_points, dtype=np.float32).reshape(-1, 3)



        success, rvec, tvec = cv2.solvePnP(
            model_points,  # 3D object points
            image_points,  # 2D image points
            cam,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            raise Exception("solvePnP failed")
        return rvec, tvec







    def drawFace(self,face):
        image=face.imc1.copy()
        for tri in self.triangle_indices:
            pts = face.points[tri].astype(np.int32)
            sample= cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 0), thickness=1)  # yellow outline
        return sample
    def drawMeanFace(self,face):
        image=face.imc1.copy()
        for tri in self.triangle_indices:
            pts = self.mean_pts[tri].astype(np.int32)
            sample= cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=1)  # yellow outline
        return sample
    def convert_to_video(self,imagelist,FPS,nameOfVideo, pad_beginning_frames: int=0, pad_end_frames: int=0):
        frames = imagelist

        if pad_beginning_frames > 0:
            frames = [frames[0]] * pad_beginning_frames + frames
        if pad_end_frames > 0:
            frames = frames + [frames[-1]] * pad_end_frames

        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(nameOfVideo, cv2.VideoWriter_fourcc(*'XVID'), FPS, (width, height))

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(rgb_frame)

        out.release()


    def bilinear_interpolate(self,img, x, y):
        """
        Perform bilinear interpolation on image `img` at non-integer coordinates (x, y).
        img: H x W x C image (RGB or grayscale)
        x, y: float coordinates in the source image
        Returns: pixel value (1D array of length C)
        """

        h, w = img.shape[:2]

        # Clamp to valid range
        if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            return np.zeros(img.shape[2], dtype=np.uint8) if img.ndim == 3 else 0

        # Get integer pixel locations
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        # Compute interpolation weights
        dx, dy = x - x0, y - y0

        # Get pixel values at corners
        Ia = img[y0, x0].astype(float)
        Ib = img[y0, x1].astype(float)
        Ic = img[y1, x0].astype(float)
        Id = img[y1, x1].astype(float)

        # Interpolate horizontally then vertically
        top = Ia * (1 - dx) + Ib * dx
        bottom = Ic * (1 - dx) + Id * dx
        value = top * (1 - dy) + bottom * dy

        return value.astype(np.uint8)


    def align_image_with_keypoints_svd(self,img1,img2,pts1,pts2, output_size=None):
        """
        Aligns img2 to img1 using SVD-based similarity transform from keypoints.

        Parameters:
            img1 (ndarray): Reference image (not transformed)
            img2 (ndarray): Image to align to img1
            pts1 (Nx2 ndarray): Keypoints from img1
            pts2 (Nx2 ndarray): Corresponding keypoints from img2
            output_size (tuple): (width, height) of output image. Defaults to img1's size.

        Returns:
            aligned_img2 (ndarray): img2 warped to match img1 using estimated transform
            transform_matrix (3x3 ndarray): Full 2D affine transform matrix
        """
        # Ensure float32 for OpenCV compatibility

        #img1=self.face1.imc1
        #img2=self.face2.imc1
        #pts1=self.face1.points
        #pts2=self.face2.points
        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)

        # Compute centroids
        c1 = np.mean(pts1, axis=0)
        c2 = np.mean(pts2, axis=0)

        # Center the points
        pts1_centered = pts1 - c1
        pts2_centered = pts2 - c2

        # Compute scale
        scale1 = np.linalg.norm(pts1_centered)  # Frobenius norm
        scale2 = np.linalg.norm(pts2_centered)
        pts1_scaled = pts1_centered / scale1
        pts2_scaled = pts2_centered / scale2

        # Compute optimal rotation using SVD
        U, _, Vt = np.linalg.svd(pts2_scaled.T @ pts1_scaled)
        R = U @ Vt

        # Final similarity transform: s * R * x + t
        s = scale1 / scale2
        A = s * R
        t = c1 - A @ c2

        # Build full 2D affine transform matrix
        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] = A
        transform_matrix[:2, 2] = t

        # Apply warp
        output_size = output_size or (img1.shape[1], img1.shape[0])
        aligned_img2 = cv2.warpAffine(img2, transform_matrix[:2], output_size)

        return aligned_img2, transform_matrix

    def crop_image_and_keypoints(self,image, keypoints, crop_box):
        x_min, y_min, x_max, y_max = crop_box
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_keypoints = keypoints - np.array([x_min, y_min])
        return cropped_image, cropped_keypoints

    def union_boxes(self,box1, box2):
        x_min = min(box1[0], box2[0])
        y_min = min(box1[1], box2[1])
        x_max = max(box1[2], box2[2])
        y_max = max(box1[3], box2[3])
        return (x_min, y_min, x_max, y_max)

    def get_crop_box_from_keypoints(self,keypoints, padding=10, image_shape=None):
        x_min, y_min = np.min(keypoints, axis=0).astype(int)
        x_max, y_max = np.max(keypoints, axis=0).astype(int)

        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = x_max + padding
        y_max = y_max + padding


        if image_shape:
            h, w = image_shape[:2]
            x_max = min(x_max, w)
            y_max = min(y_max, h)

        return (x_min, y_min, x_max, y_max)
    @staticmethod
    def resizeBoth(face1Img,face2Img):
        resized1 = cv2.resize(face1Img, (256, 256), interpolation=cv2.INTER_LINEAR)
        resized2 = cv2.resize(face2Img, (256, 256), interpolation=cv2.INTER_LINEAR)
        return resized1, resized2


class Face:

    def __init__(self, filePath,useFaceChip=None,add=None,border=True,detector: Optional[IFaceDetector] = None):
        if(useFaceChip is not None):
            self.imc1=useFaceChip
        else:
            self.im1 = cv2.imread(filePath, cv2.IMREAD_COLOR)
            self.imc1 = cv2.cvtColor(self.im1, cv2.COLOR_BGR2RGB)
        detector = detector if detector is not None else Detector()
        points = detector.get_landmarks(self.imc1)
        if(add is not None):
            points = np.append(points, add, axis=0)
        self.faceChip = detector.face_chip
        h, w = self.imc1.shape[:2]
        self.h=h
        self.w=w
        if(border):
            border_pts = np.array([
                [0, 0],  # top-left corner
                [w - 1, 0],  # top-right corner
                [w - 1, h - 1],  # bottom-right
                [0, h - 1],  # bottom-left
                [w // 2, 0],  # top edge midpoint
                [w - 1, h // 2],  # right edge midpoint
                [w // 2, h - 1],  # bottom edge midpoint
                [0, h // 2]  # left edge midpoint
            ])

            self.points = np.vstack([points, border_pts])
        else:
            self.points=points
        tri = Delaunay(self.points)
        self.triangle_indices = tri.simplices  # shape (n_triangles, 3)

        # For each triangle, you can access the actual points:
        self.triangle_coords = self.points[self.triangle_indices]
        self._points_raw = self.points.copy()
        # 1. Extract the relevant landmarks
        self.left_eye = self.points[36]
        self.right_eye = self.points[45]
        self.nose = self.points[30]



    def resize_to_fit(self,face2):


        # Find the smaller size
        target_h = min(self.h, face2.h)
        target_w = min(self.w, face2.w)

        # Resize the larger image to match
        if self.h > face2.h or self.w > face2.w:
            img1_resized = cv2.resize(self.imc1, (target_w, target_h))
            img2_resized = face2.imc1  # already smaller
        else:
            img1_resized = self.imc1  # already smaller
            img2_resized = cv2.resize(face2.imc1, (target_w, target_h))
        return img1_resized, img2_resized

    def draw_reset_points(self,points,border=False):
        self.points = points
        if(border):
            w=self.w
            h=self.h
            border_pts = np.array([
                [0, 0],  # top-left corner
                [w - 1, 0],  # top-right corner
                [w - 1, h - 1],  # bottom-right
                [0, h - 1],  # bottom-left
                [w // 2, 0],  # top edge midpoint
                [w - 1, h // 2],  # right edge midpoint
                [w // 2, h - 1],  # bottom edge midpoint
                [0, h // 2]  # left edge midpoint
            ])

            self.points = np.vstack([points, border_pts])
        tri = Delaunay(self.points)
        self.triangle_indices = tri.simplices  # shape (n_triangles, 3)

        # For each triangle, you can access the actual points:
        self.triangle_coords = self.points[self.triangle_indices]
    def drawFace(self,useFaceChip=False):
        if(useFaceChip):
            image=self.faceChip
        else:
            image=self.imc1.copy()
        for tri in self.triangle_indices:
            pts = self.points[tri].astype(np.int32)
            sample= cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 0), thickness=1)  # yellow outline
        return sample
    def maskFace(self):
        mask = np.zeros(self.imc1.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(self.points)  # or just pick face-related indices
        cv2.fillConvexPoly(mask, hull, 255)

    def centerFace(self):
        # 2. Define reference positions in the aligned target frame
        target_left_eye = np.array([self.w * 0.35, self.h * 0.4])
        target_right_eye = np.array([self.w * 0.65, self.h * 0.4])
        target_nose = np.array([self.w * 0.5, self.h * 0.55])

        # 3. Estimate similarity transform from source to target
        src_pts = np.float32([self.left_eye, self.right_eye, self.nose])
        dst_pts = np.float32([target_left_eye, target_right_eye, target_nose])
        M = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

        # 4. Warp the full image
        self.aligned_image = cv2.warpAffine(self.imc1, M, (self.w, self.h))
        return self.aligned_image

    def centerFace2(self):
        # 1. Define target (canonical) landmark positions
        target_left_eye = np.array([self.w * 0.35, self.h * 0.4])
        target_right_eye = np.array([self.w * 0.65, self.h * 0.4])
        target_nose = np.array([self.w * 0.5, self.h * 0.55])
        dst_pts = np.stack([target_left_eye, target_right_eye, target_nose], axis=0)

        # 2. Stack source points
        src_pts = np.stack([self.left_eye, self.right_eye, self.nose], axis=0)

        # 3. Normalize both sets (Procrustes)
        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)

        src_demean = src_pts - src_mean
        dst_demean = dst_pts - dst_mean

        # 4. Compute rotation + scale using SVD
        U, S, Vt = np.linalg.svd(np.dot(dst_demean.T, src_demean))
        R = np.dot(U, Vt)
        scale = np.trace(np.dot(dst_demean.T, src_demean) @ R.T) / np.sum(src_demean ** 2)

        # 5. Compute full transform matrix
        M = np.zeros((2, 3))
        M[:2, :2] = scale * R
        M[:, 2] = dst_mean - scale * R @ src_mean

        # 6. Warp image with this similarity transform
        self.aligned_image = cv2.warpAffine(self.imc1, M, (self.w, self.h), flags=cv2.INTER_LINEAR)

        return self.aligned_image

    def center_face(self,background) -> np.ndarray:
        eye_vector = self.right_eye - self.left_eye
        angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi

        center = (self.w // 2, self.h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(
            self.imc1,
            rot_mat,
            (self.w, self.h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        points_homogeneous = np.hstack(
            [self._points_raw, np.ones((self._points_raw.shape[0], 1))]
        )
        rotated_points = np.dot(rot_mat, points_homogeneous.T).T[:, :2]
        self._points_raw = rotated_points

        x_min, y_min = np.min(rotated_points, axis=0).astype(int) * 0.9
        x_max, y_max = np.max(rotated_points, axis=0).astype(int) * 1.1
        x_mean, y_mean = np.mean(rotated_points, axis=0).astype(int)

        bb_width = x_max - x_min
        bb_height = y_max - y_min
        bb_size = int(max(bb_width, bb_height))
        bb_top_left = (
            max(0, x_mean - bb_size // 2),
            max(0, y_mean - bb_size // 2),
        )

        scale_x = self.w / bb_size
        scale_y = self.h / bb_size
        translation_x = -bb_top_left[0] * scale_x
        translation_y = -bb_top_left[1] * scale_y

        affine_transform = np.array(
            [
                [scale_x, 0, translation_x],
                [0, scale_y, translation_y],
            ]
        )
        aligned_image = cv2.warpAffine(
            aligned_image,
            affine_transform,
            (self.w, self.h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        points_homogeneous = np.hstack(
            [self._points_raw, np.ones((self._points_raw.shape[0], 1))]
        )
        transformed_points = np.dot(affine_transform, points_homogeneous.T).T[:, :2]
        self.points = transformed_points

        self.im1 = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR)
        self.imc1 = aligned_image.copy()
        self.aligned_image = self.imc1
        self.draw_reset_points(self.points,background)
        return aligned_image




class Detector(IFaceDetector):
    # Load model and detector
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, img, useFaceChip=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            raise Exception("No face detected")

        shape = self.predictor(gray, faces[0])
        # Align the face using Dlib (aligns based on eyes and nose)
        self.face_chip = dlib.get_face_chip(
            img, shape
        )  # since Dlib isnt used anymore this 'align feature' is void. it also resizes every pic
        # cv2.imshow("Aligned Face", face_chip)

        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks  # shape: (68, 2)
