import argparse
import progressbar
import cv2
from subprocess import call
import glob
import face_recognition
import os


def create_folder_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def main():
    # Parse paths
    parser = argparse.ArgumentParser()
    parser.add_argument("-video", help="Video path.")
    parser.add_argument(
        "-output", default="./output", help="Output folder path.")
    parser.add_argument("-d", required=False, help="Debug mode.")
    parser.add_argument("-f", required=False, help="Fixed face size.")
    parser.add_argument("-i", required=False, type=int, help="Face size increased (int).")
    args = parser.parse_args()

    create_folder_if_not_exist(args.output)

    # Prepare frames
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    call(["ffmpeg", "-i", args.video, "-vf", "fps={}".format(fps), "-qscale:v", "2", '{}/imagename%05d.jpg'.format(args.output)])

    # Take faces
    face_size = 500
    min_face_size = 200
    threshold = 20
    processed_image_name = "auron_{}.jpg"

    with progressbar.ProgressBar(max_value=len(glob.glob("{}/*.jpg".format(args.output)))) as bar:
        pos = 1
        for idx, image_path in enumerate(glob.glob("{}/*.jpg".format(args.output))):
            frame = cv2.imread(image_path)
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)

            for top, right, bottom, left in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                face_cropped = frame[top:bottom, left:right]

                # Fake faces
                if bottom - top > min_face_size:

                    if args.f:
                        cy = face_size - (bottom - top)
                        cx = face_size - (right - left)

                        if top - (cy // 2) <= 0:
                            top = 0
                            bottom = face_size
                        else:
                            top = top - (cy // 2)
                            bottom = top + face_size

                        if left - (cx // 2) <= 0:
                            left = 0
                            right = face_size
                        else:
                            left = left - (cx // 2)
                            right = left + face_size

                    if args.i:
                        top -= int(args.i)
                        bottom += int(args.i)
                        left -= int(args.i)
                        right = left + (bottom - top)

                    crop_img = frame[top:bottom, left:right]

                    # Ignore blurry faces
                    gray = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2GRAY)
                    fm = variance_of_laplacian(gray)
                    if fm < threshold:
                        continue

                    # Draw a box around the face
                    if args.d:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv2.imwrite(args.output + "/" + processed_image_name.format(pos), crop_img)
                    pos += 1

                    if args.d:
                        print(bottom - top, right - left)

            # Display the resulting image
            if args.d:
                cv2.imshow('Video', frame)
                cv2.waitKey(0)
            os.remove(image_path)
            bar.update(idx)


if __name__ == '__main__':

    main()
