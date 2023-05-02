import os

from utility.utility import get_video_file, make_directory, get_directory


def model_download():
    os.system("""python SCUNet/main_download_pretrained_models.py --models "SCUNet" --model_dir "SCUNet/model_zoo""")


def main():
    model_download()
    video_list = get_video_file("./low_quality")
    for video in video_list:
        make_directory("./low_quality/" + video.split("/")[-1].split(".")[0])
        # os.system(
        #     "ffmpeg -i " + video + " -vf fps=30 -f image2 ./low_quality/" + video.split("/")[-1].split(".")[
        #         0] + "/%07d.png")
        #
        # os.system("python bsrgan/main_test_bsrgan.py")

    for dir in get_directory("./low_quality"):
        abs_dir_path = os.path.abspath(dir)
        os.system("python ddpm/main.py --image-dir " + abs_dir_path)


if __name__ == "__main__":
    main()
