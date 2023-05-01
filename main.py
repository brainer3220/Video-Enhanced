import os
from glob import glob


def model_download():
    os.system("""python SCUNet/main_download_pretrained_models.py --models "SCUNet" --model_dir "model_zoo""")


def get_video_file(path):
    return glob(path + "/*.mp4")


def get_directory(path):
    return glob(path + "/*")


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


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
        print(abs_dir_path)
        os.chdir("SCUNet")
        os.system(
            "python main_test_scunet_real_application.py --model_name scunet_color_real_psnr --testset_name " + "/Users/brainer/Programming/video-super-res/low_quality" + " --testsets " + "new" + " --results /Users/brainer/Programming/video-super-res/enhanced")
        os.chdir("..")


if __name__ == "__main__":
    main()
