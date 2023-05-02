from DDPMDenosing import DDPMDenosing
import argparse


def main(image_path):
    model = DDPMDenosing()
    model.denosing(image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, help='path of image directory')

    args = parser.parse_args()
    main(args.image_dir)
