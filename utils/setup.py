import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-t", "--topic", default="ERC")

args = vars(parser.parse_args())

if __name__ == "__main__":
    topic_name = args['topic']
    print(topic_name)

    from google.colab import drive

    drive.mount('/content/gdrive')

    MODEL_DIR = '/content/gdrive/MyDrive/model'
    DATA_DIR = os.path.join('/content/gdrive/MyDrive/dataset', topic_name)
    ERC_DIR = os.path.join('/content/gdrive/MyDrive/Research', topic_name)

    os.environ['MODEL_DIR'] = MODEL_DIR
    os.environ['DATA_DIR'] = DATA_DIR
    os.environ['RESEARCH_DIR'] = ERC_DIR
    sys.path.append('/content/SpeechAnalytics')
