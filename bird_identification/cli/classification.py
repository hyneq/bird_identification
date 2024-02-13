import os,sys
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), os.path.pardir, os.path.pardir))
sys.path.remove(os.path.dirname(__file__))

from bird_identification.classification.classifier import image_classifier_factory

from bird_identification.cli.prediction import FactoryImagePredictionCLI

def cli_main():
    FactoryImagePredictionCLI(image_classifier_factory).run()

if __name__ == "__main__":
    cli_main()