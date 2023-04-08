import os,sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.remove(os.path.dirname(__file__))

from classification.classifier import image_classifier_factory

from cli.prediction import PredictionCLI

def cli_main():
    PredictionCLI(image_classifier_factory).run()

if __name__ == "__main__":
    cli_main()