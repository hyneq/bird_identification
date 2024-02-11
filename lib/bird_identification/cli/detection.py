import os,sys
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), os.path.pardir, os.path.pardir))
sys.path.remove(os.path.dirname(__file__))

from bird_identification.detection.detector import object_detector_factory

from bird_identification.cli.prediction import FactoryImagePredictionCLI

def cli_main():
    FactoryImagePredictionCLI(object_detector_factory).run()

if __name__ == "__main__":
    cli_main()