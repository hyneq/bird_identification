import os,sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.remove(os.path.dirname(__file__))

from detection.detector import object_detector_factory

from cli.prediction import PredictionCLI

def cli_main():
    PredictionCLI(object_detector_factory).run()

"""
def cli_main():
    results = detect_objects(sys.argv[1])

    for result in results[0]:
        print("{} at {} with {} % confidence".format(result.label, result.bounding_box, np.round(result.confidence*100,2)))
"""

if __name__ == "__main__":
    cli_main()