from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import load_faster_rcnn


OUTPUT_PATH = ROOT / "models" / "fasterrcnn-warehouse-v3.onnx"


class FasterRcnnOnnxWrapper(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, images):
        prediction = self.detector([images[0]])[0]
        return prediction["boxes"], prediction["scores"], prediction["labels"]


def main():
    detector = load_faster_rcnn().cpu().eval()
    wrapper = FasterRcnnOnnxWrapper(detector).eval()
    sample = torch.rand(1, 3, 640, 640)

    torch.onnx.export(
        wrapper,
        sample,
        OUTPUT_PATH,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        opset_version=17,
        dynamo=False,
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
