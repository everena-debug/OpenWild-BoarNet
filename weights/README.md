# Pretrained YOLOv8n weights

Place the three trained wild-boar detectors here (about 6 MB each):

| File | Seed | Used for |
|---|---|---|
| `yolov8n_wildboar_seed0_best.pt` | 0 | Detection results and region-of-interest gating of the open-set pipeline |
| `yolov8n_wildboar_seed1_best.pt` | 1 | Detection results (seed variability) |
| `yolov8n_wildboar_seed2_best.pt` | 2 | Detection results (seed variability) |

Each file is the checkpoint with the best validation fitness (0.1 × mAP@50 + 0.9 × mAP@50–95) of its training run.

Load one with:

```python
from ultralytics import YOLO
model = YOLO("weights/yolov8n_wildboar_seed0_best.pt")
results = model.predict("image.jpg", conf=0.25)
```
