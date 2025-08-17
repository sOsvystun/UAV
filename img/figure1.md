+-----------+     +----------------------+     +---------------------------+     +----------------------------+     +-------------------+
|  Start →  | --> |  Data Acquisition    | --> | Alignment & Calibration   | --> | Multispectral Image Fusion | --> | Training Data Split|
| UAV plan  |     |  (RGB + Thermal IR) |     |  (Homography + Tuning)    |     |       (RGB ⊕ IR)           |     |    {Decision ♦}    |
+-----------+     +----------------------+     +---------------------------+     +----------------------------+     +---------♦---------+
                                                                                                                             / \
                                                                                                                            /   \
                                                                                                                        Copy‑A  Copy‑B
                                                                                                                          v       v
                                                                                   +--------------------------+    +---------------------------+
                                                                                   | YOLOv8 Baseline Model    |    | Thermal‑Focused YOLO     |
                                                                                   |  (Fused / RGB branch)    |    |  (IR‑heavy training)     |
                                                                                   +--------------------------+    +---------------------------+
                                                                                            \                           /
                                                                                             \                         /
                                                                                              v                       v
                                                                                       +----------------------------------------------------+
                                                                                       | Bounding‑Box Ensemble Fusion (Weighted IoU WBF)    |
                                                                                       +----------------------------------------------------+
                                                                                                             |
                                                                                                      +--------------+
                                                                                                      |     End      |
                                                                                                      | Robust Output|
                                                                                                      +--------------+
