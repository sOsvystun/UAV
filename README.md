# UAV Wind-Turbine Inspection Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for a cyber-physical system described in the dissertation:
> Svystun S.O. Methods and means of dynamic collection of visual data on defects of wind energy objects. — Khmelnytskyi, 2025.

The system provides a fully autonomous cycle of **"take-off → scanning → analysis → report"**, reducing the inspection time of one turbine from 4 hours to ≈25 minutes with an average defect detection accuracy of 92%.

🔗 Download the full text of the dissertation: [Svystun_Dissertation](https://nauka.khmnu.edu.ua/speczialnist-123-kompyuterna-inzheneriya-avtor-dysertacziyi-svystun-sergij-olegovych/)

---

## 1. Quick Start

Ensure you have the prerequisites installed (see section 4.1). Then, from the project root, run the unified build script:

```powershell
./build.ps1
```

This will build all the necessary components. After a successful build, you can run the server and the controller application as described in the "Usage" section.

---

## 2. Architecture

The architecture of the cyber-physical system consists of three interconnected components:

| Subsystem | Language | Purpose | Dissertation Reference |
|---|---|---|---|
| **VISION_Recognition** | C++20 / Python | Multispectral (RGB + IR) image processing, YOLOv8 + Cascade R-CNN ensemble | Chapter 3, § 3.3–3.4 |
| **UAV_server** | Rust 2024 | gRPC bus, Kubernetes scaling, streaming analysis | Chapter 2, § 2.1 |
| **UAV_Controller** | .NET 8 (MAUI) | Operator UI, telemetry, DyTAM mission planning | Chapter 2, § 2.2–2.3 |
| **VISION_Fuzzy** | Python | Fuzzy logic for defect criticality assessment | Chapter 3, § 3.5 |

---

## 3. Technical Details & Implementation

This section provides an in-depth look at each core component of the UAV Wind-Turbine Inspection Suite, detailing their functionalities, underlying technologies, and performance metrics.

### 3.1. VISION_Recognition: Advanced Defect Detection

The `VISION_Recognition` subsystem is responsible for multispectral image processing and defect detection on wind turbine blades. It leverages a sophisticated approach that integrates thermal and RGB images to enhance defect detection efficiency.

**Multispectral Image Composition:**
This method combines thermal and RGB imagery through a process involving spatial coordinate transformation, key point detection, binary descriptor creation, and weighted image overlay. This composition significantly improves defect detection accuracy.

**Ensemble Detector:**
The system utilizes an ensemble of YOLOv8 and Cascade R-CNN models for object detection. This ensemble achieves a high detection accuracy, particularly on the custom **Blade30-Thermal Dataset**.

| Metric | Original YOLOv8 | Composite Image YOLOv8 |
|---|---|---|
| Accuracy | 91% | 95% |
| Precision | 89% | 94% |
| Recall | 85% | 92% |
| F1-score | 87% | 93% |

*Data adapted from: Svystun, S., Melnychenko, O., Radiuk, P., Savenko, O., Sachenko, A., & Lysyi, A. (2024). Thermal and RGB Images Work Better Together in Wind Turbine Damage Detection. International Journal of Computing, 23(4), 526–535. [https://doi.org/10.47839/ijc.23.4.3752](https://doi.org/10.47839/ijc.23.4.3752)*

### 3.2. UAV_Controller: Dynamic Trajectory Adaptation (DyTAM)

The `UAV_Controller` subsystem implements the Dynamic Trajectory Adaptation Method (DyTAM), a novel approach for automated UAV-based wind turbine inspections. DyTAM dynamically adjusts UAV flight paths based on real-time visual data, optimizing inspection efficiency and data quality.

**Key DyTAM Features:**

* **Real-time Component Segmentation:** Identifies key turbine components (blades, tower, nacelle) from the UAV's initial viewpoint.
* **Blade Pitch Angle Classification:** Computes and classifies blade pitch angles into acute, vertical, and horizontal tilts, which dictate the selection of appropriate trajectory models.

* **Specialized Trajectory Models:** Employs optimized parameterized paths:

    * **Spiral Paths:** For vertical blade tilts, orbiting the blade while reducing radius.
    * **Helical Paths:** For horizontally tilted blades.
    * **Offset-Line Paths:** For acutely tilted blades.
    * Dedicated spiral paths are used for ascending the tower, and specialized trajectories are designed for the nacelle's lateral, rear, and top planes.

* **Wind Compensation:** Incorporates real-time wind compensation to maintain precise flight paths and consistent standoff distances even under challenging wind conditions (up to 15 m/s).

**Performance Improvements:**
DyTAM significantly enhances inspection performance compared to manual control and other state-of-the-art methods:

| Metric | Manual Control (Windy) | Automated Control (DyTAM) | Improvement |
|---|---|---|---|
| Inspection Time (min) | 22.35 (horizontal blade) | 8.46 (horizontal blade) | **78% reduction** (average across components) |
| Flight Path Length | - | - | **17% decrease** |
| Blade Coverage | - | - | **6% increase** |
| Mean Deviation (m) | 3.57 (horizontal blade, nominal) | 0.68 (horizontal blade, nominal) | **68% reduction** (average across components) |

*Data adapted from: Svystun, S., Scislo, L., Pawlik, M., Melnychenko, O., Radiuk, P., Savenko, O., & Sachenko, A. (2025). DyTAM: Accelerating Wind Turbine Inspections with Dynamic UAV Trajectory Adaptation. Energies, 18(7), 1823. [https://doi.org/10.3390/en18071823](https://doi.org/10.3390/en18071823)*

### 3.3. UAV_server: Backend & Communication Hub

The `UAV_server` component, built with Rust, acts as the central communication bus for the entire system. It handles real-time data streaming and serves as a scalable backend for analysis services.

* **gRPC Bus:** Facilitates high-performance, inter-service communication within the system.
* **Kubernetes Scaling:** Designed for scalable deployment, allowing for efficient management of computational resources for data processing and analysis.
* **Streaming Analysis:** Enables real-time processing of incoming data streams from the UAV, crucial for dynamic adaptation and immediate report generation.

### 3.4. VISION_Fuzzy: Defect Criticality Assessment

The `VISION_Fuzzy` subsystem implements a Fuzzy Inference System (FIS) for the autonomous assessment of wind turbine defect criticality. It leverages fuzzy logic to evaluate defect severity based on three key input parameters:

* **Defect Size:** The physical dimensions of the detected anomaly (in pixels).
* **Location:** The position of the defect on the blade, normalized from the root (0.0) to the tip (1.0).
* **Thermal Signature:** The temperature difference (in Celsius) indicating thermal anomalies.

The FIS employs 27 meticulously defined fuzzy rules to map these inputs to a `Criticality` score ranging from 0 to 5. This score helps in prioritizing maintenance efforts. The system uses `numpy` for numerical operations and `scikit-fuzzy` for fuzzy logic computations.

---

## 4. Installation & Setup

This project comprises several components, each with its own dependencies.

### 4.1. Prerequisites

Ensure you have the following installed:

* **Python 3.10+**: For `VISION_Fuzzy` and parts of `VISION_Recognition`.

    ```bash
    # Install Python dependencies
    pip install -r requirements.txt
    ```

* **C++ Compiler (C++20 compatible)**: GCC/Clang (Linux/macOS) or Visual Studio (Windows).
* **CMake (3.30+)**: For building `VISION_Recognition`.
* **OpenCV (4.9+)**: Required by `VISION_Recognition` (ensure bindings for C++ are available).
* **Rust (2024 edition)**: For `UAV_server`. Install via `rustup`.
* **.NET 8 SDK**: For `UAV_Controller`.
* **Visual Studio 2022 (17.9+) / VS Code with MAUI extensions**: For developing `UAV_Controller`.

### 4.2. Building the Project

To build all components of the suite, run the provided PowerShell script from the root directory. This script will handle the build process for each sub-project automatically.

```powershell
./build.ps1
```

The script will:
1.  Build the Computer Vision Module (`VISION_Recognition`).
2.  Build the Server Bus (`UAV_server`).
3.  Build the UI Controller (`UAV_Controller`).

---

## 5. Usage

1. **Run** `uav_server` in a Kubernetes cluster or locally.
    ```bash
    ./UAV_server/target/release/uav_server
    ```
2. **Connect** the ground station to the drone (PX4 / ArduPilot).
3. **Open** `UAV_Controller`, select "New Mission → DyTAM".
4. **Press** "Start". After landing, a PDF report will be automatically generated in `~/UAV_reports/`.

---

## 6. Contribution and Reproducibility

* The experimental stand, config files, and statistical processing scripts (`/experiments/`) reproduce the results of Chapter 4 of the dissertation.
* The Blade30-Thermal dataset is available upon request in [issues](https://github.com/sOsvystun/UAV/issues).
* The CI workflow `./github/workflows/test.yml` runs unit tests and checks metric reproducibility.

---

## 7. Publications and Citation

If you use the software code or any ideas from the dissertation, please cite one of the scientific publications related to this project:

* **Thermal and RGB Images Work Better Together in Wind Turbine Damage Detection**

    ```bib
    @article{Svystun2024IJC,
      author       = {Svystun, Serhii and Melnychenko, Oleksandr and Radiuk, Pavlo and Savenko, Oleg and Sachenko, Anatoliy and Lysyi, Andrii},
      title        = {Thermal and RGB Images Work Better Together in Wind Turbine Damage Detection},
      journal      = {International Journal of Computing},
      year         = {2024},
      volume       = {23},
      number       = {4},
      pages        = {526--535},
      doi          = {10.47839/ijc.23.4.3752},
      url          = {https://computingonline.net/computing/article/view/3752},
      note         = {Indexed in Scopus (Q3, SJR)}
    }
    ```

* **DyTAM: Accelerating Wind Turbine Inspections with Dynamic UAV Trajectory Adaptation**

    ```bib
    @article{Svystun2025Energies,
      author       = {Svystun, Serhii and Ścisło, Łukasz and Pawlik, Marcin and Melnychenko, Oleksandr and Radiuk, Pavlo and Savenko, Oleg and Sachenko, Anatoliy},
      title        = {DyTAM: Accelerating Wind Turbine Inspections with Dynamic UAV Trajectory Adaptation},
      journal      = {Energies},
      year         = {2025},
      volume       = {18},
      number       = {7},
      pages        = {1823},
      doi          = {10.3390/en18071823},
      url          = {https://www.mdpi.com/1996-1073/18/7/1823},
      note         = {Indexed in Scopus (Q1, SJR)}
    }
    ```

* **Criticality Assessment of Wind Turbine Defects via Multispectral UAV Fusion and Fuzzy Logic**

    ```bib
    @article{Radiuk2025CEUR,
      author       = {Radiuk, Pavlo and Svystun, Serhii and Melnychenko, Oleksandr and Sachenko, Anatoliy},
      title        = {Determining the criticality assessment of defects on wind turbine components detected by UAV sensors},
      journal      = {CEUR Workshop Proceedings},
      year         = {2025},
      volume       = {3963},
      pages        = {28},
      url          = {https://ceur-ws.org/Vol-3963/paper28.pdf}
    }
    ```

---

## 8. License

The code is distributed under the **MIT** license. Blade30-Thermal data — **CC BY-NC 4.0**.

---

> © 2024-2025 Serhii Svystun & Contributors
> Khmelnytskyi National University, Department of Computer Engineering and Information Systems
