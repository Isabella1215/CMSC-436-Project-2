# Project 2 – Perceptron Classifier

## Overview
This project implements a perceptron-based classifier to distinguish between two classes of cars (small vs. big) using datasets A, B, and C.  
We train the perceptron with **hard unipolar** and **soft unipolar** activation functions, test under different training/testing splits, plot decision boundaries, and generate confusion matrices.

The written report is in Quarto (`report.qmd`) and renders to PDF (`Bounite_Finocchio_Darko_Alexander_Project2.pdf`).

---

## Repo Structure
- **/data/** – Original datasets (A.csv, B.csv, C.csv) + train/test splits  
- **/src/** – `perceptron.py` (model + training), `run_experiments.py` (main runner)  
- **/outputs/** – Generated plots, confusion matrices, metrics.json  
- **report.qmd** – Quarto report (renders to PDF)  
- **requirements.txt** – Python dependencies  

---

## Setup

### Clone the repository
```bash
git clone https://github.com/Isabella1215/CMSC-436-Project-2.git
cd CMSC-436-Project-2
```
### Install Dependencies
- Ensure Python 3.10+ is installed.
- (Optional) create and activate a virtual environment:
  ```bash
  python -m venv .venv
  # macOS/Linux
  source .venv/bin/activate
  # Windows (PowerShell)
  .venv\Scripts\Activate.ps1
  ```
- install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Code
- Run all experiments (datasets A, B, C; hard & soft; 75/25 and 25/75):
  ```bash
  python src/run_experiments.py --alpha 0.05 --gain 1.0 --max-iters 5000
  ```


### Rendering the report
- Rendering the final quarto report to a PDF:
  ```bash
  quarto render report.qmd --to pdf
```