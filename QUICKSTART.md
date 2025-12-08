# Quick Start Guide

## Processing Your Data

### Step 1: Process a Batch of Files

1. **Put your data in a folder**
   - Place all your `.nrrd` files in a folder (e.g., `data/my_batch`)

2. **Set your input/output folders**
   - Open `process_config.yaml`
   - Change these two lines to match your folders:
     ```yaml
     input_folder: "data/my_batch"
     output_folder: "results/my_batch"
     ```

3. **Run the processing**
   ```bash
   python run_processing_pipeline.py
   ```

The pipeline will:
- Process each `.nrrd` file
- Ask if you want to skip files you've already processed
- Save results as `.pkl` files in your output folder
- Ask if you want to merge everything into one compressed file

**Tip:** If you have lots of files, say "yes" when it asks to merge - the pkl files take up a ridiculous amount of space so after processing you can just keep the merged file and delete the pkls

---

### Step 2: Analyze Your Results

1. **Choose your input**

   Open `analysis_config.yaml` and pick **one** option:

   **Option A: Analyze individual files**
   ```yaml
   input_folder: "results/my_batch"
   # input_merged_file: "results/my_batch/my_batch_merged.pkl.gz"
   ```

   **Option B: Analyze merged file** (if you merged in Step 1)
   ```yaml
   # input_folder: "results/my_batch"
   input_merged_file: "results/my_batch/my_batch_merged.pkl.gz"
   ```

2. **Run the analysis**
   ```bash
   python run_stats_pipeline.py
   ```

This currently will print statistics for each artery (lengths, diameters, angles, etc.) but you can put your actual analysis code here.


