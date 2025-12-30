# Dataset Directory

This directory contains your programming problem datasets for AutoJudge.

## Uploading Your Dataset

### Step 1: Place Your JSONL File
Copy your JSONL dataset file to this directory or any location on your system.

### Step 2: Expected Format
Your JSONL file should contain one JSON object per line. AutoJudge can automatically detect various field names, but here are the expected fields:

**Required Fields:**
- `title` (or `name`, `problem_name`, `problem_title`)
- `description` (or `problem_description`, `statement`, `problem_statement`, `text`)
- `input_description` (or `input_format`, `input`, `input_spec`)
- `output_description` (or `output_format`, `output`, `output_spec`)

**Optional Fields:**
- `problem_class` (or `difficulty`, `difficulty_class`, `level`, `class`)
  - Values: "Easy"/"Medium"/"Hard" or 1/2/3 or numeric ratings
- `problem_score` (or `difficulty_score`, `score`, `rating`, `difficulty_rating`)
  - Numeric value (will be normalized to 0-10 range)

### Example JSONL Format:
```json
{"title": "Two Sum", "description": "Given an array...", "input_description": "Array and target", "output_description": "Two indices", "problem_class": "Easy", "problem_score": 2.5}
{"title": "Merge Sort", "description": "Sort an array...", "input_description": "Unsorted array", "output_description": "Sorted array", "problem_class": "Medium", "problem_score": 5.0}
```

### Step 3: Upload and Process
Run the upload script from the backend directory:

```bash
# Preview your dataset first (recommended)
python upload_dataset.py path/to/your/dataset.jsonl --preview

# Process the dataset
python upload_dataset.py path/to/your/dataset.jsonl

# Or specify custom output location
python upload_dataset.py path/to/your/dataset.jsonl --output data/my_dataset.jsonl
```

### Step 4: What Happens Next
The script will:
1. 🔍 Detect your dataset schema automatically
2. 📊 Show statistics about your data
3. 🔄 Create train/test splits
4. 💾 Save processed data in standardized format
5. ✅ Prepare data for model training

## Supported Dataset Formats

AutoJudge can handle datasets from various sources:

### Competitive Programming Platforms:
- **Codeforces**: Difficulty ratings (800-3500) → normalized to 0-10
- **LeetCode**: Easy/Medium/Hard → mapped to classes
- **AtCoder**: Color ratings → mapped to difficulty
- **Custom**: Any numeric or categorical difficulty system

### Academic Datasets:
- Programming contest problems
- Algorithm textbook problems
- Educational coding challenges

## File Structure After Processing:
```
data/
├── your_original_dataset.jsonl
├── processed_dataset.jsonl      # Standardized format
├── train_dataset.jsonl          # Training split (80%)
├── test_dataset.jsonl           # Testing split (20%)
└── README.md                    # This file
```

## Troubleshooting

### Common Issues:
1. **Missing Fields**: The script will tell you which fields are missing
2. **Invalid JSON**: Check that each line is valid JSON
3. **Encoding Issues**: Ensure your file is UTF-8 encoded
4. **Empty Records**: Records with no meaningful content will be skipped

### Getting Help:
Run with `--preview` flag to see how your data will be interpreted before processing.

## Next Steps
After uploading your dataset:
1. Train the ML models: `python train_models.py`
2. Test the API with your custom data
3. See your own problems analyzed by AutoJudge!