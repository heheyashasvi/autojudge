#!/usr/bin/env python3
"""
Dataset upload and processing script for AutoJudge.
"""
import argparse
import sys
from pathlib import Path
import json

from ml.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(description="Upload and process dataset for AutoJudge")
    parser.add_argument("dataset_path", help="Path to your JSONL dataset file")
    parser.add_argument("--output", "-o", help="Output path for processed dataset", default="data/processed_dataset.jsonl")
    parser.add_argument("--preview", "-p", action="store_true", help="Preview dataset schema and samples")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of samples to show in preview")
    
    args = parser.parse_args()
    
    # Check if dataset file exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    print(f"📁 Loading dataset from: {dataset_path}")
    
    try:
        # Initialize loader
        loader = DatasetLoader(str(dataset_path))
        
        # Load raw data
        raw_data = loader.load_jsonl()
        print(f"✅ Successfully loaded {len(raw_data)} records")
        
        # Detect schema
        schema = loader.detect_schema()
        print(f"\n🔍 Detected Schema:")
        for expected, detected in schema.items():
            print(f"  {expected} -> {detected}")
        
        if args.preview:
            print(f"\n👀 Preview of first {args.sample_size} records:")
            for i, record in enumerate(raw_data[:args.sample_size]):
                print(f"\n--- Record {i+1} ---")
                for key, value in record.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"  {key}: {value}")
            
            # Ask user if they want to continue
            response = input(f"\n❓ Does the schema look correct? Continue processing? (y/N): ")
            if response.lower() != 'y':
                print("❌ Processing cancelled")
                sys.exit(0)
        
        # Process dataset
        print(f"\n⚙️  Processing dataset...")
        problems, labels = loader.process_dataset(schema)
        
        print(f"✅ Processed {len(problems)} valid problems")
        
        # Show statistics
        import pandas as pd
        class_dist = pd.Series(labels['class']).value_counts()
        print(f"\n📊 Class Distribution:")
        for class_name, count in class_dist.items():
            percentage = (count / len(labels['class'])) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        score_stats = pd.Series(labels['score']).describe()
        print(f"\n📈 Score Statistics:")
        print(f"  Mean: {score_stats['mean']:.2f}")
        print(f"  Std:  {score_stats['std']:.2f}")
        print(f"  Min:  {score_stats['min']:.2f}")
        print(f"  Max:  {score_stats['max']:.2f}")
        
        # Save processed data
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        loader.save_processed_data(str(output_path))
        print(f"💾 Saved processed dataset to: {output_path}")
        
        # Create train/test split
        train_problems, test_problems, train_labels, test_labels = loader.get_train_test_split()
        
        print(f"\n🔄 Train/Test Split:")
        print(f"  Training: {len(train_problems)} problems")
        print(f"  Testing:  {len(test_problems)} problems")
        
        # Save split data
        train_path = output_path.parent / "train_dataset.jsonl"
        test_path = output_path.parent / "test_dataset.jsonl"
        
        # Save training data
        with open(train_path, 'w', encoding='utf-8') as f:
            for i, problem in enumerate(train_problems):
                record = {
                    'title': problem.title,
                    'description': problem.description,
                    'input_description': problem.input_description,
                    'output_description': problem.output_description,
                    'problem_class': train_labels['class'][i],
                    'problem_score': train_labels['score'][i]
                }
                f.write(json.dumps(record) + '\n')
        
        # Save testing data
        with open(test_path, 'w', encoding='utf-8') as f:
            for i, problem in enumerate(test_problems):
                record = {
                    'title': problem.title,
                    'description': problem.description,
                    'input_description': problem.input_description,
                    'output_description': problem.output_description,
                    'problem_class': test_labels['class'][i],
                    'problem_score': test_labels['score'][i]
                }
                f.write(json.dumps(record) + '\n')
        
        print(f"💾 Saved training data to: {train_path}")
        print(f"💾 Saved testing data to: {test_path}")
        
        print(f"\n🎉 Dataset processing complete!")
        print(f"\n📋 Next steps:")
        print(f"  1. Review the processed data in: {output_path}")
        print(f"  2. Train models using: python train_models.py")
        print(f"  3. Test the API with your custom dataset!")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()