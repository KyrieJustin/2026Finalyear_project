import json
import argparse
from collections import defaultdict

def extract_answer_from_output(model_output):
    """
    Extract multiple-choice answer (A/B/C/D) from model output text
    Processing logic:
    1. Prioritize matching lines starting with A/B/C/D (e.g., "B. Dendrocygna")
    2. If not found, match standalone A/B/C/D characters in the text
    3. If still not found, return empty string
    """
    if not model_output:
        return ""
    
    # Convert to lowercase for matching while preserving original characters for extraction
    output_lower = model_output.lower()
    output_original = model_output.strip()
    
    # Rule 1: Match lines starting with A/B/C/D + dot/space (most common format)
    for char in ['A', 'B', 'C', 'D']:
        if f"{char}." in output_original or f"{char} " in output_original:
            # Extract the first occurrence of A/B/C/D
            if f"{char}." in output_original:
                return char
            elif f"{char} " in output_original and output_original.index(f"{char} ") == 0:
                return char
    
    # Rule 2: Match standalone A/B/C/D characters (no extra content)
    for char in ['A', 'B', 'C', 'D']:
        if char in output_original and len(char) == 1:
            return char
    
    # Rule 3: Match lowercase a/b/c/d and convert to uppercase
    for char in ['a', 'b', 'c', 'd']:
        if char in output_lower:
            return char.upper()
    
    # Return empty if no answer is matched
    return ""

def evaluate_json(file_path):
    """
    Core evaluation function:
    - Read JSON file
    - Extract and compare answers
    - Calculate overall/category-wise accuracy
    """
    # Initialize statistical variables
    total_count = 0
    correct_count = 0
    # Statistics by question category
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    # Record error cases (for analysis)
    error_cases = []

    # Read JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}!")
        return None

    # Iterate through each QA sample
    for idx, item in enumerate(data):
        total_count += 1
        # Get ground truth answer and model output
        ground_truth = item.get('Answer', '').strip()
        model_output = item.get('model_output', '').strip()
        category = item.get('Category', 'Unknown')
        question = item.get('Question', '')
        image_path = item.get('image_path', '')

        # Extract answer from model output
        pred_answer = extract_answer_from_output(model_output)
        # Compare answers (case-insensitive)
        is_correct = pred_answer.upper() == ground_truth.upper()

        # Update statistics
        if is_correct:
            correct_count += 1
            category_stats[category]["correct"] += 1
        else:
            # Record error cases
            error_cases.append({
                "index": idx,
                "image_path": image_path,
                "question": question,
                "ground_truth": ground_truth,
                "pred_answer": pred_answer,
                "model_output": model_output[:100] + "..." if len(model_output) > 100 else model_output
            })
        category_stats[category]["total"] += 1

    # Calculate overall accuracy
    overall_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0

    # Generate evaluation report
    report = {
        "overall": {
            "total_samples": total_count,
            "correct_samples": correct_count,
            "accuracy": round(overall_accuracy, 2)
        },
        "by_category": {},
        "error_cases": error_cases[:10]  # Keep only top 10 error cases (avoid long output)
    }

    # Calculate category-wise accuracy
    for cat, stats in category_stats.items():
        cat_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        report["by_category"][cat] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(cat_accuracy, 2)
        }

    return report

def print_evaluation_report(report, file_path):
    """
    Print evaluation report in formatted style
    """
    print("="*50)
    print(f"Evaluation Report for: {file_path}")
    print("="*50)
    # Overall results
    print("\n[Overall Performance]")
    print(f"Total Samples: {report['overall']['total_samples']}")
    print(f"Correct Samples: {report['overall']['correct_samples']}")
    print(f"Overall Accuracy: {report['overall']['accuracy']}%")

    # Category-wise results
    print("\n[Performance by Category]")
    for cat, stats in report["by_category"].items():
        print(f"- {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")

    # Error cases (optional)
    if report["error_cases"]:
        print("\n[Top 10 Error Cases (Sample)]")
        for case in report["error_cases"]:
            print(f"\nIndex: {case['index']}")
            print(f"Image: {case['image_path'].split('/')[-1]}")
            print(f"Question: {case['question'][:50]}...")
            print(f"Ground Truth: {case['ground_truth']} | Predicted: {case['pred_answer']}")
            print(f"Model Output: {case['model_output']}")
    print("="*50)

if __name__ == "__main__":
    # Support command-line argument for file path
    parser = argparse.ArgumentParser(description="Evaluate model output JSON file")
    parser.add_argument("--file", "-f", default="result_ablation_vision.json", 
                        help="Path to the result JSON file (default: result_simple_agent.json)")
    args = parser.parse_args()

    # Execute evaluation and print report
    eval_report = evaluate_json(args.file)
    if eval_report:
        print_evaluation_report(eval_report, args.file)