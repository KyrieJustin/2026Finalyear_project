import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FILES = {
    "Baseline": "result_baseline.json",
    "Knowledge-Only": "result_ablation_knowledge.json",
    "Vision-Only": "result_ablation_vision.json",
    "Full Agent (Proposed)": "result_real_agent.json"
}
OUTPUT_DIR = "visualizations"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def calculate_stats(file_path):

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {}
    total_correct = 0
    categories = ["Object", "Count", "Attribute", "Action"]
    
    for cat in categories:
        cat_items = [item for item in data if item.get("Category") == cat]
        if not cat_items: continue
        
        # ͳ�ƶԴ� (�߼�: ģ������а�����ȷѡ��)
        correct = 0
        for item in cat_items:
            gt = item["Answer"].upper()
            pred = item.get("model_output", "").upper()
            if f"{gt}." in pred or f"({gt})" in pred or pred.strip() == gt:
                correct += 1
        
        stats[cat] = (correct / len(cat_items)) * 100
        total_correct += correct
        
    stats["Overall"] = (total_correct / len(data)) * 100
    return stats

def main():
    # 1. �ռ�����
    all_data = []
    for name, path in FILES.items():
        res = calculate_stats(path)
        if res:
            for cat, acc in res.items():
                all_data.append({"Model": name, "Category": cat, "Accuracy (%)": acc})
    
    df = pd.DataFrame(all_data)
    
    # --- ͼ�� 1: Overall Accuracy (����ʵ���ܶԱ�) ---
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    overall_df = df[df["Category"] == "Overall"]
    
    # ����׼ȷ������
    overall_df = overall_df.sort_values(by="Accuracy (%)")
    
    ax = sns.barplot(x="Model", y="Accuracy (%)", data=overall_df, palette="viridis")
    plt.title("Overall Accuracy Comparison (Ablation Study)", fontsize=16, pad=20)
    plt.ylim(0, 100)
    
    # �������ϱ���ֵ
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_accuracy.png"), dpi=300)
    print("? Generated: overall_accuracy.png")

    # --- ͼ�� 2: Category Breakdown (ά����ϸ�Ա�) ---
    plt.figure(figsize=(12, 7))
    # ����ֻ�Ա� Baseline �����ǵ� Full Agent
    breakdown_df = df[df["Category"] != "Overall"]
    breakdown_df = breakdown_df[breakdown_df["Model"].isin(["Baseline", "Full Agent (Proposed)"])]
    
    ax2 = sns.barplot(x="Category", y="Accuracy (%)", hue="Model", data=breakdown_df, palette="Set2")
    plt.title("Performance Breakdown by Reasoning Dimensions", fontsize=16, pad=20)
    plt.ylim(0, 115) # ����ռ������
    plt.legend(loc='upper right')
    
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 8), 
                        textcoords = 'offset points',
                        fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "category_breakdown.png"), dpi=300)
    print("? Generated: category_breakdown.png")

if __name__ == "__main__":
    main()