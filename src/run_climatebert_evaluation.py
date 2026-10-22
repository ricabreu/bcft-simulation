# =========================================================================
# CLIMATEBERT BRAND COHERENCE EVALUATION PIPELINE
# =========================================================================
# This script fetches real data from Hugging Face ("climatebert/environmental_claims")
# and computes information-theoretic divergences (KL and Jensen-Shannon)
# to evaluate brand communication consistency against an ideal baseline.
#
# TO RUN THIS IN GOOGLE COLAB OR YOUR LOCAL ENVIRONMENT, INSTALL THESE FIRST:
# !pip install numpy pandas datasets scipy
# =========================================================================

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon

def calculate_kl_divergence(p, q):
    """
    Computes the Kullback-Leibler Divergence KL(P || Q) using log base 2.
    Handles zero values gracefully to prevent runtime crashes.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = p * np.log2(p / q)
        res[np.isnan(res)] = 0  # Converts NaN (0 * log(0)) to 0
        return np.sum(res)

def main():
    print("="*70)
    print("CLIMATEBERT ENVIRONMENTAL CLAIMS EVALUATION PIPELINE")
    print("="*70)
    
    # 1. FETCH REAL TARGET DATASET FROM HUGGING FACE
    print("Fetching the ClimateBERT environmental claims dataset from Hugging Face...")
    try:
        dataset = load_dataset("climatebert/environmental_claims", split="train")
        df = pd.DataFrame(dataset)
        print(f"Dataset successfully loaded. Total sample size: {len(df)} rows.")
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        print("Please check your internet connection or install 'datasets' package via: pip install datasets")
        return

    # 2. EXTRACT OBSERVED BRAND COMMUNICATION VECTOR (P_p)
    # Binary categories: 0 = Neutral/Financial, 1 = Explicit Environmental Claim
    category_proportions = df['label'].value_counts(normalize=True).sort_index()
    
    p_p = np.array([category_proportions[0], category_proportions[1]])
    
    print("\n[Observed Brand Distribution (P_p)]")
    print(f" - Neutral/Financial Focus: {p_p[0]:.4f} ({p_p[0]*100:.2f}%)")
    print(f" - Environmental Claims Focus: {p_p[1]:.4f} ({p_p[1]*100:.2f}%)")

    # 3. DEFINE INDEPENDENT BENCHMARK OR AUDITED AUDIT VECTOR (P*)
    # baseline scenario: 60% environmental mitigation, 40% financial data
    p_star = np.array([0.40, 0.60]) 

    print("\n[Audited Baseline/Ideal Distribution (P*)]")
    print(f" - Required Financial Focus: {p_star[0]:.4f} ({p_star[0]*100:.2f}%)")
    print(f" - Required Climate/Environmental Focus: {p_star[1]:.4f} ({p_star[1]*100:.2f}%)")

    # 4. COMPUTE MATHEMATICAL DIVERGENCES
    # Forward KL focusing on omission of crucial topics
    kl_divergence_val = calculate_kl_divergence(p_star, p_p)
    
    # Jensen-Shannon Divergence (squared distance to yield pure information divergence)
    js_distance_val = jensenshannon(p_star, p_p, base=2)
    js_divergence_val = js_distance_val ** 2
    
    # Brand Consistency Score mapping divergence onto a predictable 0 to 1 scale
    brand_consistency_score = 1 - js_divergence_val

    # 5. EXECUTE BRAND DIAGNOSTIC AND COHERENCE ASSESSMENT
    print("\n" + "="*70)
    print("DATASET BRAND METRIC EVALUATION SUMMARY")
    print("="*70)
    print(f"Kullback-Leibler Divergence KL(P* || P_p) : {kl_divergence_val:.4f}")
    print(f"Jensen-Shannon Divergence JSD(P* || P_p)  : {js_divergence_val:.4f}")
    print(f"Brand Consistency Metric C(p)             : {brand_consistency_score:.4f}")

    print("\n" + "-"*70)
    print("AUTOMATED REGULATORY DIAGNOSTIC (Default Threshold = 0.75):")
    if brand_consistency_score < 0.75:
        print("[CRITICAL WARNING]: High probability of Greenwashing or Strategic Negligence.")
        print("The observed communication strategy severely deviates from independent audit requirements.")
    else:
        print("[SUCCESS]: High Brand Consistency verified (under default threshold).")
        print("The observed corporate messaging matches its real-world obligations.")
    
    # Optional strict diagnostic
    print("\nAUTOMATED REGULATORY DIAGNOSTIC (Strict Threshold = 0.95):")
    if brand_consistency_score < 0.95:
        print("[CRITICAL WARNING]: Severe deviation detected under strict compliance rules.")
        print("The observed communication strategy underperforms relative to real obligations.")
    else:
        print("[SUCCESS]: High Brand Consistency verified under strict compliance rules.")
    print("-"*70 + "\n")

if __name__ == "__main__":
    main()
