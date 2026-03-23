#!/bin/bash
# Download RLHF Papers
# Generated: 2025-10-09
# Usage: bash download_papers.sh [category]
# Categories: all, foundational, surveys, priority

set -e  # Exit on error

BASE_DIR="/Users/pranjal/Code/ORE/papers/RLHF"

echo "📥 RLHF Papers Downloader"
echo "=========================="
echo ""

download_foundational() {
    echo "📚 Downloading Foundational Papers..."
    cd "$BASE_DIR/foundational/"

    wget -q --show-progress https://arxiv.org/pdf/1706.03741.pdf -O christiano_2017.pdf || echo "⚠️  Failed: Christiano 2017"
    wget -q --show-progress https://arxiv.org/pdf/2203.02155.pdf -O ouyang_2022_instructgpt.pdf || echo "⚠️  Failed: InstructGPT"
    wget -q --show-progress https://arxiv.org/pdf/2204.05862.pdf -O askell_2021_anthropic.pdf || echo "⚠️  Failed: Askell 2021"

    echo "✅ Foundational papers downloaded"
}

download_surveys() {
    echo "📖 Downloading Surveys..."
    cd "$BASE_DIR/surveys/"

    wget -q --show-progress https://arxiv.org/pdf/2312.14925.pdf -O kaufmann_2023_survey.pdf || echo "⚠️  Failed: Kaufmann survey"
    wget -q --show-progress https://arxiv.org/pdf/2404.08555.pdf -O abdelnabi_2024_deciphered.pdf || echo "⚠️  Failed: RLHF Deciphered"

    echo "✅ Surveys downloaded"
    echo "⚠️  Note: DPO survey requires manual download from Semantic Scholar"
}

download_alternatives() {
    echo "🔄 Downloading Alternatives to RL..."
    cd "$BASE_DIR/alternatives_to_RL/"

    wget -q --show-progress http://arxiv.org/pdf/2405.21046.pdf -O xpo_2024.pdf || echo "⚠️  Failed: XPO"

    echo "✅ Alternatives downloaded"
}

download_reward_modeling() {
    echo "🎯 Downloading Reward Modeling Papers..."
    cd "$BASE_DIR/reward_modeling/"

    wget -q --show-progress https://arxiv.org/pdf/2407.16574.pdf -O tlcr_2024.pdf || echo "⚠️  Failed: TLCR"
    wget -q --show-progress https://arxiv.org/pdf/2411.08302.pdf -O r3hf_2024.pdf || echo "⚠️  Failed: R3HF"
    wget -q --show-progress https://arxiv.org/pdf/2402.09401.pdf -O dense_reward_2024.pdf || echo "⚠️  Failed: Dense Reward"

    echo "✅ Reward modeling papers downloaded"
}

download_safety_risk() {
    echo "🛡️  Downloading Safety & Risk Papers..."
    cd "$BASE_DIR/safety_risk/"

    wget -q --show-progress https://arxiv.org/pdf/2410.23569.pdf -O ra_pbrl_2025.pdf || echo "⚠️  Failed: RA-PbRL"
    wget -q --show-progress https://arxiv.org/pdf/2310.12773.pdf -O safe_rlhf_2023.pdf || echo "⚠️  Failed: Safe RLHF"
    wget -q --show-progress https://arxiv.org/pdf/2406.15568.pdf -O robust_rlhf_2024.pdf || echo "⚠️  Failed: Robust RLHF"
    wget -q --show-progress https://arxiv.org/pdf/2503.22723.pdf -O zero_shot_llm_2025.pdf || echo "⚠️  Failed: Zero-shot LLM"

    echo "✅ Safety & risk papers downloaded"
}

download_efficiency() {
    echo "⚡ Downloading Efficiency & Scalability Papers..."
    cd "$BASE_DIR/efficiency_scalability/"

    wget -q --show-progress https://arxiv.org/pdf/2410.18252.pdf -O async_rlhf_2025.pdf || echo "⚠️  Failed: Async RLHF"
    wget -q --show-progress http://arxiv.org/pdf/2309.00267.pdf -O rlaif_2024.pdf || echo "⚠️  Failed: RLAIF"
    wget -q --show-progress https://arxiv.org/pdf/2402.00782.pdf -O active_queries_2024.pdf || echo "⚠️  Failed: Active Queries"
    wget -q --show-progress https://arxiv.org/pdf/2402.02423.pdf -O uni_rlhf_2024.pdf || echo "⚠️  Failed: Uni-RLHF"

    echo "✅ Efficiency papers downloaded"
}

download_personalization() {
    echo "👤 Downloading Personalization Papers..."
    cd "$BASE_DIR/personalization/"

    wget -q --show-progress https://arxiv.org/pdf/2408.10075.pdf -O poddar_2024.pdf || echo "⚠️  Failed: Poddar 2024"
    wget -q --show-progress https://arxiv.org/pdf/2503.19201.pdf -O p_sharelora_2025.pdf || echo "⚠️  Failed: P-ShareLoRA"
    wget -q --show-progress https://arxiv.org/pdf/2501.11463.pdf -O curiosity_driven_2025.pdf || echo "⚠️  Failed: Curiosity-Driven"

    echo "✅ Personalization papers downloaded"
}

download_multimodal() {
    echo "🖼️  Downloading Multimodal Papers..."
    cd "$BASE_DIR/multimodal/"

    wget -q --show-progress https://arxiv.org/pdf/2502.10391.pdf -O mm_rlhf_2025.pdf || echo "⚠️  Failed: MM-RLHF"

    echo "✅ Multimodal papers downloaded"
}

download_federated() {
    echo "🔐 Downloading Federated & Privacy Papers..."
    cd "$BASE_DIR/federated_privacy/"

    wget -q --show-progress http://arxiv.org/pdf/2412.15538.pdf -O fedrlhf_2025.pdf || echo "⚠️  Failed: FedRLHF"

    echo "✅ Federated papers downloaded"
}

download_priority() {
    echo "⭐ Downloading PRIORITY Papers Only..."
    echo ""

    # High priority papers for our Oxford article
    cd "$BASE_DIR/surveys/"
    echo "  → Kaufmann survey (already cited)..."
    wget -q --show-progress https://arxiv.org/pdf/2312.14925.pdf -O kaufmann_2023_survey.pdf || echo "⚠️  Failed"

    cd "$BASE_DIR/safety_risk/"
    echo "  → RA-PbRL (risk-aware extension)..."
    wget -q --show-progress https://arxiv.org/pdf/2410.23569.pdf -O ra_pbrl_2025.pdf || echo "⚠️  Failed"

    cd "$BASE_DIR/reward_modeling/"
    echo "  → TLCR (token-level continuous rewards)..."
    wget -q --show-progress https://arxiv.org/pdf/2407.16574.pdf -O tlcr_2024.pdf || echo "⚠️  Failed"

    cd "$BASE_DIR/efficiency_scalability/"
    echo "  → RLAIF (AI feedback scalability)..."
    wget -q --show-progress http://arxiv.org/pdf/2309.00267.pdf -O rlaif_2024.pdf || echo "⚠️  Failed"

    echo ""
    echo "✅ Priority papers downloaded"
    echo "⚠️  Note: DPO survey still requires manual download"
}

# Main script
case "${1:-priority}" in
    all)
        download_foundational
        download_surveys
        download_alternatives
        download_reward_modeling
        download_safety_risk
        download_efficiency
        download_personalization
        download_multimodal
        download_federated
        ;;
    foundational)
        download_foundational
        ;;
    surveys)
        download_surveys
        ;;
    alternatives)
        download_alternatives
        ;;
    reward)
        download_reward_modeling
        ;;
    safety)
        download_safety_risk
        ;;
    efficiency)
        download_efficiency
        ;;
    personalization)
        download_personalization
        ;;
    multimodal)
        download_multimodal
        ;;
    federated)
        download_federated
        ;;
    priority)
        download_priority
        ;;
    *)
        echo "Usage: $0 [category]"
        echo ""
        echo "Categories:"
        echo "  priority         Download high-priority papers only (default)"
        echo "  all              Download all papers"
        echo "  foundational     Christiano 2017, InstructGPT, Anthropic"
        echo "  surveys          Kaufmann, Abdelnabi, DPO survey"
        echo "  alternatives     XPO, DPO variants"
        echo "  reward           TLCR, R3HF, Dense Reward"
        echo "  safety           RA-PbRL, Safe RLHF, Robust RLHF"
        echo "  efficiency       Async RLHF, RLAIF, Active Queries"
        echo "  personalization  Poddar, P-ShareLoRA, Curiosity"
        echo "  multimodal       MM-RLHF"
        echo "  federated        FedRLHF"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Download complete!"
echo ""
echo "📂 Papers saved to: $BASE_DIR"
echo ""
echo "📋 Next steps:"
echo "  1. Review RLHF_PAPERS_INDEX.md for paper descriptions"
echo "  2. Check downloaded PDFs"
echo "  3. Read priority papers first (marked ⭐ in index)"
echo ""
