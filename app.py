#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import matplotlib.pyplot as plt
from model_utils import run_experiment

# --- Streamlit Page Setup ---
st.set_page_config(page_title="ANN vs SNN Energy Dashboard", layout="wide")

# --- Title ---
st.title("⚡ ANN vs SNN Energy and Performance Comparison")
st.markdown("Compare neural efficiency and performance across Artificial and Spiking Neural Networks.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📤 Upload your dataset (CSV file)", type=["csv"])

if uploaded_file:
    with st.spinner("Training models and evaluating performance... ⏳"):
        results_df = run_experiment(uploaded_file)

    st.success("✅ Experiment completed successfully!")

    # --- Show Results Table ---
    st.subheader("📊 Metrics Summary")
    st.dataframe(results_df, use_container_width=True)

    # --- Layout Columns ---
    col1, col2 = st.columns(2)

    # ---------------------- #
    #  LEFT: Performance Metrics
    # ---------------------- #
    with col1:
        st.subheader("📈 Model Performance Metrics")

        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.ravel()

        for i, m in enumerate(metrics):
            axes[i].bar(results_df["Model"], results_df[m],
                        alpha=0.85, color="#3498db", width=0.4)
            axes[i].set_title(m)
            axes[i].set_ylim(0, 1)
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis='x', rotation=25)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ---------------------- #
    #  RIGHT: Energy Consumption
    # ---------------------- #
    with col2:
        st.subheader("⚙️ Energy Consumption")

        # Log-scale plot for better dynamic range
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = ["#4CAF50", "#FF9800"]
        bars = ax2.bar(results_df["Model"], results_df["Energy"],
                       color=colors, alpha=0.85, width=0.5)

        ax2.set_yscale("log")
        ax2.set_ylabel("Energy (Relative Units, log scale)")
        ax2.set_title("Energy Comparison (Log Scale)")
        ax2.tick_params(axis='x', rotation=30)

        # Annotate bars with readable text
        for bar in bars:
            height = bar.get_height()
            y_offset = height * 1.5 if height < 1000 else height * 1.1
            fontsize = 11 if height > 100 else 12
            ax2.text(bar.get_x() + bar.get_width() / 2, y_offset,
                     f"{int(height):,}", ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    # --- Footer ---
    st.markdown("---")
    st.caption("Developed by **Sri Krishna Vardhan** • Neural Efficiency at a Glance ⚡")
