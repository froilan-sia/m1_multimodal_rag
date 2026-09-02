#!/usr/bin/env python3
"""
Generate a synthetic PDF with a chart for testing hallucinations.
Run: python scripts/generate_test_pdf.py

This creates data/synthetic_chart.pdf which can be ingested with:
    python main.py ingest --pdf data/synthetic_chart.pdf
"""

import matplotlib.pyplot as plt
from pathlib import Path
import pymupdf as fitz
import tempfile
import os

def create_chart_pdf():
    """Generate a PDF with a quarterly revenue chart."""
    
    # ---------- 1. Create the chart using matplotlib ----------
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    revenue = [10.2, 11.5, 12.4, 11.8]  # Peak is Q3
    labels = [f"${v}M" for v in revenue]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(quarters, revenue, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, label in zip(bars, labels):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title('Quarterly Revenue 2024', fontsize=16, fontweight='bold')
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Revenue ($M)', fontsize=12)
    plt.ylim(0, 15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        chart_path = tmp.name
        plt.savefig(chart_path, dpi=150)
        plt.close()
    
    # ---------- 2. Create PDF with the chart ----------
    pdf_path = Path("./data/synthetic_chart.pdf")
    pdf_path.parent.mkdir(exist_ok=True)
    
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter size
    
    # Insert the chart image (full page)
    rect = fitz.Rect(50, 50, 562, 700)  # Position and size
    page.insert_image(rect, filename=chart_path)
    
    doc.save(str(pdf_path))
    doc.close()
    
    os.unlink(chart_path)
    
    # Print ground truth to console
    print("✅ Synthetic PDF created:", pdf_path)
    print("📄 Ingest it with: python main.py ingest --pdf", pdf_path)
    print("\n📊 Ground Truth for this chart:")
    print("   - Highest revenue: Q3 at $12.4 million")
    print("   - Trend: Increase from Q1 to Q3, then slight decline in Q4")
    print("\n⚠️ The VLM will likely hallucinate on this chart, providing incorrect quarter and values.")

if __name__ == "__main__":
    create_chart_pdf()
