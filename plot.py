import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['TORGO Test Set (56 Files)', 'ANGELA Custom Set (10 Files)']
our_model_wer = [40.40, 32.94]      # Your Scores
openai_wer = [57.58, 36.47]         # OpenAI Medium Scores (Using the better 57% run)

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Bars
rects1 = ax.bar(x - width/2, openai_wer, width, label='OpenAI Whisper Medium (Standard)', color='#ff9999')
rects2 = ax.bar(x + width/2, our_model_wer, width, label='Your Fine-Tuned Model', color='#99ff99')

# Styling
ax.set_ylabel('Word Error Rate (Lower is Better)')
ax.set_title('Final Benchmarking: Standard vs. Fine-Tuned Model')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.set_ylim(0, 80)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('final_victory_chart.png')
print("Chart saved as 'final_victory_chart.png'")
plt.show()