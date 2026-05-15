import pandas as pd
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
import numpy as np

df = pd.read_excel(r"D:\UVic\Masters 2.2\Health Econ 524\Final Paper\Data\Cloude\Raw\Combined_dataset_1.xlsx")

df['mortality_rate'] = (df['amenable_mortality_std'] / df['total_population']) * 100000

df['pct_65plus'] = (df['pct_65plus'] / df['total_population']) * 100

# ── Graph 1 — Mortality rate over time by province ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for province, group in df.groupby('Province'):
    ax.plot(group['Year'], group['mortality_rate'], marker='o', markersize=3, label=province)
ax.set_title('Amenable mortality rate per 100,000 by province (2001-2023)')
ax.set_xlabel('Year')
ax.set_ylabel('Mortality rate per 100,000')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('graph1_mortality_by_province.png', dpi=150)
plt.show()

# ── Graph 2 — Mortality rate vs physicians scatter ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
for province, group in df.groupby('Province'):
    ax.scatter(group['physicians_per100k'], group['mortality_rate'], label=province, s=20)
    for _, row in group.iterrows():
        ax.annotate(str(int(row['Year'])),
                   (row['physicians_per100k'], row['mortality_rate']),
                   fontsize=5,
                   ha='center',
                   va='bottom')
ax.set_title('Mortality rate vs physicians per 100,000')
ax.set_xlabel('Physicians per 100,000')
ax.set_ylabel('Mortality rate per 100,000')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('graph2_mortality_vs_physicians.png', dpi=150)
plt.show()

# ── Graph 3 — Mortality rate vs median income scatter ─────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
for province, group in df.groupby('Province'):
    ax.scatter(group['median_income'], group['mortality_rate'], label=province, s=20)
    for _, row in group.iterrows():
        ax.annotate(str(int(row['Year'])),
                   (row['median_income'], row['mortality_rate']),
                   fontsize=5,
                   ha='center',
                   va='bottom')
ax.set_title('Mortality rate vs median income')
ax.set_xlabel('Median income (2023 constant dollars)')
ax.set_ylabel('Mortality rate per 100,000')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('graph3_mortality_vs_income.png', dpi=150)
plt.show()

# ── Graph 4 — Average mortality rate by province ───────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
avg_mortality = df.groupby('Province')['mortality_rate'].mean().sort_values(ascending=False)
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(avg_mortality)))
avg_mortality.plot(kind='bar', ax=ax, color=colors)
ax.set_title('Which province has the most preventable deaths?')
ax.set_xlabel('Province')
ax.set_ylabel('Average mortality rate per 100,000')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('graph4_avg_mortality_by_province.png', dpi=150)
plt.show()

# ── Regression — set index AFTER graphs ────────────────────────────────────
df = df.set_index(['Province', 'Year'])
y = df['mortality_rate']
X = df[['physicians_per100k', 'median_income', 'pct_65plus', 'spending_percapita', 'education']]
model = PanelOLS(y, X, entity_effects=True, time_effects=True)
results = model.fit(cov_type='clustered', cluster_entity=True)
print(results.summary)
