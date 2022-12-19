# Databricks notebook source
# MAGIC %pip install bamboo

# COMMAND ----------

import pandas as pd
salaries = pd.read_csv("data/salary_data_cleaned.csv")
cost_of_living = pd.read_csv("data/cost-of-living.csv")

# COMMAND ----------

salaries.head()

# COMMAND ----------

s = salaries.loc[:, ["Job Title", "min_salary", "max_salary", "avg_salary", "Location", "Size", "Type of ownership", "Industry", "Revenue"]]
s["Location"] = s["Location"].str.split(",", 0, expand=True)[0]
s = s.loc[:, ["Job Title", "Location", "min_salary", "max_salary", "avg_salary", "Industry"]]
s

# COMMAND ----------

data = s.merge(cost_of_living, left_on="Location", right_on="city")
data.drop(["Location", "Unnamed: 0"], axis="columns", inplace=True)
data.dropna(axis=0, inplace=True)
data["total_cost_of_living"] = data.iloc[:, 7:-1].sum(axis=1)

# COMMAND ----------

df = data.groupby("city").apply(
    lambda df: pd.DataFrame(
        {
            "job": df["Job Title"],
            "industry": df["Industry"],
            "city": df["city"],
            "country": df["country"],
            "cost_of_living": df["total_cost_of_living"],
            "min_salary": df["min_salary"],
            "max_salary": df["max_salary"],
            "avg_salary": df["avg_salary"],
            
        }
    )
).drop_duplicates()
df["ratio"] = (df["min_salary"] / df["cost_of_living"])*100
df

# COMMAND ----------

top_ratio = df.sort_values(by=["ratio"], ascending=[False]).drop_duplicates("city").iloc[:10]
top_ratio

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
new_df = top_ratio.loc[:, ["ratio", "city"]]
x = np.arange(len(new_df.ratio))
width = 0.75

# new_df.columns
values = new_df.ratio
labels = new_df.city

da_rects = ax.bar(x, values, width, edgecolor="black")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Ratio')
ax.set_title('Best cities for programmers')
ax.set_xticks(ticks=x)
ax.set_xticklabels(labels, rotation = -45)
ax.legend()

plt.show()

# COMMAND ----------

top_paid_cities = df.sort_values(by="min_salary", ascending=False).drop_duplicates("city", keep="first")
new_df = pd.pivot_table(top_paid_cities, index=["city"]).sort_values(by="ratio", ascending=False).iloc[:5]
new_df.drop(["cost_of_living", "ratio"], inplace=True, axis=1)
new_df

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.arange(len(new_df.columns))
width = 0.1

# new_df.columns
chicago_values = new_df.values[0]
washington_values = new_df.values[1]
brisbane_values = new_df.values[2]
ann_arbor_values = new_df.values[3]
bellevue_values = new_df.values[4]

da_rects = ax.bar(x, chicago_values, width, label='Chicago', edgecolor="black")
de_rects = ax.bar(x + width, washington_values, width, label='Washington', edgecolor="black")
ds_rects = ax.bar(x + 2 * width, brisbane_values, width, label='Brisbane', edgecolor="black")
sc_rects = ax.bar(x + 3 * width, ann_arbor_values, width, label='Ann Arbor', edgecolor="black")
o_rects = ax.bar(x + 4 * width, bellevue_values, width, label='Bellevue', edgecolor="black")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Salary')
ax.set_title('Salary per city')
ax.set_xticks(ticks=[y + 2 * width for y in x])
ax.set_xticklabels(["Average", "Maximum", "Minimum"])
ax.legend()

plt.show()

# COMMAND ----------


def map_values(df):
    job = df['job'].lower()
    if "data science" in job or "data scientist" in job:
        return "data science"
    elif "data analyst" in job or "analytics" in job:
        return "data analyst"
    elif "data engineer" in job:
        return "data engineer"
    elif "scientist" in job:
        return "scientist"
    return "other"
top_paid_job_types = df
top_paid_job_types["job_types"] = df.apply(map_values, axis=1)
top_paid_job_types = df.sort_values(by="min_salary", ascending=False)#.drop_duplicates("job_types", keep="first")
new_df = pd.pivot_table(top_paid_job_types, index=["job_types"])
new_df.drop(["cost_of_living", "ratio"], inplace=True, axis=1)
new_df

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

data_analysts = new_df.values[0]
data_engineer = new_df.values[1]
data_science = new_df.values[2]
other = new_df.values[3]
scientist = new_df.values[4]

new_df.values

x = np.arange(len(new_df.columns))
width = 0.1

fig, ax = plt.subplots()
fig
da_rects = ax.bar(x, data_analysts, width, label='Data Analysts', edgecolor="black")
de_rects = ax.bar(x + width, data_engineer, width, label='Data Engineers', edgecolor="black")
ds_rects = ax.bar(x + 2 * width, data_science, width, label='Data Sciencs', edgecolor="black")
sc_rects = ax.bar(x + 3 * width, scientist, width, label='Scientists', edgecolor="black")
o_rects = ax.bar(x + 4 * width, other, width, label='Other', edgecolor="black")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Money per group')
ax.set_xticks(ticks=[y + 2 * width for y in x])
ax.set_xticklabels([x.replace("_", " ").title() for x in new_df.columns])
ax.legend()

plt.show()

