import pandas as pd

meta_data = pd.read_csv("data\\metadata.csv")
print(meta_data.head())

content = pd.read_csv("data\\content.csv",encoding="utf-8" )
content = content.tail(1000)  # Load only the first 1000 rows for preview
content.to_csv("data\\content_preview.csv", encoding="utf-8", index=False)  # Save the preview to a new CSV file
print(content.head())

relationships = pd.read_csv("data\\relationships.csv",encoding="utf-8")
print(relationships.head())
