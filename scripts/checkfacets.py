import requests
collection_name = "legal_chunks_hybrid_v1"
url = f"http://localhost:6333/collections/{collection_name}/facet"
fields = [
    "loai_van_ban_slug", 
    "co_quan_ban_hanh_slug", 
    "pham_vi_slug", 
    "tinh_trang_hieu_luc_slug", 
    "nganh_slug", 
    "linh_vuc_slug"
]
for field in fields:
    print(f"\n--- Top values for {field} ---")
    try:
        resp = requests.post(url, json={"key": field, "limit": 15})
        data = resp.json()
        if "result" in data and "hits" in data["result"]:
            for hit in data["result"]["hits"]:
                print(f"  {hit['value']}: {hit['count']}")
        else:
            print("No data or error:", data)
    except Exception as e:
        print("Error:", e)