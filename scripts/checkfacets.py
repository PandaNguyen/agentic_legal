import requests
collection_name = "legal_chunks_hybrid_v1"
url = f"http://localhost:6333/collections/{collection_name}/facet"
fields = [
    "ngay_ban_hanh_ts",
    "effective_year",
    "so_ky_hieu_slug",
    "ngay_het_hieu_luc_ts",
    "expiry_year",
    "loai_van_ban",
    "so_ky_hieu",
    "nganh_slug",
    "issue_year",
    "nganh",
    "pham_vi_slug",
    "linh_vuc_slug",
    "pham_vi",
    "tinh_trang_hieu_luc",
    "linh_vuc",
    "loai_van_ban_slug",
    "tinh_trang_hieu_luc_slug",
    "co_quan_ban_hanh",
    "ngay_co_hieu_luc_ts",
    "co_quan_ban_hanh_slug"
]
for field in fields:
    print(f"\n--- Top values for {field} ---")
    try:
        resp = requests.post(url, json={"key": field, "limit": 30})
        data = resp.json()
        if "result" in data and "hits" in data["result"]:
            for hit in data["result"]["hits"]:
                print(f"  {hit['value']}: {hit['count']}")
        else:
            print("No data or error:", data)
    except Exception as e:
        print("Error:", e)