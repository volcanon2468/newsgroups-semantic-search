
import requests
import json

BASE = "http://localhost:8000"

print("=== TEST 1: Health Check ===")
r = requests.get(f"{BASE}/health")
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

print("\n=== TEST 2: Cache Stats (empty) ===")
r = requests.get(f"{BASE}/cache/stats")
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

print("\n=== TEST 3: Query - Space exploration (cache miss) ===")
r = requests.post(f"{BASE}/query", json={"query": "What are the latest developments in space exploration?"})
print(f"Status: {r.status_code}")
data = r.json()
print(f"cache_hit: {data['cache_hit']}")
print(f"dominant_cluster: {data['dominant_cluster']}")
print(f"Results: {len(data['result'])} documents")
for i, res in enumerate(data["result"][:2]):
    doc_preview = res["document"][:80]
    print(f"  [{i}] {res['category']} (sim={res['similarity']}) - {doc_preview}...")

print("\n=== TEST 4: Query - Same topic rephrased (cache hit?) ===")
r = requests.post(f"{BASE}/query", json={"query": "Recent news about NASA and space missions"})
data = r.json()
print(f"cache_hit: {data['cache_hit']}")
print(f"matched_query: {data['matched_query']}")
print(f"similarity_score: {data['similarity_score']}")

print("\n=== TEST 4b: Query - Close rephrasing (should be cache hit) ===")
r = requests.post(f"{BASE}/query", json={"query": "What are the new developments in space exploration?"})
data = r.json()
print(f"cache_hit: {data['cache_hit']}")
print(f"matched_query: {data['matched_query']}")
print(f"similarity_score: {data['similarity_score']}")

print("\n=== TEST 5: Query - Different topic (guns/politics) ===")
r = requests.post(f"{BASE}/query", json={"query": "Gun control and firearms legislation debate"})
data = r.json()
print(f"cache_hit: {data['cache_hit']}")
print(f"dominant_cluster: {data['dominant_cluster']}")
print(f"Results: {len(data['result'])} documents")
for i, res in enumerate(data["result"][:2]):
    doc_preview = res["document"][:80]
    print(f"  [{i}] {res['category']} (sim={res['similarity']}) - {doc_preview}...")

print("\n=== TEST 6: Cache Stats (after queries) ===")
r = requests.get(f"{BASE}/cache/stats")
print(json.dumps(r.json(), indent=2))

print("\n=== TEST 7: Delete Cache ===")
r = requests.delete(f"{BASE}/cache")
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

print("\n=== TEST 8: Cache Stats (after delete) ===")
r = requests.get(f"{BASE}/cache/stats")
print(json.dumps(r.json(), indent=2))


print("\nAll endpoint tests completed successfully!")
