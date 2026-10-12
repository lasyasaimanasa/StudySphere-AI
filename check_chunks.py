import sqlite3
from config import Config

connection = sqlite3.connect(Config.DATABASE_PATH)
connection.row_factory = sqlite3.Row

materials = connection.execute("""
    SELECT id, filename, processing_status
    FROM materials
    ORDER BY id
""").fetchall()

print("\n=== UPLOADED MATERIALS ===")
for material in materials:
    print(
        f"\nID: {material['id']} | "
        f"File: {material['filename']} | "
        f"Status: {material['processing_status']}"
    )

    chunks = connection.execute("""
        SELECT chunk_index, page_number, content
        FROM document_chunks
        WHERE material_id = ?
        ORDER BY chunk_index
    """, (material["id"],)).fetchall()

    print(f"Chunks: {len(chunks)}")

    for chunk in chunks:
        print(f"\n  Chunk {chunk['chunk_index']} | Page {chunk['page_number']}")
        print(f"  {chunk['content'][:300]}")

connection.close()