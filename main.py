import sqlite3
con=sqlite3.connect(r'data\\processed\\hybrid_ingest_checkpoint.sqlite3')
rows=con.execute('select doc_id, updated_at from doc_status where status=''done'' and chunk_count=0 order by cast(doc_id as integer) limit 50').fetchall()
print('\n'.join(f'{r[0]} {r[1]}' for r in rows))
