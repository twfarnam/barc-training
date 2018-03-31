import sqlite3

db_path = 'barc.db'
db = sqlite3.connect(db_path, check_same_thread=False)

def count():
    cursor = db.cursor()
    return cursor.execute('SELECT count(*) FROM images').fetchone()[0]

def images():
    cursor = db.cursor()
    query = 'SELECT id FROM images' 
    result = cursor.execute(query).fetchall()
    return [ row[0] for row in result ]

def categories():
    cursor = db.cursor()
    query = '''
        SELECT id, room || ' | ' || object, count(*)
        FROM categories
        LEFT JOIN images_categories AS i ON i.category_id = categories.id
        GROUP BY categories.id
        ORDER BY room, object
    '''
    categories = cursor.execute(query).fetchall()
    ids = [ row[0] for row in categories ]
    labels = [ str(row[1]) for row in categories ]
    weights = [ row[2] for row in categories ]
    return ids, labels, weights

def cat_for_image(image_id, category_ids):
    cursor = db.cursor()
    query = 'SELECT category_id FROM images_categories WHERE image_id == "%s"'
    result = cursor.execute(query % image_id).fetchall()
    image_cats = [ row[0] for row in result ]
    return [ 1 if cat in image_cats else 0 for cat in category_ids ]

