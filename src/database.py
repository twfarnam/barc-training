import sqlite3

db_path = 'barc.db'
db = sqlite3.connect(db_path, check_same_thread=False)

def images():
    cursor = db.cursor()
    query = 'SELECT id FROM images WHERE deleted_at IS NULL' 
    result = cursor.execute(query).fetchall()
    return [ row[0] for row in result ]

def categories():
    cursor = db.cursor()
    query = '''
        SELECT categories.id, room || ' | ' || object, count(*)
        FROM categories
        LEFT JOIN images_categories ON category_id = categories.id
        WHERE image_id NOT IN
            (SELECT id FROM images WHERE deleted_at IS NOT NULL)
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

